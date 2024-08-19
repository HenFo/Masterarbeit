import json
import math
import os
import sys
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    LlamaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from accelerate.utils import broadcast_object_list

sys.path.append("/home/fock/code/MultiModalInstructERC")
from utils.dataset import ERCDataset, IemocapDataset

import argparse

import torch.nn as nn
from peft import LoraConfig

from utils import (
    MeldDataset,
    MmLlama,
    MmLlamaConcat,
    MmLlamaConfig,
    MmLlamaMerge,
    MmLlamaProcessor,
    SequenceClassificationCollator,
)

# LANGUAGE_MODEL = "/home/fock/code/MultiModalInstructERC/models/language/LLaMA2"
# LORA_ADAPTER = "/home/fock/code/MultiModalInstructERC/models/language/adapter/InstructERC_unbalanced"
# ACOUSTIC_MODEL = "/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
# OUTPUT_PATH = "/home/fock/code/MultiModalInstructERC/experiments/multimodal/mlp/concat/"
# DS_TRAIN_PATH = "/home/fock/code/MultiModalInstructERC/datasets/meld/train_sent_emo.csv"
# DS_DEV_PATH = "/home/fock/code/MultiModalInstructERC/datasets/meld/dev_sent_emo.csv"
# DS_TEST_PATH = "/home/fock/code/MultiModalInstructERC/datasets/meld/test_sent_emo.csv"


@dataclass()
class Args:
    evaluation: bool = False
    llm_id: str = None
    adapter_id: str = None
    acoustic_id: str = None
    output_path: str = None
    checkpoint_path: str = None
    train_dataset: str = None
    dev_dataset: str = None
    test_dataset: str = None
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    task: str = "normal"
    stage: int = 1
    batch_size: int = 3
    eval_batch_size: int = 1
    deepspeed_config: str = None
    lr: float = 2e-5
    epochs: int = 15
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0
    weight_decay: float = 0
    train_llm: bool = False
    lora_dim: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_module_name: str = ".*?llama.*?[qkvo]_proj"
    resume_training: bool = False
    window_size: int = 5
    time_till_aux: int = epochs
    do_auxilary_task: bool = False
    alpha: float = 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--llm_id", type=str, default=None)
    parser.add_argument("--adapter_id", type=str, default=None)
    parser.add_argument("--acoustic_id", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--dev_dataset", type=str, default=None)
    parser.add_argument("--test_dataset", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--task", type=str, default="normal")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_llm", action="store_true")
    parser.add_argument("--lora_dim", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_module_name", type=str, default=".*?[qkvo]_proj")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--time_till_aux", type=int, default=15)
    parser.add_argument("--do_auxilary_task", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    return Args(**vars(args))


args = parse_args()

print(args)


# args = Args(
#     batch_size=2,
#     gradient_accumulation_steps=16,
#     llm_id=LANGUAGE_MODEL,
#     acoustic_id=ACOUSTIC_MODEL,
#     adapter_id=LORA_ADAPTER,
#     output_path=OUTPUT_PATH,
#     checkpoint_path="/home/fock/code/MultiModalInstructERC/experiments/multimodal/mlp/concat/interpolate/stage_1",
#     train_dataset=DS_TRAIN_PATH,
#     test_dataset=DS_TEST_PATH,
#     dev_dataset=DS_DEV_PATH,
#     task="normal",
#     deepspeed_config="deepspeed_config.json",
#     epochs=7,
#     lr=1e-5,
#     train_llm=True,
#     resume_training=False,
#     stage=1
# )


def get_grouped_parameters(model):
    no_decay = ["bias", "norm.weight"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return grouped_parameters


def get_scheduler(
    optimizer,
    len_dataset,
    batch_size,
    gradient_accumulation_steps,
    epochs,
    min_lr_frac=0.5,
):
    num_steps = math.ceil(len_dataset / (batch_size * gradient_accumulation_steps))
    num_steps *= epochs
    warmup_steps = math.ceil(num_steps * args.warmup_ratio)
    def cosine_decay_with_warmup(step):
        if step < warmup_steps:
            return float(step) / max(1.0, warmup_steps)

        progress = float(step - warmup_steps) / max(1, num_steps - warmup_steps)
        cos_progress = 0.5 * (1 + math.cos(math.pi * progress))
        return cos_progress + (1 - cos_progress) * min_lr_frac

    scheduler = LambdaLR(optimizer, lr_lambda=cosine_decay_with_warmup)

    return scheduler


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<audio>", "</audio>"]})
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_for_stage(model: nn.Module, stage: int):
    if stage == 1:
        return _load_model_for_stage_1(model)
    elif stage == 2:
        return _load_model_for_stage_2(model)
    else:
        raise ValueError("Invalid stage number")


def _load_model_for_stage_1(model: MmLlamaMerge):
    """Load model for stage 1 training (Projector training)"""
    model.freeze_encoder(train_norm=True)
    model.freeze_scaling()
    return model


def _load_model_for_stage_2(model: MmLlamaMerge):
    model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_path, "best_model.pth")), strict=False
    )
    lora_config = LoraConfig(
        r=args.lora_dim,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_module_name,
        bias="none",
    )
    model = model.apply_training_lora(lora_config)
    model.aux_scalar = 1 / args.epochs
    model.freeze_scaling()
    model.freeze_projector()

    return model


def load_model_for_test(model: MmLlama):
    model.load_state_dict(
        torch.load(os.path.join(args.output_path, "best_model.pth")), strict=False
    )
    model = model.apply_inference_lora(args.output_path)
    return model.cuda()


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def setup_config_and_processor():
    # Load configurations
    llm_config = AutoConfig.from_pretrained(args.llm_id)
    ac_config = AutoConfig.from_pretrained(args.acoustic_id)
    ac_processor = AutoProcessor.from_pretrained(args.acoustic_id)

    # setup of tokenizer
    tokenizer = load_tokenizer()

    # setup of processor
    processor = MmLlamaProcessor(ac_processor, tokenizer)

    ## setup of config
    audio_start_token_id = tokenizer.additional_special_tokens_ids[0]
    audio_end_token_id = tokenizer.additional_special_tokens_ids[1]
    config = MmLlamaConfig(
        llm_config=llm_config,
        audio_config=ac_config,
        audio_token_id=audio_start_token_id,
        audio_end_token_id=audio_end_token_id,
        pad_token_id=tokenizer.pad_token_id,
        llm_pretrained_adapter=args.adapter_id,
    )

    return tokenizer, processor, config


def dataset_class(dataset_path: str) -> ERCDataset:
    if "meld" in dataset_path:
        return MeldDataset
    if "iemocap" in dataset_path:
        return IemocapDataset
    else:
        raise ValueError("Invalid dataset path")


def train():
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    tokenizer, processor, config = setup_config_and_processor()

    ## setup datasets
    train_dataset = dataset_class(args.train_dataset)(
        args.train_dataset,
        mode="train",
        task=args.task,
        window=args.window_size,
        audio_placement="enclose",
    )
    eval_dataset = dataset_class(args.dev_dataset)(
        args.dev_dataset,
        mode="test",  # for iemocap
        task="normal",
        window=args.window_size,
        audio_placement="enclose",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=SequenceClassificationCollator(processor, mode="train"),
    )
    # get model
    model = MmLlamaMerge(config, train_llm=args.train_llm, alpha=args.alpha)
    model = load_model_for_stage(model, args.stage)

    # setup optimizer
    grouped_parameters = get_grouped_parameters(model)
    optimizer = AdamW(grouped_parameters, lr=args.lr, betas=(0.9, 0.95))
    lr_scheduler: LambdaLR = get_scheduler(
        optimizer,
        len(train_dataset),
        args.batch_size,
        args.gradient_accumulation_steps,
        args.epochs,
        args.min_lr_ratio,
    )

    # setup accelerator
    (model, optimizer, lr_scheduler, train_dataloader) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    accelerator.register_for_checkpointing(lr_scheduler)

    # training loop
    best_eval_loss = float("inf")
    train_losses = []
    eval_losses = []

    start_epoch = 0

    checkpoint_path = os.path.join(args.output_path, "checkpoint")
    if (
        os.path.exists(os.path.join(checkpoint_path, "checkpoint_metadata.bin"))
        and args.resume_training
    ):
        print("############## Loading checkpoint ##############")
        (
            model,
            start_epoch,
            best_eval_loss,
            train_losses,
            eval_losses,
        ) = load_checkpoint(accelerator, model, checkpoint_path)


    for epoch in range(max(0, start_epoch - 1), args.epochs):
        epoch += 1
        model.train()
        batch_iterator = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch}",
        )
        running_loss = 0
        for step, batch in batch_iterator:
            try:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    running_loss += loss.item()

                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()

                batch_iterator.set_description(
                    f"Epoch: {epoch} / {args.epochs}, Current Loss: {loss.item():.4f}"
                )
            except torch.cuda.OutOfMemoryError:
                print("OutOfMemoty error on step", step, "skipping step")
                print("Text size", batch["text"]["input_ids"].size())
                print("Audio size", batch["acoustic"]["input_values"].size())
                torch.cuda.empty_cache()

        running_loss /= len(train_dataloader)
        train_losses.append(running_loss)

        # evaluation
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=SequenceClassificationCollator(
                processor, mode="train"
            ),  # mode=train because we test performance using loss,  not f1
            sampler=SequentialSampler(eval_dataset),
        )

        if accelerator.is_main_process:
            eval_loss = evaluate(accelerator, model, epoch, eval_dataloader)
            eval_losses.append(eval_loss)
            if eval_loss < best_eval_loss:
                print("Saving model")
                best_eval_loss = eval_loss
                print(f"Best Loss: {best_eval_loss}")
                save_model(accelerator, tokenizer, model)
                print("model saved")

            with open(os.path.join(args.output_path, "train_losses.json"), "wt") as f:
                json.dump(train_losses, f)
            with open(os.path.join(args.output_path, "eval_losses.json"), "wt") as f:
                json.dump(eval_losses, f)

        objects_to_broadcast = [eval_losses, best_eval_loss]
        broadcast_object_list(objects_to_broadcast)
        accelerator.wait_for_everyone()
        eval_losses, best_eval_loss = objects_to_broadcast

        if running_loss == math.nan:
            print("nan loss, stopping training")
            break

        print("###### Saving checkpoint ######")
        save_checkpoint(
            accelerator,
            model,
            epoch,
            best_eval_loss,
            train_losses,
            eval_losses,
            checkpoint_path=checkpoint_path,
        )
        accelerator.wait_for_everyone()

        if args.do_auxilary_task:
            set_auxilary_changes(
                model=accelerator.unwrap_model(model),
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                epoch=epoch,
                best_loss=best_eval_loss,
                eval_losses=eval_losses,
            )


def set_auxilary_changes(**kwargs):
    if args.stage == 1:
        _set_stage_1_changes(**kwargs)
    elif args.stage == 2:
        _set_stage_2_changes(**kwargs)


def _set_stage_1_changes(model, best_loss: float, eval_losses: list[float], **_):
    pass


def _set_stage_2_changes(model: MmLlamaMerge, epoch, **_):
    if epoch == 10:
        print("Unfreezing scaling and projector")
        model.unfreeze_scaling()
        model.unfreeze_projector()

    model.aux_scalar = min(1, epoch / 10)
    print("Using aux_scalar of", model.aux_scalar)


def save_model(accelerator: Accelerator, tokenizer, model):
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(
        unwrapped_model.state_dict(modules=["projector", "alpha"]),
        os.path.join(args.output_path, "best_model.pth"),
    )

    if args.train_llm:
        model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)


def save_checkpoint(
    accelerator: Accelerator,
    model,
    epoch,
    best_eval_loss,
    train_losses,
    eval_losses,
    checkpoint_path="checkpoint",
):
    create_folder_if_not_exists(checkpoint_path)
    accelerator.save_state(checkpoint_path, exclude_frozen_parameters=True)
    if accelerator.is_main_process:
        accelerator.save(
            {
                "epoch": epoch,
                "best_eval_loss": best_eval_loss,
                "train_losses": train_losses,
                "eval_losses": eval_losses,
            },
            os.path.join(checkpoint_path, "checkpoint_metadata.bin"),
        )
        if args.train_llm:
            model.save_pretrained(checkpoint_path)


def load_checkpoint(accelerator: Accelerator, model: MmLlama, checkpoint_path: str):
    accelerator.load_state(checkpoint_path, load_module_strict=False)
    checkpoint = torch.load(os.path.join(checkpoint_path, "checkpoint_metadata.bin"))
    if args.train_llm:
        model = accelerator.unwrap_model(model)
        model.apply_training_lora(adapter_id=checkpoint_path, resume_training=True)
        model = accelerator.prepare_model(model)

    epoch = checkpoint["epoch"]
    best_eval_loss = checkpoint["best_eval_loss"]
    train_losses = checkpoint["train_losses"]
    eval_losses = checkpoint["eval_losses"]

    return (
        model,
        epoch,
        best_eval_loss,
        train_losses,
        eval_losses,
    )


def prepare_batch(batch: dict[str, dict[str, torch.Tensor]]):
    for k in batch:
        for kk in batch[k]:
            batch[k][kk] = batch[k][kk].cuda()
    return batch


def test():
    tokenizer, processor, config = setup_config_and_processor()

    model = MmLlamaMerge(config)
    model = load_model_for_test(model)

    ## setup datasets
    test_dataset = dataset_class(args.test_dataset)(
        args.test_dataset,
        mode="test",
        audio_placement="enclose",
        task="normal",
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=SequenceClassificationCollator(processor, mode="dev"),
    )
    # get model
    evaluate_f1(tokenizer, model, test_dataloader)


def evaluate(
    accelerator: Accelerator,
    model: MmLlamaConcat,
    epoch: int,
    dataloader: DataLoader,
) -> float:
    dataloader = accelerator.prepare(dataloader)
    eval_batch_iterator = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    running_loss = 0
    model.eval()
    for step, batch in enumerate(eval_batch_iterator):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs["loss"]
            running_loss += loss.item()
    running_loss /= len(dataloader)
    print(f"Loss in Epoch {epoch}: {running_loss}")
    return running_loss


def evaluate_f1(
    tokenizer: LlamaTokenizerFast,
    model: MmLlamaConcat,
    dataloader: DataLoader,
):
    eval_batch_iterator = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    all_targets = []
    all_preds = []
    all_inputs = []
    model.eval()
    for inputs, labels in eval_batch_iterator:
        with torch.no_grad():
            try:
                inputs = prepare_batch(inputs)
                preds = model.generate(**inputs)
            except TimeoutError:
                print("TimeoutError on input", inputs)
                preds = torch.zeros((inputs["text"]["input_ids"].size(0), 1))

        all_preds.extend(preds.cpu())
        all_targets.extend(labels)
        all_inputs.extend(inputs["text"]["input_ids"].cpu())

    all_preds = all_preds[: len(dataloader.dataset)]
    all_targets = all_targets[: len(dataloader.dataset)]
    all_inputs = all_inputs[: len(dataloader.dataset)]

    all_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    all_inputs = tokenizer.batch_decode(all_inputs, skip_special_tokens=True)

    # clean predictions
    def clean_pred(input: str, pred: str):
        pred = pred.replace(input, "")
        pred = pred.strip()
        return pred

    all_preds = list(map(clean_pred, all_inputs, all_preds))
    f1 = f1_score(all_targets, all_preds, average="weighted")
    preds_for_eval = []
    for i, (inp, pred, target) in enumerate(zip(all_inputs, all_preds, all_targets)):
        preds_for_eval.append(
            {
                "index": i,
                "input": inp,
                "output": pred,
                "target": target,
            }
        )
    with open(os.path.join(args.output_path, "preds_test.json"), "wt") as f:
        json.dump(preds_for_eval, f)

    print(f"F1 in Test: {f1}")
    return f1


if __name__ == "__main__":
    if args.evaluation:
        test()
    else:
        create_folder_if_not_exists(args.output_path)
        train()
