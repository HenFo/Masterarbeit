import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Type

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
    set_seed,
)
from accelerate.utils import broadcast_object_list

sys.path.append("/home/fock/code/MultiModalInstructERC")

from utils.model import (
    MmLlama,
    MmLlamaConfig,
    MmLlamaForSequenceClassification,
)
from utils.processor import MmLlamaProcessor
from utils.collator import SequenceClassificationCollator
from utils.dataset import ERCDataset, IemocapDataset, MeldDataset

import argparse

import torch.nn as nn
from peft import LoraConfig


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
    lr: float = 2e-5
    epochs: int = 15
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0
    weight_decay: float = 0
    resume_training: bool = False
    window_size: int = 5
    time_till_aux: int = epochs
    do_auxiliary_task: bool = False
    alpha: float = 1.0
    seed: int = 42
    ignore_text: bool = False
    ignore_audio: bool = False


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
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--time_till_aux", type=int, default=15)
    parser.add_argument("--do_auxiliary_task", action="store_true")
    parser.add_argument("--ignore_text", action="store_true")
    parser.add_argument("--ignore_audio", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return Args(**vars(args))


args = parse_args()

print(args)

set_seed(args.seed)


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
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_for_stage(
    model: nn.Module, stage: int
) -> tuple[
    MmLlamaForSequenceClassification,
    Callable[[MmLlamaForSequenceClassification], MmLlamaForSequenceClassification],
]:
    if stage == 1:
        return _load_model_for_stage_1(model)
    elif stage == 2:
        return _load_model_for_stage_2(model)
    elif stage == 3:
        return _load_model_for_stage_3(model)
    else:
        raise ValueError("Invalid stage number")


def _load_model_for_stage_1(model: MmLlamaForSequenceClassification):
    model.ignore_text = True

    def execute_after_prepare(model: MmLlamaForSequenceClassification):
        model.freeze_encoder(train_norm=False)
        return model

    return model, execute_after_prepare


def _load_model_for_stage_2(model: MmLlamaForSequenceClassification):
    model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_path, "best_model.pth")), strict=False
    )

    model.ignore_acoustic = True

    def execute_after_prepare(model: MmLlamaForSequenceClassification):
        model.freeze_encoder(train_norm=False)
        return model

    return model, execute_after_prepare


def _load_model_for_stage_3(model: MmLlamaForSequenceClassification):
    state_dict = torch.load(os.path.join(args.checkpoint_path, "best_model.pth"))
    state_dict["classifier.weight"] = (
        state_dict["text_projector.classifier.weight"]
        + state_dict["audio_projector.classifier.weight"]
    ) / 2
    model.load_state_dict(state_dict, strict=False)

    def execute_after_prepare(model: MmLlamaForSequenceClassification):
        model.freeze_encoder(train_norm=False)
        return model

    return model, execute_after_prepare



def load_model_for_test(model: MmLlamaForSequenceClassification):
    model.load_state_dict(
        torch.load(os.path.join(args.output_path, "best_model.pth")), strict=False
    )

    model.ignore_text = args.ignore_text
    model.ignore_acoustic = args.ignore_audio

    return model.cuda()


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def setup_config_and_processor(dataset: ERCDataset):
    # Load configurations
    llm_config = AutoConfig.from_pretrained(args.llm_id)
    ac_config = AutoConfig.from_pretrained(args.acoustic_id)
    ac_processor = AutoProcessor.from_pretrained(args.acoustic_id)

    # setup of tokenizer
    tokenizer = load_tokenizer()

    # setup of processor
    processor = MmLlamaProcessor(ac_processor, tokenizer)

    # setup of config
    config = MmLlamaConfig(
        llm_config=llm_config,
        audio_config=ac_config,
        pad_token_id=tokenizer.pad_token_id,
        llm_pretrained_adapter=args.adapter_id,
        num_labels=len(dataset.emotions),
    )

    return tokenizer, processor, config


def dataset_class(dataset_path: str) -> Type[ERCDataset]:
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


    ## setup datasets
    train_dataset = dataset_class(args.train_dataset)(
        args.train_dataset,
        mode="train",
        task=args.task,
        window=args.window_size,
        audio_placement="none",
    )
    eval_dataset = dataset_class(args.dev_dataset)(
        args.dev_dataset,
        mode="test",  # for iemocap
        task="normal",
        window=args.window_size,
        audio_placement="none",
    )

    tokenizer, processor, config = setup_config_and_processor(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=SequenceClassificationCollator(processor, train_dataset),
    )

    # get model
    model = MmLlamaForSequenceClassification(config)
    model, execute_after_prepare = load_model_for_stage(model, args.stage)

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

    model = execute_after_prepare(model)

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
                    loss = outputs.loss
                    main_loss = outputs.main_loss.item()
                    running_loss += main_loss

                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()

                batch_iterator.set_description(
                    f"Epoch: {epoch} / {args.epochs}, Current Main-Loss: {main_loss:.4f}"
                )
            except torch.cuda.OutOfMemoryError:
                print("OutOfMemoty error on step", step, "skipping step")
                print("Text size", batch["text"]["input_ids"].size())
                print("Audio size", batch["acoustic"]["input_values"].size())
                torch.cuda.empty_cache()

        running_loss /= len(train_dataloader)
        train_losses.append(running_loss)

        # evaluation

        if accelerator.is_main_process:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=SequenceClassificationCollator(processor, eval_dataset),
                sampler=SequentialSampler(eval_dataset),
            )

            eval_loss, eval_gate_loss = evaluate_train(model, epoch, eval_dataloader)
            eval_losses.append(
                {
                    "eval_loss": eval_loss,
                    # "eval_gate_loss": eval_gate_loss
                }
            )
            eval_loss = evaluate_train(model, epoch, eval_dataloader)
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
            exit(1)

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

        if args.do_auxiliary_task:
            set_auxiliary_changes(
                model=accelerator.unwrap_model(model),
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                epoch=epoch,
                best_loss=best_eval_loss,
                eval_losses=eval_losses,
                train_losses=train_losses,
            )


def set_auxiliary_changes(**kwargs):
    if args.stage == 1:
        _set_stage_1_changes(**kwargs)
    elif args.stage == 2:
        _set_stage_2_changes(**kwargs)


def _set_stage_1_changes(model: MmLlamaForSequenceClassification, epoch: int, **_):
    pass


def _set_stage_2_changes(
    model: MmLlamaForSequenceClassification,
    train_losses: list[float],
    eval_losses: list[float],
    epoch: int,
    **_,
):
    pass


def save_model(accelerator: Accelerator, tokenizer, model):
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(
        unwrapped_model.state_dict(exclude=["llama", "wave2vec2"]),
        os.path.join(args.output_path, "best_model.pth"),
    )


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


def load_checkpoint(accelerator: Accelerator, model: MmLlama, checkpoint_path: str):
    accelerator.load_state(checkpoint_path, load_module_strict=False)
    checkpoint = torch.load(os.path.join(checkpoint_path, "checkpoint_metadata.bin"))

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
        if type(batch[k]) is torch.Tensor:
            batch[k] = batch[k].cuda()
            continue
        for kk in batch[k]:
            batch[k][kk] = batch[k][kk].cuda()

    return batch


def test():
    ## setup datasets
    test_dataset = dataset_class(args.test_dataset)(
        args.test_dataset,
        mode="test",
        audio_placement="none",
        task="normal",
        window=args.window_size,
    )

    tokenizer, processor, config = setup_config_and_processor(test_dataset)

    test_dataloader_f1 = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=SequenceClassificationCollator(processor, test_dataset),
    )

    model = MmLlamaForSequenceClassification(config)
    model = load_model_for_test(model)

    f1, loss = evaluate_f1(tokenizer, model, test_dataloader_f1)
    print(f"F1 in Test: {f1}")
    print(f"Loss in Test: {loss}")


def evaluate_train(
    model: MmLlamaForSequenceClassification,
    epoch: int,
    dataloader: DataLoader,
) -> float:
    eval_batch_iterator = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    running_loss = 0
    model.eval()
    for step, batch in enumerate(eval_batch_iterator):
        with torch.no_grad():
            outputs = model(**prepare_batch(batch))
            loss = outputs.main_loss
            running_loss += loss.item()
    running_loss /= len(dataloader)
    print(f"Loss in Epoch {epoch}: {running_loss}")
    return running_loss


def evaluate_f1(
    tokenizer: LlamaTokenizerFast,
    model: MmLlamaForSequenceClassification,
    dataloader: DataLoader,
):
    eval_batch_iterator = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    all_targets = []
    all_preds = []
    all_inputs = []
    all_gates = []

    def gate_hook(module, inputs, outputs):
        all_gates.extend(outputs.cpu())
    model.gate.register_forward_hook(gate_hook)

    running_loss = 0.0
    model.eval()
    for inputs in eval_batch_iterator:
        labels = inputs["labels"]
        with torch.no_grad():
            try:
                inputs = prepare_batch(inputs)
                preds = model(**inputs)
            except TimeoutError:
                print("TimeoutError on input", inputs)
                preds = torch.zeros((inputs["text"]["input_ids"].size(0), 1))

        running_loss += preds.main_loss
        all_preds.extend(preds.logits.cpu())
        all_targets.extend(labels.cpu())
        all_inputs.extend(inputs["text"]["input_ids"].cpu())

    running_loss /= len(dataloader)
    all_preds = torch.stack(all_preds[: len(dataloader.dataset)])
    all_targets = all_targets[: len(dataloader.dataset)]
    all_inputs = all_inputs[: len(dataloader.dataset)]

    all_preds_cert = torch.softmax(all_preds, dim=-1).max(dim=-1).values.tolist()
    all_preds = all_preds.argmax(dim=-1).tolist()
    all_inputs = tokenizer.batch_decode(all_inputs, skip_special_tokens=True)

    dataset = dataloader.dataset
    all_preds = list(map(dataset.id2label, all_preds))
    all_targets = list(map(dataset.id2label, all_targets))

    f1 = f1_score(all_targets, all_preds, average="weighted")
    preds_for_eval = []
    for i, (inp, pred, target, cert) in enumerate(zip(all_inputs, all_preds, all_targets, all_preds_cert)):
        preds_for_eval.append(
            {
                "index": i,
                "input": inp,
                "output": pred,
                "target": target,
                "certainty": cert
            }
        )
    suffix = (
        "_no_audio" if args.ignore_audio else "_no_text" if args.ignore_text else ""
    )
    with open(os.path.join(args.output_path, f"preds_test{suffix}.json"), "wt") as f:
        json.dump(preds_for_eval, f)

    return f1, running_loss


def evaluate_loss(
    model: MmLlamaConcat,
    dataloader: DataLoader,
) -> float:
    eval_batch_iterator = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    running_loss = 0
    model.eval()
    for step, batch in enumerate(eval_batch_iterator):
        with torch.no_grad():
            outputs = model(**prepare_batch(batch))
            loss = outputs.loss
            running_loss += loss.item()
    running_loss /= len(dataloader)
    return running_loss


if __name__ == "__main__":
    if args.evaluation:
        test()
    else:
        create_folder_if_not_exists(args.output_path)
        train()
