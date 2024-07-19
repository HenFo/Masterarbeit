import json
import math
from dataclasses import dataclass
import os

from sklearn.metrics import f1_score
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    LlamaTokenizerFast,
)
from utils import (
    MeldDataset,
    MmLlama,
    MmLlamaConfig,
    MmLlamaProcessor,
    SequenceClassificationCollator,
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn as nn
import argparse

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
    weight_decay: float = 0
    train_llm: bool = False
    lora_dim: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_module_name: str = ".*?llama.*?[qkvo]_proj"
    resume_training: bool = False
    window_size: int = 5
    time_till_aux: int = epochs
    include_target_text_percentage_decay: float = 0.3
    do_auxilary_task: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=bool, default=False)
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
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_llm", action="store_true")
    parser.add_argument("--lora_dim", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_module_name", type=str, default=".*?llama.*?[qkvo]_proj"
    )
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--time_till_aux", type=int, default=15)
    parser.add_argument(
        "--include_target_text_percentage_decay", type=float, default=0.3
    )
    parser.add_argument("--do_auxilary_task", action="store_true")
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
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return grouped_parameters


def get_scheduler(optimizer, len_dataset, batch_size, epochs):
    num_steps = math.ceil(len_dataset / batch_size)
    num_steps *= epochs + 3
    warmup_steps = math.ceil(num_steps * args.warmup_ratio)
    # scheduler = get_constant_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps
    # )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps
    )
    return scheduler


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<audio>"]})
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_for_stage(model: nn.Module, stage: int):
    if stage == 1:
        return load_model_for_stage_1(model)
    elif stage == 2:
        return load_model_for_stage_2(model)
    else:
        raise ValueError("Invalid stage number")


def load_model_for_stage_1(model: nn.Module):
    if args.resume_training:
        model.load_state_dict(
            torch.load(os.path.join(args.output_path, "best_model.pth"))
        )
    model.freeze_encoder()
    return model


def load_model_for_stage_2(model: nn.Module):
    model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_path, "best_model.pth"))
    )
    if args.resume_training:
        model = PeftModel.from_pretrained(model, args.output_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            # task_type="CAUSAL_LM",
            r=args.lora_dim,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_module_name,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    # model.unfreeze_projector()
    return model


def load_model_for_test(model: nn.Module):
    model.load_state_dict(torch.load(os.path.join(args.output_path, "best_model.pth")))
    # model = PeftModel.from_pretrained(model, args.output_path, is_trainable=False)
    # model = model.merge_and_unload(progressbar=True)
    return model.cuda()


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train():
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Load configurations
    llm_config = AutoConfig.from_pretrained(args.llm_id)
    ac_config = AutoConfig.from_pretrained(args.acoustic_id)
    ac_processor = AutoProcessor.from_pretrained(args.acoustic_id)

    # setup of tokenizer
    tokenizer = load_tokenizer()

    # setup of processor
    processor = MmLlamaProcessor(ac_processor, tokenizer)

    ## setup of config
    audio_token_id = tokenizer.additional_special_tokens_ids[0]
    config = MmLlamaConfig(
        llm_config, ac_config, audio_token_id, tokenizer.pad_token_id, args.adapter_id
    )

    ## setup datasets
    include_text = 0 if args.stage == 2 and args.do_auxilary_task else 1
    train_dataset = MeldDataset(
        args.train_dataset,
        mode="train",
        task=args.task,
        window=args.window_size,
        include_target_text_percentage=include_text,
    )
    eval_dataset = MeldDataset(
        args.dev_dataset,
        mode="dev",
        task="normal",
        window=args.window_size,
        include_target_text_percentage=include_text,
    )
    # test_dataset = MeldDataset(args.test_dataset, mode="test", task="normal")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=SequenceClassificationCollator(processor, mode="train"),
    )
    # get model
    model = MmLlama(config, train_llm=args.train_llm)
    model = load_model_for_stage(model, args.stage)

    # setup optimizer
    grouped_parameters = get_grouped_parameters(model)
    optimizer = AdamW(grouped_parameters, lr=args.lr, betas=(0.9, 0.95))
    lr_scheduler = get_scheduler(
        optimizer, len(train_dataset), args.batch_size, args.epochs
    )

    # setup accelerator
    (model, optimizer, lr_scheduler, train_dataloader) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    # training loop
    best_eval_loss = float("inf")
    train_losses = []
    eval_losses = []
    for epoch in range(args.epochs):
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

                    accelerator.backward(loss)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                batch_iterator.set_description(
                    f"Epoch: {epoch} / {args.epochs}, Loss: {loss.item():.4f}"
                )
            except torch.cuda.OutOfMemoryError:
                print("OutOfMemoty error on step", step, "skipping step")
                print("Text size", batch["text"]["input_ids"].size())
                print("Audio size", batch["acoustic"]["input_values"].size())
                torch.cuda.empty_cache()

        lr_scheduler.step()
        running_loss /= len(train_dataloader)
        train_losses.append(running_loss)
        # evaluation
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=SequenceClassificationCollator(processor, mode="train"),
            sampler=SequentialSampler(eval_dataset),
        )

        if args.do_auxilary_task:
            set_auxilary_changes(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                epoch=epoch,
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

        accelerator.wait_for_everyone()


def set_auxilary_changes(**kwargs):
    if args.stage == 1:
        set_stage_1_changes(**kwargs)
    elif args.stage == 2:
        set_stage_2_changes(**kwargs)


def set_stage_1_changes(train_dataloader, epoch, **kwargs):
    if epoch > args.time_till_aux:
        # reduce dataset.include_target_text_percentage linearly to 0.3 till end of training

        end = args.include_target_text_percentage_decay
        percentage = end + (1 - end) * (
            1 - ((epoch - args.time_till_aux) / (args.epochs - args.time_till_aux))
        )
        print("Setting include_target_text_percentage on training data to", percentage)
        train_dataloader.dataset.include_target_text_percentage = percentage


def set_stage_2_changes(eval_dataloader, train_dataloader, epoch, **kwargs):
    eval_dataloader.dataset.include_target_text_percentage = 0
    if epoch > args.time_till_aux:
        # increase dataset.include_target_text_percentage linearly to 0.3 till end of training

        eval_dataloader.dataset.include_target_text_percentage = 1
        start = args.include_target_text_percentage_decay
        percentage = start + (1 - start) * (
            (epoch - args.time_till_aux) / (args.epochs - args.time_till_aux)
        )
        print("Setting include_target_text_percentage on training data to", percentage)
        train_dataloader.dataset.include_target_text_percentage = percentage


def save_model(accelerator, tokenizer, model):
    unwrapped_model = accelerator.unwrap_model(model)
    if args.stage == 1:
        torch.save(
            unwrapped_model.state_dict(),
            os.path.join(args.output_path, "best_model.pth"),
        )
    if args.train_llm:
        model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)


def prepare_batch(batch: dict[str, dict[str, torch.Tensor]]):
    for k in batch:
        for kk in batch[k]:
            batch[k][kk] = batch[k][kk].cuda()
    return batch


def test():
    # Load configurations
    llm_config = AutoConfig.from_pretrained(args.llm_id)
    ac_config = AutoConfig.from_pretrained(args.acoustic_id)
    ac_processor = AutoProcessor.from_pretrained(args.acoustic_id)

    # setup of tokenizer
    tokenizer = load_tokenizer()

    # setup of processor
    processor = MmLlamaProcessor(ac_processor, tokenizer)

    ## setup of config
    audio_token_id = tokenizer.additional_special_tokens_ids[0]
    config = MmLlamaConfig(
        llm_config, ac_config, audio_token_id, tokenizer.pad_token_id, args.adapter_id
    )

    model = MmLlama(config)
    model = load_model_for_test(model)

    ## setup datasets
    test_dataset = MeldDataset(args.test_dataset, mode="test", task="normal", include_target_text_percentage=1)

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
    model: MmLlama,
    epoch: int,
    dataloader: DataLoader,
):
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
    model: MmLlama,
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
