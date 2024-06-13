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

LANGUAGE_MODEL = "/home/fock/code/MultiModalInstructERC/models/language/LLaMA2"
LORA_ADAPTER = "/home/fock/code/MultiModalInstructERC/models/language/adapter/InstructERC_unbalanced"
ACOUSTIC_MODEL = "/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
OUTPUT_PATH = "/home/fock/code/MultiModalInstructERC/experiments/multimodal/mlp/concat/"
DS_TRAIN_PATH = "/home/fock/code/MultiModalInstructERC/meld/train_sent_emo.csv"
DS_DEV_PATH = "/home/fock/code/MultiModalInstructERC/meld/dev_sent_emo.csv"
DS_TEST_PATH = "/home/fock/code/MultiModalInstructERC/meld/test_sent_emo.csv"


@dataclass()
class Args:
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


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_llm", type=bool, default=False)
    parser.add_argument("--lora_dim", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_module_name", type=str, default=".*?llama.*?[qkvo]_proj"
    )
    parser.add_argument("--resume_training", type=bool, default=False)
    args = parser.parse_args()
    return Args(**vars(args))


args = parse_args()


# args = Args(
#     batch_size=2,
#     gradient_accumulation_steps=32,
#     llm_id=LANGUAGE_MODEL,
#     acoustic_id=ACOUSTIC_MODEL,
#     adapter_id=LORA_ADAPTER,
#     output_path=OUTPUT_PATH,
#     train_dataset=DS_TRAIN_PATH,
#     test_dataset=DS_TEST_PATH,
#     dev_dataset=DS_DEV_PATH,
#     task="normal",
#     deepspeed_config="deepspeed_config.json",
#     epochs=7,
#     lr=1e-5,
#     train_llm=True,
#     resume_training=False,
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
    num_steps *= epochs
    warmup_steps = math.ceil(num_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps
    )
    return scheduler


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
    model.unfreeze_projector()
    return model


def train():
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Load configurations
    llm_config = AutoConfig.from_pretrained(args.llm_id)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    ac_config = AutoConfig.from_pretrained(args.acoustic_id)
    ac_processor = AutoProcessor.from_pretrained(args.acoustic_id)

    # setup of tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ["<audio>"]})
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    # setup of processor
    processor = MmLlamaProcessor(ac_processor, tokenizer)

    ## setup of config
    audio_token_id = tokenizer.additional_special_tokens_ids[0]
    config = MmLlamaConfig(
        llm_config, ac_config, audio_token_id, tokenizer.pad_token_id, args.adapter_id
    )

    ## setup datasets
    train_dataset = MeldDataset(args.test_dataset, mode="train", task=args.task)
    eval_dataset = MeldDataset(args.test_dataset, mode="dev", task="normal")
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
    best_f1 = 0
    for epoch in range(args.epochs):
        epoch += 1
        model.train()
        batch_iterator = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch}",
        )
        for step, batch in batch_iterator:
            try:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs["loss"]

                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                batch_iterator.set_description(
                    f"Epoch: {epoch} / {args.epochs}, Loss: {loss.item():.4f}"
                )
            except torch.cuda.OutOfMemoryError:
                print("OutOfMemoty error on step", step, "skipping step")
                print("Text size", batch["text"]["input_ids"].size())
                print("Audio size", batch["audio"]["input_values"].size())

        # evaluation
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=SequenceClassificationCollator(processor, mode="dev"),
            sampler=SequentialSampler(eval_dataset),
        )


        if accelerator.is_main_process:
            f1 = evaluate(accelerator, tokenizer, model, epoch, eval_dataloader)
            if f1 > best_f1:
                print(f"Best F1: {best_f1}")
                print("Saving model")
                best_f1 = f1
                unwrapped_model = accelerator.unwrap_model(model)
                create_folder_if_not_exists(args.output_path)
                torch.save(
                    unwrapped_model.state_dict(),
                    os.path.join(args.output_path, "best_model.pth"),
                )
                if args.train_llm:
                    model.save_pretrained(args.output_path)
                tokenizer.save_pretrained(args.output_path)
                print("model saved")
        accelerator.wait_for_everyone()


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def evaluate(
    accelerator: Accelerator,
    tokenizer: LlamaTokenizerFast,
    model: MmLlama,
    epoch: int,
    dataloader: DataLoader,
):
    dataloader = accelerator.prepare(dataloader)
    eval_batch_iterator = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    all_targets = []
    all_preds = []
    all_inputs = []
    model.eval()
    for inputs, labels in eval_batch_iterator:
        with torch.no_grad():
            try:
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
    with open(f"preds_epoch_{epoch}.json", "wt") as f:
        json.dump(preds_for_eval, f)

    print(f"F1 in Epoch {epoch}: {f1}")
    return f1


if __name__ == "__main__":
    train()
