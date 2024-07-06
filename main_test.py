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
    parser.add_argument("--test_dataset", type=str, default=None)
    args = parser.parse_args()
    return Args(**vars(args))


args = parse_args()



def load_model_for_test(model: nn.Module):
    model.load_state_dict(
        torch.load(os.path.join(args.output_path, "best_model.pth"))
    )
    model = PeftModel.from_pretrained(model, args.output_path, is_trainable=False)
    model = model.merge_and_unload(progressbar=True)
    return model


def prepare_batch(batch:dict[str, dict[str, torch.Tensor]]):
    for k in batch:
        for kk in batch[k]:
            batch[k][kk] = batch[k][kk].cuda()
    return batch
    

def test():
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
    test_dataset = MeldDataset(args.test_dataset, mode="test", task="normal")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=SequenceClassificationCollator(processor, mode="dev"),
    )
    # get model
    model = MmLlama(config, train_llm=args.train_llm)
    model = load_model_for_test(model)
    model = model.cuda()
    evaluate(tokenizer, model, test_dataloader)




def evaluate(
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
    with open(os.path.join(args.output_path, "final_test_preds.json"), "wt") as f:
        json.dump(preds_for_eval, f)

    print(f"F1 for Test: {f1}")
    return f1


if __name__ == "__main__":
    test()
