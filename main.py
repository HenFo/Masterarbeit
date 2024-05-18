from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader
from utils import (
    MeldDataset,
    MmLlama,
    MmLlamaConfig,
    MmLlamaProcessor,
    SequenceClassificationCollator,
)
from accelerate import Accelerator
from dataclasses import dataclass

LANGUAGE_MODEL = "/home/fock/code/MultiModalInstructERC/models/language/LLaMA2"
LORA_ADAPTER = "/home/fock/code/MultiModalInstructERC/models/language/adapter/InstructERC_unbalanced"
ACOUSTIC_MODEL = "/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
OUTPUT_PATH = "/home/fock/code/MultiModalInstructERC/experiments/multimodal/mlp/concat"
DS_TRAIN_PATH = "/home/fock/code/MultiModalInstructERC/meld/train_sent_emo.csv"
DS_DEV_PATH = "/home/fock/code/MultiModalInstructERC/meld/dev_sent_emo.csv"
DS_TEST_PATH = "/home/fock/code/MultiModalInstructERC/meld/test_sent_emo.csv"


@dataclass()
class Args:
    llm_id: str = None
    adapter_id: str = None
    acoustic_id: str = None
    output_path: str = None
    train_dataset: str = None
    dev_dataset: str = None
    test_dataset: str = None
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    task: str = "normal"
    batch_size: int = 1


args = Args(
    gradient_accumulation_steps=8,
    llm_id=LANGUAGE_MODEL,
    acoustic_id=ACOUSTIC_MODEL,
    output_path=OUTPUT_PATH,
    train_dataset=DS_TRAIN_PATH,
)


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

    # setup of config
    audio_token_id = tokenizer.additional_special_tokens_ids[0]
    config = MmLlamaConfig(
        llm_config,
        ac_config,
        audio_token_id,
        tokenizer.pad_token_id,
    )

    # get model
    # model = MmLlama(config)

    # setup datasets
    # TODO Zu Trainingsdatensatz ändern !!!!!!!!!!!!!!!!!!!!!!
    train_dataset = MeldDataset(args.test_dataset, mode="train", task=args.task)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=SequenceClassificationCollator(processor),
    )

    print(next(iter(train_dataloader)))


if __name__ == "__main__":
    train()
