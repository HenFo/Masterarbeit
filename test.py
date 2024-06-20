from utils import MeldDataset, SequenceClassificationCollator, MmLlamaProcessor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoProcessor, AutoModel

LANGUAGE_MODEL = "/home/fock/code/MultiModalInstructERC/models/language/LLaMA2"
LORA_ADAPTER = "/home/fock/code/MultiModalInstructERC/models/language/adapter/InstructERC_unbalanced"
ACOUSTIC_MODEL = "/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
OUTPUT_PATH = "/home/fock/code/MultiModalInstructERC/experiments/multimodal/mlp/concat/"
DS_TRAIN_PATH = "/home/fock/code/MultiModalInstructERC/datasets/meld/train_sent_emo.csv"
DS_DEV_PATH = "/home/fock/code/MultiModalInstructERC/datasets/meld/dev_sent_emo.csv"
DS_TEST_PATH = "/home/fock/code/MultiModalInstructERC/datasets/meld/test_sent_emo.csv"


tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
ac_config = AutoConfig.from_pretrained(ACOUSTIC_MODEL)
ac_processor = AutoProcessor.from_pretrained(ACOUSTIC_MODEL)

# setup of tokenizer
tokenizer.add_special_tokens({"additional_special_tokens": ["<audio>"]})
tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.padding_side = "left"
processor = MmLlamaProcessor(ac_processor, tokenizer)

train_dataset = MeldDataset(DS_TRAIN_PATH, mode="train", task="normal", window=5)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=SequenceClassificationCollator(processor, mode="train"),
)

wave2vec2 = AutoModel.from_pretrained(ac_config._name_or_path)

# processed_sizes = [wave2vec2(**batch["acoustic"]).last_hidden_state.size()[1] for batch in train_dataloader]
# processed_sizes.sort()
# processed_sizes

for batch in train_dataloader:
    # print(batch["acoustic"]["input_values"].size())
    print(wave2vec2(**batch["acoustic"]).last_hidden_state.size()[1])
    break
    

# train_dataset[0]