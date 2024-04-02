from transformers import AutoProcessor, AutoConfig
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from utils import MeldAudioDataset, DynamicPadCollator, MyEmotionModel



MODEL = "/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
MODEL_PATH = "/home/fock/code/MultiModalInstructERC/experiments/acoustic/wav2vec2/classification/"
DS_TEST_PATH = "/home/fock/code/MultiModalInstructERC/meld/test_sent_emo.csv"


def eval():
    # accelerator = Accelerator()

    config = AutoConfig.from_pretrained(MODEL)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = MyEmotionModel.from_pretrained(MODEL_PATH, config=config, num_labels=len(MeldAudioDataset.get_labels()))

    test_dataset = MeldAudioDataset(DS_TEST_PATH, "test", keep_order=True)

    collator = DynamicPadCollator(processor)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collator, num_workers=16)

    # model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model = model.cuda()
    
    model.eval()
    all_preds = []
    all_targets = []
    for batch in tqdm(test_dataloader):
        x, y = batch
        x = {k: v.cuda() for k, v in x.items()}
        
        with torch.no_grad():
            _, logits = model(**x)

        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.numpy())

    all_preds = list(map(lambda x: MeldAudioDataset.id2label(x), all_preds))
    all_targets = list(map(lambda x: MeldAudioDataset.id2label(x), all_targets))

    print(f1_score(all_targets, all_preds, average='weighted'))


    df = test_dataset.dataset
    df["predicted_emotion"] = all_preds
    df.to_csv("/home/fock/code/MultiModalInstructERC/results/audio/test_results.csv", index=True)

if __name__ == "__main__":
    eval()
