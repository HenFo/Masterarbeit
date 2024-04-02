# %%
from transformers import AutoProcessor, AutoConfig, get_scheduler
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from utils import MeldAudioDataset, DynamicPadCollator, MyEmotionModel

# %%
MODEL = "/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
MODEL_PATH = "/home/fock/code/MultiModalInstructERC/experiments/acoustic/wav2vec2/classification/"
DS_TRAIN_PATH = "/home/fock/code/MultiModalInstructERC/meld/train_sent_emo.csv"
DS_DEV_PATH = "/home/fock/code/MultiModalInstructERC/meld/dev_sent_emo.csv"
DS_TEST_PATH = "/home/fock/code/MultiModalInstructERC/meld/test_sent_emo.csv"

EPOCHS = 20
EPOCHS_TO_UNFREEZE = 10
BATCH_SIZE = 16
GRAD_ACC_STEPS = 4

def train(mixed_precision="bf16"):
    accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=GRAD_ACC_STEPS)
    config = AutoConfig.from_pretrained(MODEL)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = MyEmotionModel.from_pretrained(MODEL, config=config, ignore_mismatched_sizes=True, num_labels=len(MeldAudioDataset.get_labels()))
    model.freeze_encoder()

    train_dataset = MeldAudioDataset(DS_TRAIN_PATH, "train", data_percentage=1)
    dev_dataset = MeldAudioDataset(DS_DEV_PATH, "dev", data_percentage=1)
    # test_dataset = MeldAudioDataset(DS_TEST_PATH, "test")
    
    batch_size = BATCH_SIZE // GRAD_ACC_STEPS
    collator = DynamicPadCollator(processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=8)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=8)

    optimizer = torch.optim.Adam(model.classifier.parameters())
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * EPOCHS)

    model, optimizer, lr_scheduler, train_dataloader, dev_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, dev_dataloader)


    train_loss_per_epoch = []
    best_f1 = 0
    for epoch in range(EPOCHS):
        if epoch == EPOCHS_TO_UNFREEZE:
            accelerator.print("Unfreezing encoder layers")
            new_params = accelerator.unwrap_model(model).unfreeze_encoder_layers([11])
            optimizer.add_param_group({"params": new_params})

        epoch += 1
        model.train()
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Running epoch {epoch} / {EPOCHS}",
        )
        losses = []
        for step, batch in enumerate(batch_iterator):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                x, y = batch
                _, logits = model(**x)
                loss = F.cross_entropy(logits, y)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
        
            current_loss = loss.item()
            batch_iterator.set_description(
                f"Epochs {epoch}/{EPOCHS}. Current Loss: {current_loss:.5f}"
            )
            losses.append(current_loss)
        train_loss_per_epoch.append(np.mean(losses))

        model.eval()
        all_preds = []
        all_targets = []
        for batch in tqdm(dev_dataloader):
            x, y = batch
            with torch.no_grad():
                _, logits = model(**x)

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            preds, y = accelerator.gather_for_metrics((preds, y))
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        
        f1 = f1_score(all_targets, all_preds, average='weighted')
        accelerator.print(f"Epoch {epoch} F1: {f1}")
        if f1 > best_f1:
            best_f1 = f1
            accelerator.wait_for_everyone()
            accelerator.print("Saving model...")
            accelerator.unwrap_model(model).save_pretrained(MODEL_PATH)



if __name__ == "__main__":
    train()


