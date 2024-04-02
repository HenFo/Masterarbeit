import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as AT
import numpy as np
import pandas as pd
from typing import List, Tuple
import os
from tqdm.auto import tqdm


class MeldAudioDataset(Dataset):
    def __init__(self, dataset_path:str, mode, target_sample_rate:int = 16000, window:int = 5, data_percentage:float = 1.0, keep_order:bool = False):
        assert mode in ["train", "dev", "test"]
        self.mode = mode
        self.dataset_path = dataset_path
        self.target_sample_rate = target_sample_rate
        self.dataset:pd.DataFrame = MeldAudioDataset.prepare_dataset(dataset_path, window, data_percentage, keep_order)
        self.dataset = self.clean_dataset(self.dataset)
        self.ds_sample_rate = self.guess_samplerate()
        self.resampler = AT.Resample(orig_freq=self.ds_sample_rate, new_freq=self.target_sample_rate)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        row = self.dataset.iloc[index, :]
        path = self.build_path(row)
        wavs, _ = torchaudio.load(path)
        wav =  wavs[torch.argmax(torch.std(wavs, dim=1))]
        wav = self.resampler(wav)
        y = MeldAudioDataset.label2id(row["Emotion"])

        return wav.numpy(), y


    def guess_samplerate(self) -> int:
        first_row = self.dataset.iloc[0]
        path = self.build_path(first_row)
        _, sr = torchaudio.load(path)
        return sr
    
    def build_path(self, row:pd.Series) -> str:
        filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        path = os.path.join(os.path.dirname(self.dataset_path), "audio", self.mode, filename)
        return path
    
    def clean_dataset(self, df:pd.DataFrame) -> pd.DataFrame:
        corrupted_audio = []
        for i, row in tqdm(df.iterrows(), desc=f"Cleaning {self.mode} dataset", total=len(df)):
            path = self.build_path(row)
            try:
                info = torchaudio.info(path)
                if info.num_frames > 2000:
                    tqdm.write(f"Too long audio file: {path}")
                    corrupted_audio.append(i)
            except RuntimeError:
                tqdm.write(f"Corrupted audio file: {path}")
                corrupted_audio.append(i)
        print(f"Found {len(corrupted_audio)} corrupted audio files")
        return df.drop(corrupted_audio)

    @classmethod
    def get_labels(cls) -> List[str]:
        return ['sadness', 'surprise', 'neutral', 'joy', 'anger', 'disgust', 'fear']
    
    @classmethod
    def label2id(cls, label:str) -> int:
        return cls.get_labels().index(label)
    
    @classmethod
    def id2label(cls, id:int) -> str:
        return cls.get_labels()[id]

    @classmethod
    def transform_speaker_to_id(cls, df:pd.DataFrame) -> pd.DataFrame:
        name_to_id = {name:i for i, name in enumerate(df["Speaker"].unique())}
        df["Speaker"] = df["Speaker"].apply(lambda name: f"Speaker_{name_to_id[name]}")
        return df

    @classmethod
    def prepare_dataset(cls, path:str, window:int = 5, data_percentage: float = 1.0, keep_order: bool = False) -> pd.DataFrame:
        ds = pd.read_csv(path, index_col=0).reset_index(drop=True)
        ds = ds.sample(frac=data_percentage, replace=False)
        if keep_order:
            ds = ds.sort_values(["Dialogue_ID", "Utterance_ID"])
        ds["Utterance"] = ds["Utterance"].str.replace("’", "")
        ds["Utterance"] = ds["Utterance"].str.replace("‘", "'")
        ds = cls.transform_speaker_to_id(ds)
        ds = ds.groupby("Dialogue_ID").apply(cls.transform_speaker_to_id).reset_index(drop=True)
        # ds["prompt"] = ds["Speaker"] + ": \"" + ds["Utterance"] + "\""
        # ds = MeldAudioDataset.create_window_view(ds, window)
        return ds

    @classmethod
    def create_window_view(cls, df:pd.DataFrame, window:int = 5) -> List[pd.DataFrame]:
        groups = df.groupby("Dialogue_ID")
        dialogue_windows = []
        for _, group in groups:
            for i, _ in enumerate(group.sort_values("Utterance_ID").iterrows()):
                start = max(0, i-window)
                end = i+1
                history = group.iloc[start:end]
                dialogue_windows.append(history)
        return dialogue_windows