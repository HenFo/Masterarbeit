import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as AT
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
import os
from tqdm.auto import tqdm
from functools import lru_cache


class MeldAudioDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        mode,
        target_sample_rate: int = 16000,
        window: int = 5,
        data_percentage: float = 1.0,
        keep_order: bool = False,
    ):
        assert mode in ["train", "dev", "test"]
        self.mode = mode
        self.dataset_path = dataset_path
        self.target_sample_rate = target_sample_rate
        self.dataset: pd.DataFrame = MeldAudioDataset.prepare_dataset(
            dataset_path, window, data_percentage, keep_order
        )
        self.dataset = self.clean_dataset(self.dataset)
        self.ds_sample_rate = self.guess_samplerate()
        self.resampler = AT.Resample(
            orig_freq=self.ds_sample_rate, new_freq=self.target_sample_rate
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        row = self.dataset.iloc[index, :]
        path = self.build_path(row)
        wavs, _ = torchaudio.load(path)
        wav = wavs[torch.argmax(torch.std(wavs, dim=1))]
        wav = self.resampler(wav)
        y = MeldAudioDataset.label2id(row["Emotion"])

        return wav.numpy(), y

    def guess_samplerate(self) -> int:
        first_row = self.dataset.iloc[0]
        path = self.build_path(first_row)
        _, sr = torchaudio.load(path)
        return sr

    def build_path(self, row: pd.Series) -> str:
        filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        path = os.path.join(
            os.path.dirname(self.dataset_path), "audio", self.mode, filename
        )
        return path

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        corrupted_audio = []
        for i, row in tqdm(
            df.iterrows(), desc=f"Cleaning {self.mode} dataset", total=len(df)
        ):
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
        return ["sadness", "surprise", "neutral", "joy", "anger", "disgust", "fear"]

    @classmethod
    def label2id(cls, label: str) -> int:
        return cls.get_labels().index(label)

    @classmethod
    def id2label(cls, id: int) -> str:
        return cls.get_labels()[id]

    @classmethod
    def transform_speaker_to_id(cls, df: pd.DataFrame) -> pd.DataFrame:
        name_to_id = {name: i for i, name in enumerate(df["Speaker"].unique())}
        df["Speaker"] = df["Speaker"].apply(lambda name: f"Speaker_{name_to_id[name]}")
        return df

    @classmethod
    def prepare_dataset(
        cls,
        path: str,
        window: int = 5,
        data_percentage: float = 1.0,
        keep_order: bool = False,
    ) -> pd.DataFrame:
        ds = pd.read_csv(path, index_col=0).reset_index(drop=True)
        ds = ds.sample(frac=data_percentage, replace=False)
        if keep_order:
            ds = ds.sort_values(["Dialogue_ID", "Utterance_ID"])
        ds["Utterance"] = ds["Utterance"].str.replace("’", "")
        ds["Utterance"] = ds["Utterance"].str.replace("‘", "'")
        ds = cls.transform_speaker_to_id(ds)
        ds = (
            ds.groupby("Dialogue_ID")
            .apply(cls.transform_speaker_to_id)
            .reset_index(drop=True)
        )
        ds["Emotion"] = ds["Emotion"].str.apply(cls.meld2instruct)
        # ds["prompt"] = ds["Speaker"] + ": \"" + ds["Utterance"] + "\""
        ds = MeldAudioDataset.create_window_view(ds, window)
        return ds
    
    @classmethod
    def meld2instruct(cls, label:str) -> str:
        if label == "anger":
            return "angry"
        if label == "joy":
            return "joyful"
        if label == "sadness":
            return "sad"
        return label


    @classmethod
    def create_window_view(
        cls, df: pd.DataFrame, window: int = 5
    ) -> List[pd.DataFrame]:
        groups = df.groupby("Dialogue_ID")
        dialogue_windows = []
        for _, group in groups:
            for i, _ in enumerate(group.sort_values("Utterance_ID").iterrows()):
                start = max(0, i - window)
                end = i + 1
                history = group.iloc[start:end]
                dialogue_windows.append(history)
        return dialogue_windows




class MeldDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        mode,
        task: str = "normal",
        mix_rate: float = 0.1,
        target_sample_rate: int = 16000,
        window: int = 5,
        data_percentage: float = 1.0,
    ):
        assert mode in ["train", "dev", "test"]
        assert task in ["normal", "speaker", "emotion", "mixed"]
        self.mode = mode
        self.dataset_path = dataset_path
        self.target_sample_rate = target_sample_rate
        self.window = window
        self.task = task
        self.mix_rate = mix_rate
        self.dataset: pd.DataFrame = self._prepare_dataset(dataset_path)
        self.emotions = "surprise, anger, neutral, joy, sadness, fear, disgust"
        self.speaker = "Speaker_0, Speaker_1, Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7"
        self.ds_sample_rate = self._guess_samplerate()
        self.resampler = AT.Resample(
            orig_freq=self.ds_sample_rate, new_freq=self.target_sample_rate
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, str], str]:
        history = self._get_history(index)
        history = history.iloc[-self.window :]
        audio = None
        text = None

        if self.task == "normal":
            audio, text = self._get_normal_input(history)
        if self.task == "emotion":
            text = self._generate_emotion_prediction(history)
        if self.task == "mixed":
            if np.random.random() > self.mix_rate:
                audio, text = self._get_normal_input(history)
            else:
                text = self._generate_emotion_prediction(history)
           
        if self.task == "speaker":
            text = self._generate_speaker_ident_task(history)
        
        return (audio, text["input"]), text["target"]

    def _get_normal_input(self, history):
        corrupt = self._check_audio_corruption(history.iloc[-1].name)
        audio = None
        text = None
        if corrupt:
            text = self._generate_input(history, include_audio=False)
        else:
            audio = self._get_audio(history.iloc[-1])
            text = self._generate_input(history)
        return audio, text

    def _get_history(self, index) -> pd.DataFrame:
        target = self.dataset.iloc[index, :]
        history = self.dataset[
            (self.dataset["Dialogue_ID"] == target["Dialogue_ID"])
            & (self.dataset["Utterance_ID"] <= target["Utterance_ID"])
        ]
        return history

    def _get_audio(self, row) -> torch.Tensor:
        path = self._build_path(row)
        wavs, _ = torchaudio.load(path)
        wav = wavs[torch.argmax(torch.std(wavs, dim=1))]
        wav = self.resampler(wav)

        return wav.numpy()

    def _build_path(self, row: pd.Series) -> str:
        filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
        path = os.path.join(
            os.path.dirname(self.dataset_path), "audio", self.mode, filename
        )
        return path


    @lru_cache(maxsize=None)
    def _check_audio_corruption(self, row_name: int) -> bool:
        row = self.dataset.loc[row_name]
        path = self._build_path(row)
        try:
            info = torchaudio.info(path)
            # cut everything longer than 20 seconds
            if info.num_frames / info.sample_rate > 20:
                return True
        except RuntimeError:
            return True
        return False

    def _guess_samplerate(self) -> int:
        first_row = self.dataset.iloc[0]
        path = self._build_path(first_row)
        _, sr = torchaudio.load(path)
        return sr

    def _prepare_dataset(self, path) -> pd.DataFrame:
        ds = pd.read_csv(path, index_col=0).reset_index(drop=True)
        ds["Utterance"] = ds["Utterance"].str.replace("’", "")
        ds["Utterance"] = ds["Utterance"].str.replace("‘", "'")
        ds = (
            ds.groupby("Dialogue_ID")
            .apply(self._transform_speaker_to_id)
            .reset_index(drop=True)
        )
        return ds

    def _transform_speaker_to_id(self, df: pd.DataFrame) -> pd.DataFrame:
        name_to_id = {name: i for i, name in enumerate(df["Speaker"].unique())}
        df["Speaker"] = df["Speaker"].apply(lambda name: f"Speaker_{name_to_id[name]}")
        return df

    def _generate_speaker_ident_task(self, dialog: pd.DataFrame) -> dict:
        prompts = dialog["Speaker"] + ': "' + dialog["Utterance"] + '"'
        dialog_chain = " \t ".join(prompts.iloc[:-1])
        target = dialog.iloc[-1]
        instruction = f"Please select the Speaker label of the next utterance <Speaker: {target['Utterance']}> from <{self.speaker}>:"
        prompt = f"Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speaker. ### {dialog_chain} ### {instruction}"
        return {"input": prompt, "target": target["Speaker"]}

    def _generate_emotion_prediction(self, dialog: pd.DataFrame) -> dict:
        if len(dialog) > 1:
            prompts = dialog["Speaker"] + ': "' + dialog["Utterance"] + '"'
            dialog_chain = " \t ".join(prompts.iloc[:-1])
            target = dialog.iloc[-1]
            instruction = f"Based on the above historical utterances, next utterance is spoken by <{target['Speaker']}>, please predict the emotion states of <{target['Speaker']}> from <{self.emotions}>:"
            prompt = f"Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speaker. ### {dialog_chain} ### {instruction}"
            return {"input": prompt, "target": target["Emotion"]}

        return self._generate_input(dialog, self.emotions)

    def _generate_input(
        self,
        dialog: pd.DataFrame,
        include_audio: bool = True,
        include_text: bool = True,
    ) -> dict:
        prompts = dialog["Speaker"] + ': "' + dialog["Utterance"] + '"'
        dialog_chain = " \t ".join(prompts)
        target = dialog.iloc[-1]
        instruction = f"Please select the emotional label of <{target['Speaker']}: "
        if include_audio:
            instruction += "<audio>"
        if include_text:
            instruction += f"\"{target['Utterance']}\""
        instruction += f"> from <{self.emotions}>:"

        prompt = f"Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speaker. ### {dialog_chain} ### {instruction}"
        return {"input": prompt, "target": target["Emotion"]}


if __name__ == "__main__":
    PATH = "/home/fock/code/MultiModalInstructERC/meld/test_sent_emo.csv"
    tasks = ["normal", "speaker", "emotion", "mixed"]
    ds = MeldDataset(PATH, mode="test", window=10)
    print(ds.dataset[ds.dataset["corrupt"]])
    for t in tasks:
        print(t)
        ds.task = t
        print(ds[6])
        print("########################\n")
