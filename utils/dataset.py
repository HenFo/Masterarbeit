from abc import ABC, abstractmethod
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as AT
import numpy as np
import pandas as pd
from typing import List, Tuple
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
    def meld2instruct(cls, label: str) -> str:
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


class ERCDataset(Dataset, ABC):
    emotions: list[str]
    speaker: list[str]

    def __init__(
        self,
        dataset_path: str,
        mode,
        task: str = "normal",
        audio_placement: str = "target",
        mix_rate: float = 0.1,
        target_sample_rate: int = 16000,
        window: int = 5,
        include_audio_percentage: float = 1.0,
        include_target_text_percentage: float = 1.0,
    ):
        assert mode in ["train", "dev", "test"]
        assert task in [
            "normal",
            "speaker",
            "emotion",
            "mixed",
            "audio_only",
            "text_only",
        ]
        assert audio_placement in ["target", "front", "enclose", "none"]
        self.mode = mode
        self.dataset_path = dataset_path
        self.target_sample_rate = target_sample_rate
        self.window = window
        self.task = task
        self.audio_placement = audio_placement
        self.mix_rate = mix_rate
        self.include_audio_percentage = include_audio_percentage
        self.include_target_text_percentage = include_target_text_percentage
        self.dataset: pd.DataFrame = self._prepare_dataset(dataset_path)
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

        if self.task == "normal" or self.task == "text_only":
            audio, text = self._get_normal_input(
                history
            )
            if self.task == "text_only":
                audio = None
        elif self.task == "emotion":
            text = self._generate_emotion_prediction(history)
        elif self.task == "mixed":
            if np.random.random() > self.mix_rate:
                audio, text = self._get_normal_input(history)
            else:
                text = self._generate_emotion_prediction(history)

        elif self.task == "speaker":
            text = self._generate_speaker_ident_task(history)

        elif self.task == "audio_only":
            audio, text = self._get_audio_only_input(history)

        return (audio, text["input"]), text["target"]

    def label2id(self, label) -> int:
        return self.emotions.index(label)

    def id2label(self, id) -> str:
        return self.emotions[id]

    def _get_normal_input(self, history, include_audio=True):
        corrupt = self._check_audio_corruption(history.iloc[-1].name)
        audio = None
        text = None
        if corrupt:
            text = self._generate_input(history, include_audio=False)
        else:
            audio = self._get_audio(history.iloc[-1])
            include_audio = (
                np.random.random() < self.include_audio_percentage and include_audio
            )
            include_text = (
                np.random.random() < self.include_target_text_percentage
                or not include_audio
            )
            text = self._generate_input(history, include_audio, include_text)
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

    @abstractmethod
    def _build_path(self, row: pd.Series) -> str:
        raise NotImplementedError()

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

    @abstractmethod
    def _prepare_dataset(self, path: str) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform_speaker_to_id(self, df: pd.DataFrame) -> pd.DataFrame:
        name_to_id = {name: i for i, name in enumerate(df["Speaker"].unique())}
        df["Speaker"] = df["Speaker"].apply(lambda name: f"Speaker_{name_to_id[name]}")
        return df

    def _generate_speaker_ident_task(self, dialog: pd.DataFrame) -> dict:
        prompts = dialog["Speaker"] + ': "' + dialog["Utterance"] + '"'
        dialog_chain = " \t ".join(prompts.iloc[:-1])
        target = dialog.iloc[-1]
        instruction = f"Please select the Speaker label of the next utterance <Speaker: {target['Utterance']}> from <{', '.join(self.speaker)}>:"
        prompt = f"Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speakers. ### {dialog_chain} ### {instruction}"
        return {"input": prompt, "target": target["Speaker"]}

    def _generate_emotion_prediction(self, dialog: pd.DataFrame) -> dict:
        if len(dialog) > 1:
            prompts = dialog["Speaker"] + ': "' + dialog["Utterance"] + '"'
            dialog_chain = " \t ".join(prompts.iloc[:-1])
            target = dialog.iloc[-1]
            instruction = f"Based on the above historical utterances, next utterance is spoken by <{target['Speaker']}>, please predict the emotion states of <{target['Speaker']}> from <{', '.join(self.emotions)}>:"
            prompt = f"Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speakers. ### {dialog_chain} ### {instruction}"
            return {"input": prompt, "target": target["Emotion"]}

        return self._generate_input(dialog, self.emotions)

    def _get_audio_only_input(self, dialog: pd.DataFrame) -> dict:
        target = dialog.iloc[-1]
        corrupt = self._check_audio_corruption(target.name)
        audio = self._get_audio(target) if not corrupt else None
        instruction = f"Please select the emotional state of the speaker from <{', '.join(self.emotions)}> based on the given audio features:"
        prompt = f"Now you are expert of sentiment and emotional analysis. The following are audio features of one person speaking as part of a conversation, noted between '[' and ']': Audio features: [<audio>] {instruction}"
        return audio, {"input": prompt, "target": target["Emotion"]}

    def _generate_input(
        self,
        dialog: pd.DataFrame,
        include_audio: bool = True,
        include_text: bool = True,
    ) -> dict:
        prompts = dialog["Speaker"] + ': "' + dialog["Utterance"] + '"'
        dialog_chain = "### " + " \t ".join(prompts) + " ###"
        target = dialog.iloc[-1]
        instruction = f"Please select the emotional label of <{target['Speaker']}: "
        if include_audio:
            if self.audio_placement == "target":
                dialog_chain += " Audio features of last utterance: [<audio>]"
            if self.audio_placement == "front":
                dialog_chain = (
                    f"Audio features of last utterance: [<audio>] {dialog_chain}"
                )
        if include_text:
            if self.audio_placement == "enclose":
                target_utterance = f"<audio> {target['Utterance']} </audio>"
                instruction += f'"{target_utterance}"'
                idx = dialog_chain.rfind(target["Utterance"])
                dialog_chain = (
                    dialog_chain[:idx]
                    + target_utterance
                    + dialog_chain[idx + len(target["Utterance"]) :]
                )
            else:
                instruction += f"\"{target['Utterance']}\""

        instruction += f"> from <{', '.join(self.emotions)}>"
        if include_audio and self.audio_placement == "target":
            instruction += " based on both the context and audio features:"

        prompt = f"Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speaker. {dialog_chain} {instruction}"
        return {"input": prompt, "target": target["Emotion"]}


class MeldDataset(ERCDataset):
    def __init__(
        self,
        dataset_path: str,
        mode,
        task: str = "normal",
        audio_placement: str = "target",
        mix_rate: float = 0.1,
        target_sample_rate: int = 16000,
        window: int = 5,
        include_audio_percentage: float = 1.0,
        include_target_text_percentage: float = 1.0,
    ):
        super().__init__(
            dataset_path,
            mode,
            task,
            audio_placement,
            mix_rate,
            target_sample_rate,
            window,
            include_audio_percentage,
            include_target_text_percentage,
        )
        self.emotions = [
            "surprise",
            "anger",
            "neutral",
            "joy",
            "sadness",
            "fear",
            "disgust",
        ]
        self.speaker = [
            "Speaker_0",
            "Speaker_1",
            "Speaker_2",
            "Speaker_3",
            "Speaker_4",
            "Speaker_5",
            "Speaker_6",
            "Speaker_7",
        ]

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

    def _build_path(self, row: pd.Series) -> str:
        filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
        path = os.path.join(
            os.path.dirname(self.dataset_path), "audio", self.mode, filename
        )
        return path


class IemocapDataset(ERCDataset):
    def __init__(
        self,
        dataset_path: str,
        mode,
        task: str = "normal",
        audio_placement: str = "target",
        mix_rate: float = 0.1,
        target_sample_rate: int = 16000,
        window: int = 5,
        include_audio_percentage: float = 1.0,
        include_target_text_percentage: float = 1.0,
    ):
        self.emotions = ["neutral", "angry", "frustrated", "happy", "excited", "sad"]
        self.speaker = ["Speaker_0", "Speaker_1"]
        super().__init__(
            dataset_path,
            mode,
            task,
            audio_placement,
            mix_rate,
            target_sample_rate,
            window,
            include_audio_percentage,
            include_target_text_percentage,
        )

        self.filtered_dataset = self.dataset[
            self.dataset["Emotion"].isin(self.emotions)
        ]
        indices = self.dataset.index[
            self.dataset.isin(self.filtered_dataset).all(axis=1)
        ]
        self.index_mapping = {i: j for i, j in enumerate(indices)}

    def __getitem__(self, index):
        index = self.index_mapping[index]
        return super().__getitem__(index)

    def __len__(self):
        return len(self.filtered_dataset)

    def _prepare_dataset(self, path: str) -> pd.DataFrame:
        label_mapping = {
            "neu": "neutral",
            "ang": "angry",
            "fru": "frustrated",
            "hap": "happy",
            "exc": "excited",
            "sad": "sad",
        }
        ds = pd.read_csv(path)
        split = pd.read_csv(os.path.join(os.path.dirname(path), "iemocap_split.csv"))
        ds = pd.merge(ds, split, on="Dialogue_ID")
        ds["Emotion"] = ds["Emotion"].map(label_mapping).astype(str)
        ds = ds[ds["Split"] == self.mode]
        ds = (
            ds.groupby("Dialogue_ID")
            .apply(self._transform_speaker_to_id)
            .reset_index(drop=True)
        )
        return ds

    def _build_path(self, row: pd.Series) -> str:
        return row["wav_path"]


if __name__ == "__main__":
    PATH = "/home/fock/code/MultiModalInstructERC/datasets/iemocap/iemocap.csv"
    # tasks = ["normal", "speaker", "emotion", "mixed"]
    ds = IemocapDataset(
        PATH, mode="test", window=1, task="text_only", audio_placement="target"
    )

    x = ds[3]
    print(x[0][1])
