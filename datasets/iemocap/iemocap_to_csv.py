import os
import re
from glob import glob
from typing import Callable

import pandas as pd
import argparse


argParser = argparse.ArgumentParser()

argParser.add_argument("--output", type=str, default="iemocap.csv")
argParser.add_argument("--root", type=str, default="./IEMOCAP_full_release")

args = argParser.parse_args()

wav_paths = glob(os.path.join(os.path.abspath(args.root), r"Session*/sentences/wav/*/*.wav"))


def get_speaker_from_path(path: str) -> str:
    return path.split("/")[-1].split("_")[-1][0]


def get_speaker_turn_id_from_path(path: str) -> int:
    return int(path.split("/")[-1].split("_")[-1].split(".")[0][1:])


def get_session_from_path(path: str) -> int:
    return int(path.split("/")[-5][-1])


def get_method_from_path(path: str) -> str:
    if path.split("/")[-1].split("_")[1].startswith("impro"):
        return "impro"
    elif path.split("/")[-1].split("_")[1].startswith("script"):
        return "script"
    else:
        raise ValueError("unknown method")


def get_dialogue_id_from_path(path: str) -> str:
    return path.split("/")[-2]


def get_utterance_id_from_path(path: str) -> str:
    return path.split("/")[-1].split(".")[0]


def get_dialogue_data_from_wav_path(dir: str) -> Callable[[str], str]:
    def inner(path: str) -> str:
        return (
            os.path.dirname(path.replace("sentences", "dialog").replace("wav", dir))
            + ".txt"
        )

    return inner


def get_transcription_path_from_wav_path(path: str) -> str:
    transcript_path = get_dialogue_data_from_wav_path("transcriptions")(path)
    assert os.path.exists(transcript_path), f"{transcript_path} does not exist"
    return transcript_path


def get_dialogue_transcript_from_wav_path(path: str) -> str:
    transcript_path = get_transcription_path_from_wav_path(path)
    with open(transcript_path, "r") as f:
        data = f.readlines()

    return data


def get_transcript_from_wav_path(path: str) -> str:
    data = get_dialogue_transcript_from_wav_path(path)
    utterance_id = get_utterance_id_from_path(path)

    def is_transcript_line(line: str) -> bool:
        return utterance_id in line

    line = list(filter(is_transcript_line, data))
    assert len(line) > 0, f"{path} does not contain transcript line"
    line = line[0]

    transcript = line.split(":")[1].strip()
    return transcript


def get_emotion_from_wav_path(path: str) -> str:
    emotion_path = get_dialogue_data_from_wav_path("EmoEvaluation")(path)
    assert os.path.exists(emotion_path), f"{emotion_path} does not exist"
    with open(emotion_path, "r") as f:
        data = f.readlines()

    utterance_id = get_utterance_id_from_path(path)

    def is_emotion_line(line: str) -> bool:
        return line.startswith("[") and utterance_id in line

    line = list(filter(is_emotion_line, data))
    assert len(line) > 0, f"{path} does not contain emotion line"

    line = line[0]
    emotion = re.findall(utterance_id + r"\s+(\w{3})", line)
    assert len(emotion) == 1, f"could not match emotion in line '{line}'"
    return emotion[0]


def get_dialogue_utterance_turn_id_from_wav_path(path: str) -> int:
    data = get_dialogue_transcript_from_wav_path(path)
    utterance_id = get_utterance_id_from_path(path)

    def is_utterance_line(line: str) -> bool:
        return utterance_id in line

    return [i for i, line in enumerate(data) if is_utterance_line(line)][0]


df = pd.DataFrame({"wav_path": wav_paths})
df["Speaker"] = df["wav_path"].apply(get_speaker_from_path)
df["Session"] = df["wav_path"].apply(get_session_from_path)
df["Speaker_turn_ID"] = df["wav_path"].apply(get_speaker_turn_id_from_path)
df["Utterance_ID"] = df["wav_path"].apply(get_dialogue_utterance_turn_id_from_wav_path)
df["Method"] = df["wav_path"].apply(get_method_from_path)
df["Dialogue_ID"] = df["wav_path"].apply(get_dialogue_id_from_path)
df["Utterance_Name"] = df["wav_path"].apply(get_utterance_id_from_path)
df["Utterance"] = df["wav_path"].apply(get_transcript_from_wav_path)
df["Emotion"] = df["wav_path"].apply(get_emotion_from_wav_path)

df = df.sort_values(by=["Session", "Dialogue_ID", "Utterance_ID"])


df.reset_index(drop=True).to_csv(args.output, index=False)
