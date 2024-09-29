import os
import sys

sys.path.append(os.path.abspath("../../"))

import json
import re
from typing import Type

import numpy as np
import pandas as pd
import torch
from fuzzywuzzy import fuzz, process
from plotnine import (
    aes,
    geom_text,
    geom_tile,
    ggplot,
    scale_color_manual,
    scale_fill_cmap,
    scale_x_discrete,
    xlab,
    ylab,
)
from sklearn.metrics import confusion_matrix
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from utils.collator import SequenceClassificationCollator, SequenceGenerationCollator
from utils.dataset import ERCDataset, IemocapDataset, MeldDataset
from utils.model import MmLlama, MmLlamaConfig
from utils.processor import MmLlamaProcessor


def get_dialog(prompt: str) -> tuple[list[str], set[str]]:
    base_prompt = "Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speaker."
    prompt = prompt.replace(base_prompt, "")
    dialogue = re.findall(r"###(.*)###", prompt)[0].strip()
    dialog_text = re.findall(
        r"Speaker_\d:\s?\"(.*?)(?=\"\s?\t|\"\s?$)", dialogue, re.MULTILINE
    )
    involved_speakers = set(re.findall(r"(Speaker_\d)", dialogue))
    return dialog_text, involved_speakers


def build_result_dataframe(path: str) -> pd.DataFrame:
    with open(path, "rt") as f:
        data = json.load(f)
    df = pd.DataFrame.from_records(data, index=["index"])
    return df


def extract_dialogue_information(df: pd.DataFrame):
    """
    Extract information from dialogue column of the dataframe.

    The column "utterance" is created by extracting the last utterance from the dialogue.
    The column "dialogue_length" is created by counting the number of utterances in the dialogue.
    The column "utterence_length" is created by counting the number of words in the last utterance.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe

    Returns
    -------
    pd.DataFrame
        The dataframe with the additional columns "utterance", "dialogue_length", and "utterence_length"
    """

    df = df.copy()
    df["utterance"] = df["input"].str.extract('<Speaker_\d: "(.*?)">')
    df["dialogue_length"] = df["input"].apply(lambda x: len(get_dialog(x)[0]))
    df["utterence_length"] = df["utterance"].apply(lambda x: len(x.split()))

    return df


def merge_result_dataframes(
    dataframes: list[pd.DataFrame], names: list[str]
) -> pd.DataFrame:
    """
    Merge a list of DataFrames on the columns "index" and "target".

    The DataFrames are merged in the order they appear in the list. The columns of the
    DataFrames are suffixed with the corresponding name from the list `names` in the order
    they appear in the list.

    Parameters
    ----------
    dataframes : list of pandas DataFrames
        The DataFrames to merge.
    names : list of str
        The names used to suffix the columns of the DataFrames.

    Returns
    -------
    pandas DataFrame
        The merged DataFrame.
    """

    results = dataframes[0]
    for i, df in enumerate(dataframes[1:]):
        results = results.merge(df, on=["index", "target"], suffixes=["", "_next"])
        results = results.rename(
            columns={c: c.replace("_next", "_" + names[i + 1]) for c in results.columns}
        )
    return results


def get_instructerc_results(base_path: str, dataset: str) -> pd.DataFrame:
    """
    Reads the results from a instructERC prediction file and returns a DataFrame
    with the results.

    The DataFrame contains the following columns:

    - index
    - input
    - target
    - output
    - dialogue_length
    - utterence_length

    The column 'dialogue_length' contains the length of the dialogue in the
    corresponding input and the column 'utterence_length' contains the length
    of the utterance in the corresponding input.

    Parameters
    ----------
    base_path : str
        The base path of the instructERC results.
    dataset : str
        The name of the dataset.

    Returns
    -------
    pandas DataFrame
        The DataFrame with the results.
    """
    path = os.path.join(base_path, dataset, "preds_for_eval.txt")
    with open(path, "rt") as f:
        for _ in range(15):  # Skip the first 16 lines
            next(f)
        json_data = f.read()
        data = json.loads(json_data)

    results = pd.DataFrame.from_records(data, index=["index"])

    results = extract_dialogue_information(results)

    return results


def print_confusion_matrix(
    results: pd.DataFrame,
    target_labels: list[str] | None = None,
    output_column: str = "output",
) -> None:
    target_labels = (
        results["target"].unique() if target_labels is None else target_labels
    )
    cm = confusion_matrix(
        results["target"], results[output_column], labels=target_labels
    )
    cm_df = pd.DataFrame(cm, index=target_labels, columns=target_labels)
    cm_melted = cm_df.reset_index().melt(id_vars="index", value_name="count")
    cm_melted.columns = ["actual", "predicted", "count"]
    cm_melted["actual"] = pd.Categorical(cm_melted["actual"], categories=target_labels)
    cm_melted["predicted"] = pd.Categorical(
        cm_melted["predicted"], categories=target_labels
    )

    # Calculate total counts for each actual class
    total_counts = cm_melted.groupby("actual")["count"].sum().reset_index()
    total_counts.columns = ["actual", "total_count"]

    # Merge total counts back to the melted DataFrame
    cm_melted = cm_melted.merge(total_counts, on="actual")

    # Calculate the fraction
    cm_melted["sqrt_fraction"] = np.sqrt(cm_melted["count"]) / np.sqrt(
        cm_melted["total_count"]
    )
    cm_melted["fraction"] = (cm_melted["count"] / cm_melted["total_count"]).round(2)
    cm_melted["label"] = (
        cm_melted["count"].astype(str) + " (" + cm_melted["fraction"].astype(str) + ")"
    )
    cm_melted["p_group"] = cm_melted["fraction"].apply(
        lambda x: "high" if x > 0.5 else "low"
    )

    p = (
        ggplot(
            cm_melted, aes("factor(predicted)", "factor(actual)", fill="sqrt_fraction")
        )
        + geom_tile(show_legend=False)
        + geom_text(aes(label="label", color="p_group"), size=8, show_legend=False)
        + ylab("Predicted")
        + xlab("True")
        + scale_x_discrete(limits=target_labels[::-1])
        + scale_fill_cmap(cmap_name="magma")
        + scale_color_manual(["black", "white"])
    )

    p.show()



def fuzzy_join(
    df1: pd.DataFrame, df2: pd.DataFrame, col1: str, col2: str, suffixes=("", "_y")
) -> pd.DataFrame:
    """
    Join two pandas DataFrames based on two columns that are not 100% equal by fuzzy matching the content of the two columns.
    """

    df2 = df2.copy()

    df2["input_y"] = df2[col2]

    df_joined = pd.merge(df1, df2, on=["target", "dialogue_length"], suffixes=suffixes)

    col1, col2 = col1 + suffixes[0], col2 + suffixes[1]

    df_joined["similarity"] = df_joined.apply(
        lambda x: fuzz.ratio(x[col1], x[col2]), axis=1
    )

    df_joined = (
        df_joined.groupby(["target", "dialogue_length"])
        .apply(lambda x: x.loc[x["similarity"].idxmax()])
        .reset_index(drop=True)
    )

    return df_joined


# def fuzzy_join(
#     df1: pd.DataFrame, df2: pd.DataFrame, col1: str, col2: str, suffixes=("", "_y")
# ) -> pd.DataFrame:
#     """
#     Join two pandas DataFrames based on two columns that are not 100% equal by fuzzy matching the content of the two columns.
#     """

#     df2 = df2.copy()
#     df2["fuzzy_match"] = df2[col2]

#     df1 = df1.copy()
#     df1["fuzzy_match"] = df1[col1].apply(
#         lambda x: process.extractOne(x, df2["fuzzy_match"], scorer=fuzz.ratio)[0]
#     )
#     df1 = pd.merge(df1, df2, on="fuzzy_match", suffixes=suffixes)

#     df1 = df1.drop(columns=["fuzzy_match"])

#     return df1


def get_model(
    llm_path: str,
    adapter_path: str,
    acoustic_path: str,
    checkpoint_path: str,
    model_class: Type[MmLlama],
    num_labels: int = 0,
    model_kwargs: dict = {},
):
    llm_config = AutoConfig.from_pretrained(llm_path)
    ac_config = AutoConfig.from_pretrained(acoustic_path)
    ac_processor = AutoProcessor.from_pretrained(acoustic_path)

    # setup of tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<audio>", "</audio>"]})
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    # setup of processor
    processor = MmLlamaProcessor(ac_processor, tokenizer)

    ## setup of config
    audio_token_id = tokenizer.additional_special_tokens_ids[0]
    audio_end_token_id = tokenizer.additional_special_tokens_ids[1]
    config = MmLlamaConfig(
        llm_config=llm_config,
        audio_config=ac_config,
        audio_token_id=audio_token_id,
        audio_end_token_id=audio_end_token_id,
        pad_token_id=tokenizer.pad_token_id,
        llm_pretrained_adapter=adapter_path,
        num_labels=num_labels,
    )

    model: MmLlama = model_class(config, **model_kwargs)
    model.load_state_dict(
        torch.load(os.path.join(checkpoint_path, "best_model.pth")), strict=False
    )
    model = model.apply_inference_lora(checkpoint_path)
    if torch.cuda.is_available():
        model = model.to("cuda")

    return model, config, processor


def dataset_class(dataset_path: str) -> Type[ERCDataset]:
    if "meld" in dataset_path:
        return MeldDataset
    if "iemocap" in dataset_path:
        return IemocapDataset
    else:
        raise ValueError("Invalid dataset path")


T_Collator = Type[SequenceGenerationCollator] | Type[SequenceClassificationCollator]


def get_samples(
    dataset_path: str,
    processor: MmLlamaProcessor,
    collator_type: T_Collator,
    sample_indexes: list[int],
    dataset_kwargs: dict = {},
):
    """
    Gets a list of samples from a dataset.

    Args:
        dataset_path (str): The path to the dataset.
        processor (MmLlamaProcessor): The processor to use for the dataset.
        collator_type (T_Collator): The type of collator to use.
        sample_indexes (List[int]): The list of indexes of the samples to retrieve.
        dataset_kwargs (dict, optional): Additional keyword arguments to pass to the dataset constructor. Defaults to {}.

    Returns:
        List[dict]: A list of samples from the dataset, preprocessed by the processor and collated by the collator.
    """

    test_dataset = dataset_class(dataset_path)(
        dataset_path, mode="test", **dataset_kwargs
    )
    raw_samples = [test_dataset[i] for i in sample_indexes]
    samples = collator_type(processor, mode="dev")(raw_samples)
    samples = [sample for sample in samples]

    return samples


if __name__ == "__main__":
    df1 = pd.DataFrame({"a": ["aaa", "bbb", "ccc"], "b": [1, 2, 3]})
    df2 = pd.DataFrame({"a": ["aba", "bcb", "cac"], "c": [4, 5, 6]})

    df = fuzzy_join(df1, df2, "a", "a")
    print(df)
