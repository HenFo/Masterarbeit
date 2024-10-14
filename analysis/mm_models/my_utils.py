import os
import sys

sys.path.append(os.path.abspath("../../"))

import json
import re
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from plotnine import (
    aes,
    element_text,
    geom_text,
    geom_tile,
    ggplot,
    ggtitle,
    scale_color_manual,
    scale_fill_cmap,
    scale_fill_gradient2,
    scale_x_discrete,
    theme,
    theme_bw,
    xlab,
    ylab,
)
from sklearn.metrics import confusion_matrix
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from utils.collator import SequenceClassificationCollator, SequenceGenerationCollator
from utils.dataset import ERCDataset, IemocapDataset, MeldDataset
from utils.model import MmLlama, MmLlamaConfig
from utils.processor import MmLlamaProcessor

plt.rcParams["figure.figsize"] = (10, 12)
plt.rcParams["figure.dpi"] = 96


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
    results: pd.DataFrame | None = None,
    target_labels: list[str] | None = None,
    output_column: str = "output",
    target_column: str = "target",
    title: str = "Confusion Matrix",
    xlab_name: str = "Predicted Emotion",
    ylab_name: str = "True Emotion",
    text_size: float = 12,
    label_scaling_adjustment: float = 0,
    name: str | None = None,
    show_percentage: bool = True,
) -> None:
    target_labels = (
        results[target_column].unique() if target_labels is None else target_labels
    )
    cm = confusion_matrix(
        results[target_column], results[output_column], labels=target_labels
    )
    cm_melted = _prepare_confusion_matrix(cm, target_labels)

    # Calculate the fraction
    cm_melted["color_scale"] = np.sqrt(cm_melted["count"]) / np.sqrt(
        cm_melted["total_count"]
    )

    cm_melted["fraction"] = (cm_melted["count"] / cm_melted["total_count"]).round(2)
    cm_melted["label"] = (
        cm_melted["count"].astype(str)
    )

    if show_percentage:
        cm_melted["label"] = cm_melted["label"] + "\n" + (cm_melted["fraction"]*100).astype(int).astype(str) + "%"

    cm_melted["p_group"] = cm_melted["fraction"].apply(
        lambda x: "high" if x > 0.5 else "low"
    )

    p = (
        ggplot(
            cm_melted, aes("factor(predicted)", "factor(actual)", fill="color_scale")
        )
        + geom_tile(show_legend=False)
        + geom_text(aes(label="label", color="p_group"), size=text_size, show_legend=False)
        + ggtitle(title)
        + ylab(ylab_name)
        + xlab(xlab_name)
        + scale_x_discrete(limits=target_labels[::-1])
        + scale_fill_cmap(cmap_name="magma")
        + scale_color_manual(["black", "white"])
        + theme_bw()
        + theme(
            title=element_text(size=18 + label_scaling_adjustment),  # Increases title size
            axis_title=element_text(size=20 + label_scaling_adjustment),  # Increases axis title size
            axis_text=element_text(size=16 + label_scaling_adjustment),  # Increases axis tick label size
            axis_text_x=element_text(rotation=45),  # Rotates x-axis tick labels
        )
    )

    if name is not None:
        p.save(name, width=7, height=7, dpi=300)

    p.show()


def print_confusion_matrix_difference(
    results: pd.DataFrame | None = None,
    target_labels: list[str] | None = None,
    output_column1: str = "output",
    output_column2: str = "output_y",
    target_column: str = "target",
    title: str = "Difference in Confusion Matrices",
    name: str | None = None,
) -> None:
    target_labels = (
        results[target_column].unique() if target_labels is None else target_labels
    )
    cm1 = confusion_matrix(
        results[target_column], results[output_column1], labels=target_labels
    )
    cm2 = confusion_matrix(
        results[target_column], results[output_column2], labels=target_labels
    )

    cm = cm1 - cm2

    cm_melted = _prepare_confusion_matrix(cm, target_labels)

    # cm_melted["color_scale"] = (cm_melted["count"].max() - cm_melted["count"]) / (
    #     cm_melted["count"].max() - cm_melted["count"].min()
    # )
    cm_melted["color_scale"] = cm_melted["count"]

    cm_melted["label"] = cm_melted["count"].astype(str)

    cm_melted["p_group"] = cm_melted["count"].apply(
        lambda x: "high" if x < 0 else "low"
    )

    p = (
        ggplot(
            cm_melted, aes("factor(predicted)", "factor(actual)", fill="color_scale")
        )
        + geom_tile(show_legend=False)
        + geom_text(aes(label="label", color="p_group"), size=12, show_legend=False)
        + ggtitle(title)
        + ylab("True Emotion")
        + xlab("Predicted Emotion")
        + scale_x_discrete(limits=target_labels[::-1])
        + scale_fill_gradient2(low="#ed1213", mid="white", high="#1fae08")
        + scale_color_manual(["black", "black"])
        + theme_bw()
        + theme(
            title=element_text(size=18),
            axis_title=element_text(size=20),  # Increases axis title size
            axis_text=element_text(size=16),  # Increases axis tick label size
            axis_text_x=element_text(rotation=45),  # Rotates x-axis tick labels
        )
    )

    if name is not None:
        p.save(name, width=7, height=7, dpi=300)

    p.show()


def _prepare_confusion_matrix(cm: np.ndarray, target_labels: list[str]) -> pd.DataFrame:
    cm_df = pd.DataFrame(cm, index=target_labels, columns=target_labels)
    cm_melted = cm_df.reset_index().melt(id_vars="index", value_name="count")
    cm_melted.columns = ["actual", "predicted", "count"]
    cm_melted["actual"] = pd.Categorical(cm_melted["actual"], categories=target_labels)
    cm_melted["predicted"] = pd.Categorical(
        cm_melted["predicted"], categories=target_labels
    )

    # Calculate total counts for each actual class
    total_counts = (
        cm_melted.groupby("actual", observed=True)["count"].sum().reset_index()
    )
    total_counts.columns = ["actual", "total_count"]

    # Merge total counts back to the melted DataFrame
    cm_melted = cm_melted.merge(total_counts, on="actual")
    return cm_melted


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
    sample_indexes: list[int],
    collator_type: T_Collator,
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


def classify_sentiment(lab: str, positive: list[str], negative: list[str]) -> str:
    """
    Classify a sentiment label into one of three categories: positive, negative, or neutral.

    Args:
        lab (str): The sentiment label to classify.
        positive (list[str]): A list of sentiment labels that are considered positive.
        negative (list[str]): A list of sentiment labels that are considered negative.

    Returns:
        str: The classified sentiment label.
    """
    if lab in negative:
        return "negative"
    elif lab in positive:
        return "positive"
    return "neutral"


def parse_iemocap_annotations(file_content: str) -> list[dict]:
    label_mapping = {
        "neu": "neutral",
        "ang": "angry",
        "fru": "frustrated",
        "hap": "happy",
        "exc": "excited",
        "sad": "sad",
    }
    # Split the content by lines
    lines = file_content.strip().split("\n")

    # Initialize variables
    results = []

    # Iterate over the lines to find the annotations
    i = 1
    while i < len(lines):
        line = lines[i].strip()

        # Process block's first line
        if line and "[" in line:  # Indicates a new block
            first_line = line.split("\t")
            label = label_mapping.get(
                first_line[2], first_line[2]
            )  # Capitalize the label for comparison
            i += 1

            annotations = []
            # Process the next four lines for annotations
            for _ in range(4):
                annotator_line = lines[i].strip()
                if "\t" in annotator_line:
                    try:
                        emotions = annotator_line.split("\t")[1].split(";")
                        emotions = [e.strip().lower() for e in emotions if e.strip()]
                    except IndexError:
                        emotions = []
                    annotations.append(emotions)
                i += 1

            # Flatten the annotations for first_annotations
            annotations = [emotion for sublist in annotations for emotion in sublist]

            # Calculate agreement score
            occurrences_of_label = annotations.count(label)
            agreement_score = occurrences_of_label / 4  # len(annotations)

            # Store the result in the list
            result = {
                "label": label,  # Keep the original case for the label
                "annotations": annotations,
                "agreement_score": round(agreement_score, 2),
                "spread": len(set(annotations)),
            }
            results.append(result)
        else:
            i += 1  # Skip lines without blocks

    return results
