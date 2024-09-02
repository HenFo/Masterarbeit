import numpy as np
import torch

from .dataset import ERCDataset
from .processor import MmLlamaProcessor


class DynamicPadCollator(object):
    def __init__(self, processor, mode="train", sample_rate=16000):
        self.mode = mode
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, batch):
        inputs = [x[0] for x in batch]
        ys = [x[1] for x in batch]
        inputs = self.processor(
            inputs,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="longest",
        )

        return inputs, torch.Tensor(ys).long()


class SequenceClassificationCollator(object):
    def __init__(
        self, processor: MmLlamaProcessor, dataset: ERCDataset, sample_rate=16000
    ):
        self.processor = processor
        self.sample_rate = sample_rate
        self.dataset = dataset

    def __call__(self, batch):
        acoustics = [a for (a, _), _ in batch]
        texts = [t for (_, t), _ in batch]
        labels = [self.dataset.label2id(l) for _, l in batch]

        processed_inputs = self.processor(
            text=texts,
            acoustic=acoustics,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )

        text_inputs, acoustic_inputs = (
            processed_inputs["text"],
            processed_inputs["acoustic"],
        )

        for k in acoustic_inputs:
            acoustic_inputs[k] = torch.Tensor(np.asarray(acoustic_inputs[k]))

        return {
            "text": text_inputs,
            "acoustic": acoustic_inputs,
            "labels": torch.Tensor(labels).long(),
        }


class SequenceGenerationCollator(object):
    def __init__(self, processor: MmLlamaProcessor, mode="train", sample_rate=16000):
        assert mode in ["train", "dev"]
        self.mode = mode
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, batch) -> torch.Any:
        if self.mode == "dev":
            return self._prepare_dev_data(batch)
        return self._prepare_train_data(batch)

    def _prepare_dev_data(self, batch):
        acoustics = [a for (a, _), _ in batch]
        texts = [t for (_, t), _ in batch]
        labels = [l for _, l in batch]

        processed_inputs = self.processor(
            text=texts,
            acoustic=acoustics,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )

        text_inputs, acoustic_inputs = (
            processed_inputs["text"],
            processed_inputs["acoustic"],
        )

        for k in acoustic_inputs:
            acoustic_inputs[k] = torch.Tensor(np.asarray(acoustic_inputs[k]))

        return {
            "text": text_inputs,
            "acoustic": acoustic_inputs,
        }, labels

    def _prepare_train_data(self, batch):
        # [(audio, text), target]
        acoustics = [a for (a, _), _ in batch]
        texts = [t for (_, t), _ in batch]
        labels = [l for _, l in batch]

        processed_inputs = self.processor(
            text=texts,
            acoustic=acoustics,
            sampling_rate=self.sample_rate,
            return_tensors=None,
        )
        processed_labels = self.processor(
            text=labels, tokenizer_args={"add_special_tokens": False}
        )

        text_inputs, acoustic_inputs, label_inputs = (
            processed_inputs["text"],
            processed_inputs["acoustic"],
            processed_labels["text"],
        )

        # merge text with label
        all_input_ids = []
        all_labels = []
        for text_inp, label_inp in zip(
            text_inputs["input_ids"], label_inputs["input_ids"]
        ):
            label = (
                [-100] * len(text_inp)
                + label_inp
                + [self.processor.tokenizer.eos_token_id]
            )
            input_ids = text_inp + label_inp + [self.processor.tokenizer.eos_token_id]
            all_input_ids.append(input_ids)
            all_labels.append(label)

        # add padding
        longest_text_size = max(map(len, all_input_ids))
        all_attention_masks = []
        for i, (input_ids, labels) in enumerate(zip(all_input_ids, all_labels)):
            padding = [self.processor.tokenizer.unk_token_id] * (
                longest_text_size - len(input_ids)
            )
            attention_mask = [0] * len(padding) + [1] * len(input_ids)
            all_attention_masks.append(attention_mask)
            all_input_ids[i] = padding + input_ids
            all_labels[i] = [-100] * len(padding) + labels

        return {
            "text": {
                "input_ids": torch.Tensor(all_input_ids).long().contiguous(),
                "attention_mask": torch.Tensor(all_attention_masks).long().contiguous(),
                "labels": torch.Tensor(all_labels).long().contiguous(),
            },
            "acoustic": acoustic_inputs,
        }
