# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import numpy as np
import torch
from torch import TensorType
from transformers import BatchFeature


class MmLlamaProcessor:
    """
    Processor class for MMLLaMA.
    Adapted from the Llava Processor implementation
    """

    def __init__(self, audio_processor=None, tokenizer=None):
        # super().__init__(feature_extractor=audio_processor, tokenizer=tokenizer)
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str] = None,
        acoustic: List[torch.Tensor] = None,
        sampling_rate: int = 16000,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        tokenizer_args: dict = {},
    ) -> BatchFeature:
        if acoustic is not None:
            def clean_audio(acoustic: np.ndarray|None):
                if acoustic is None:
                    return np.zeros((4000,))
                audio_length = acoustic.shape[0]
                if audio_length < 4000:
                    return np.concatenate((acoustic, np.zeros((4000-audio_length,))))
                return acoustic
            
            acoustic = list(map(clean_audio, acoustic))
            audio_features = self.audio_processor(
                acoustic, sampling_rate=sampling_rate, return_tensors="pt", padding=True
            )
            empty_batch_indices, *_ = torch.where(torch.all(audio_features["input_values"] == 0, dim=-1))
            audio_features["attention_mask"][empty_batch_indices,:] = 0

        else:
            audio_features = None
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **tokenizer_args,
            )
        else:
            text_inputs = None

        return BatchFeature(data={"text": text_inputs, "acoustic": audio_features})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
