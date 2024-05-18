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

import torch
from torch import TensorType
from transformers import BatchFeature, ProcessorMixin


class MmLlamaProcessor(ProcessorMixin):
    """
    Processor class for MMLLaMA.
    Adapted from the Llava Processor implementation
    """


    def __init__(self, audio_processor=None, tokenizer=None):
        super().__init__(audio_processor, tokenizer)
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: str = None,
        acoustic: torch.Tensor = None,
        sampling_rate: int = 16000,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
    ) -> BatchFeature:
        if acoustic is not None:
            audio_features = self.audio_processor(acoustic, sampling_rate=sampling_rate, return_tensors=return_tensors)
        else:
            audio_features = None
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        return BatchFeature(data={**text_inputs, "acoustic": audio_features})

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