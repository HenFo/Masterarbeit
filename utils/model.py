from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from peft.peft_model import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels = None):
        super().__init__()
        num_labels: int = config.num_labels if num_labels is None else num_labels
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        emotion_embeddings = self.dense(x)
        x = torch.tanh(emotion_embeddings)
        x = self.dropout(x)
        x = self.out_proj(x)
        return emotion_embeddings, x
    

class AudeeringEmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = ClassificationHead(config)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values).last_hidden_state
        mean_output = torch.mean(outputs, dim=1)
        embeddings, logits = self.classifier(mean_output)
        return embeddings, logits
    

class AcousticEmotionRecogniser(Wav2Vec2PreTrainedModel):
    def __init__(self, config, num_labels = None):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = ClassificationHead(config, num_labels)

    def forward(self, **kwargs)-> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(**kwargs).last_hidden_state
        mean_output = torch.mean(outputs, dim=1)
        embeddings, logits = self.classifier(mean_output)
        return embeddings, logits
    
    def freeze_encoder(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> List[torch.nn.Parameter]:
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
        return list(self.wav2vec2.parameters())
    
    def unfreeze_encoder_layers(self, layers:List[int]) -> List[torch.nn.Parameter]:
        params = []
        for name, param in self.named_parameters():
            for layer in layers:
                if f"wav2vec2.encoder.layers.{layer}" in name:
                    param.requires_grad = True
                    params.append(param)
                else:
                    param.requires_grad = False
        return params
    

class ModalityProjector(nn.Module):
    def __init__(self, ac_dim:int, t_dim:int):
        super(ModalityProjector, self).__init__()
        self.proj1 = nn.Linear(ac_dim, t_dim)
        self.ac = nn.SiLU()
        self.proj2 = nn.Linear(t_dim, t_dim)
        
    def forward(self, acoustic_embeddings: torch.Tensor):
        projected = self.proj1(acoustic_embeddings)
        projected = self.ac(projected)
        projected = self.proj2(projected)

        return projected


class MmLlamaConfig(PretrainedConfig):
    def __init__(self, llm_config, audio_config, audio_token_id: int, pad_token_id: int, llm_pretrained_adapter = None):
        self.llm_config = llm_config
        self.llm_pretrained_adapter = llm_pretrained_adapter
        self.audio_config = audio_config
        self.audio_token_id = audio_token_id
        self.ignore_index = -100
        self.pad_token_id = pad_token_id
    

class MMLLaMA(nn.Module):
    def __init__(self, config: MmLlamaConfig) -> None:
        super(MMLLaMA, self).__init__()
        self.config = config
        self.llama = AutoModelForCausalLM.from_pretrained(config.llm_config._name_or_path)
        self.llama.resize_token_embeddings(self.config.audio_token_id + 1, 8)
        if config.llm_pretrained_adapter:
            self.llama = PeftModel.from_pretrained(self.llama, self.config.llm_pretrained_adapter)
            self.llama = self.llama.merge_and_unload()

        self.wave2vec2 = AutoModel.from_pretrained(config.audio_config._name_or_path)

        self.projector = ModalityProjector(1024, 4096)

    def forward(self, input_ids: torch.Tensor, acoustic: Union[torch.Tensor, None] = None, **kwargs):
        inputs = self._get_inputs(input_ids, acoustic, **kwargs)
        return self.llama(**inputs)
    
    def _get_inputs(self, input_ids: torch.Tensor, acoustic: Union[torch.Tensor, None] = None, **kwargs) -> Dict[str, torch.Tensor]:
        text_embeddings: torch.Tensor = self.llama.get_input_embeddings()(input_ids)
        if acoustic is None:
            kwargs["inputs_embeds"] = text_embeddings
            return kwargs
        
        acoustic_embeddings = self.wave2vec2(acoustic).last_hidden_state
        acoustic_embeddings = self.projector(acoustic_embeddings)

        return self._merge_modalities(acoustic_embeddings, text_embeddings, input_ids, labels=input_ids, **kwargs)
    
    def generate(self, input_ids: torch.Tensor, acoustic: Union[torch.Tensor, None] = None, **kwargs):
        inputs = self._get_inputs(input_ids, acoustic, **kwargs)
        inputs_embeds, attention_mask = inputs["inputs_embeds"], inputs["attention_mask"]
        gen_kwargs = {k:v for k,v in kwargs.items() if k not in inputs}
        return self.llama.generate(inputs_embeds = inputs_embeds, attention_mask = attention_mask, **gen_kwargs)

    
    def _merge_modalities(self, audio_features, inputs_embeds, input_ids, attention_mask, labels, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Adapted from the huggingface LLaVA implementation
        @see https://github.com/huggingface/transformers/blob/91d155ea92da372b319a79dd4eef69533ee15170/src/transformers/models/llava/modeling_llava.py#L280C5-L356C81
        """
        _, audio_length, embed_dim = audio_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.config.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_id
        # Compute the maximum embed dimension
        max_embed_dim = audio_length - 1 + sequence_length
        batch_indices, non_audio_indices = torch.where(input_ids != self.config.audio_token_id)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_audio_token_mask * (audio_length - 1) + 1), -1) - 1
        nb_audio_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        audio_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        audio_to_overwrite &= audio_to_overwrite.cumsum(-1) - 1 >= nb_audio_pad[:, None].to(target_device)

        final_embedding[audio_to_overwrite] = audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.config.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return {"inputs_embeds": final_embedding, "attention_mask": final_attention_mask, "labels": final_labels, "position_ids": position_ids}
        
    




if __name__ == "__main__":
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cA = AutoConfig.from_pretrained(
        "/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    )
    cT = AutoConfig.from_pretrained(
        "/home/fock/code/MultiModalInstructERC/models/language/LLaMA2"
    )
    tT = AutoTokenizer.from_pretrained(
        "/home/fock/code/MultiModalInstructERC/models/language/LLaMA2"
    )
    tT.add_special_tokens({"additional_special_tokens": ["<audio>"]})
    tT.pad_token_id = tT.unk_token_id
    tT.padding_side = "left"

    cMM = MmLlamaConfig(
        cT,
        cA,
        tT.encode("<audio>")[-1],
        tT.pad_token_id,
        "/home/fock/code/MultiModalInstructERC/models/language/adapter/InstructERC_unbalanced",
    )

    m = MMLLaMA(cMM)
    inpA = torch.zeros((1, 32000))
    inpT = tT(
        '<audio> Now you are expert of sentiment and emotional analysis. The following conversation noted between \'### ###\' involves several speaker. ### Speaker_0: "Why do all youre coffee mugs have numbers on the bottom?" \t Speaker_1: "Oh. Thats so Monica can keep track. That way if one on them is missing, she can be like, \'Wheres number 27?!" ### Please select the emotional label of <Speaker_1: "Oh. Thats so Monica can keep track. That way if one on them is missing, she can be like, \'Wheres number 27?!"> from <surprise, anger, neutral, joy, sadness, fear, disgust>:',
        return_tensors="pt",
    )

    r = m.generate(acoustic=inpA, **inpT)
    print(tT.batch_decode(r, skip_special_tokens=True))
