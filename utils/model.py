from typing import List
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torch
import torch.nn as nn


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
    

class MyEmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config, num_labels = None):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = ClassificationHead(config, num_labels)

    def forward(self, **kwargs):
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