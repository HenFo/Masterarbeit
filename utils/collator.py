import torch
from .processor import MmLlamaProcessor

class DynamicPadCollator(object):
    def __init__(self, processor, mode="train", sample_rate = 16000):   
        self.mode = mode
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, batch):
        inputs = [x[0] for x in batch]
        ys = [x[1] for x in batch]
        inputs = self.processor(inputs, sampling_rate=self.sample_rate, return_tensors="pt", padding="longest")
        
        return inputs, torch.Tensor(ys).long()
    

class SequenceClassificationCollator(object):
    def __init__(self, processor:MmLlamaProcessor, mode="train", sample_rate = 16000):
        assert mode in ["train", "dev"]
        self.mode = mode
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, batch) -> torch.Any:
        if self.mode == "dev":
            return self.processor(**batch)
        return self._prepare_data(batch)
    
    def _prepare_data(self, batch):
        print(batch)
        return batch

    
