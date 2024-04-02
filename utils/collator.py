import torch

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