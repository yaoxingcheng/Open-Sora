import torch

from opensora.registry import MODELS


@MODELS.register_module("classes")
class ClassEncoder:
    def __init__(self, num_classes, model_max_length=None, device="cuda", dtype=torch.float):
        self.num_classes = num_classes
        self.y_embedder = None

        self.model_max_length = model_max_length
        self.output_dim = num_classes
        self.device = device
        self.dtype=dtype

    def encode(self, text):
        cids = [torch.nn.functional.one_hot(torch.tensor(int(t)), num_classes=self.num_classes).unsqueeze(0) for t in text]
        return dict(y=torch.stack(cids, dim=0).unsqueeze(1).to(self.dtype).to(self.device))

    def null(self, n):
        return torch.tensor([self.num_classes] * n).to(self.device)
