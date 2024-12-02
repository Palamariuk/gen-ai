import torch
from torch import nn


# Define a generic module
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def configure_optimizers(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")

    def training_step(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")

    def set_device(self, device):
        self.device = device
        self.to(device)
