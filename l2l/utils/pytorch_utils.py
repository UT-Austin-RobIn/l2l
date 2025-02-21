import torch
import torch.nn as nn

class TensorModule(nn.Module):
    """A dummy module that wraps a single tensor and allows it to be handled like a network (for optimizer etc)."""
    def __init__(self, t):
        super().__init__()
        self.t = nn.Parameter(t)

    def forward(self, *args, **kwargs):
        return self.t
    
def freeze_module(module):
    for param in module.parameters():
        if param.requires_grad:
            param.requires_grad = False
    return module