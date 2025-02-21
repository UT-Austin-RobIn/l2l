import torch
import torch.nn as nn
from typing import Dict, Callable

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest-exact'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=None)
        return x

def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(root_module: nn.Module, features_per_group: int=16) -> nn.Module:
    '''
    Replace all BatchNorm2d with GroupNorm.
    '''
    replace_submodules(
            root_module=root_module,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features//features_per_group, 
                num_channels=x.num_features)
        )
    
    return root_module

def freeze_module(module):
    for child in module.children():
        for param in child.parameters():
            param.requires_grad = False