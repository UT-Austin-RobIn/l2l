import torch
import torch.nn as nn

class DictNormalizer(nn.Module):

    def __init__(self, normalization_stats, type='gaussian', device=None) -> None:
        super(DictNormalizer, self).__init__()

        if type == 'gaussian':
            normalizer_type = GaussianNormalizer
        else:
            raise ValueError(f"Normalizer type {type} not recognized")
    
        self.normalizers = nn.ModuleDict()
        for key, value in normalization_stats.items():
            self.normalizers[key] = normalizer_type(key, value)

    def normalize(self, x):
        for key in x.keys():
            if key in self.normalizers:
                x[key] = self.normalize_by_key(x[key], key)
        return x
    
    def normalize_by_key(self, x, key):
        return self.normalizers[key].normalize(x)

    def denormalize(self, x):
        for key in x.keys():
            if key in self.normalizers:
                x[key] = self.denormalize_by_key(x[key], key)
        return x
    
    def denormalize_by_key(self, x, key):
        return self.normalizers[key].denormalize(x)

class GaussianNormalizer(nn.Module):

    def __init__(self, key, normalization_stat) -> None:
        super(GaussianNormalizer, self).__init__()

        self.key = key
        self.register_buffer('mean', torch.from_numpy(normalization_stat['mean']))
        self.register_buffer('std', torch.from_numpy(normalization_stat['std']))

    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def denormalize(self, x):
        return x * self.std + self.mean
    
    def __repr__(self):
        return f"GaussianNormalizer({self.key})\nmean: {self.mean}\nstd: {self.std}\n"