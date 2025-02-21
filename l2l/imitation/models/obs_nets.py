import numpy as np
import torch
import torch.nn as nn
from l2l.imitation.models.base_nets import MLP
from collections import OrderedDict

class ObservationEncoder(nn.Module):

    def __init__(
            self,
            config,
            keys_to_shapes,
            return_dict=False,
        ):
        super(ObservationEncoder, self).__init__()
        self.return_dict = return_dict
        self.keys_to_shapes = keys_to_shapes

        self.obs_to_modality = {}
        self.modality_to_aug = nn.ModuleDict()
        self.obs_to_nets = nn.ModuleDict()
        for modality, obs_keys in config.obs.items():
            
            if 'augmentation' in config.encoder[modality].keys() and \
                len(config.encoder[modality].augmentation) > 0 and \
                len(obs_keys) > 0:

                aug_list = []
                for aug in config.encoder[modality].augmentation:
                    aug_list.append(
                        aug.aug_class(input_shape=self.keys_to_shapes[obs_keys[0]], **aug.aug_kwargs)
                    ) # assuming all obs of each modality are of same shape
                self.modality_to_aug[modality] = nn.Sequential(*aug_list)

            for obs_key in obs_keys:
                self.obs_to_modality[obs_key] = modality

                if modality == 'low_dim':
                    if config.encoder[modality].core_class is None:
                        self.obs_to_nets[obs_key] = IdentityObsEncoder(input_shape=self.keys_to_shapes[obs_key])
                    else:
                        self.obs_to_nets[obs_key] = config.encoder[modality].core_class(
                            input_shape=self.keys_to_shapes[obs_key], **config.encoder[modality].core_kwargs
                        )

                elif modality == 'rgb':
                    self.obs_to_nets[obs_key] = config.encoder[modality].core_class(
                            input_shape=self.keys_to_shapes[obs_key], **config.encoder[modality].core_kwargs
                        )
                    
                elif modality == 'depth':
                    self.obs_to_nets[obs_key] = config.encoder[modality].core_class(
                            input_shape=self.keys_to_shapes[obs_key], **config.encoder[modality].core_kwargs
                        )
                else:
                    raise ValueError(f"Modality {modality} not supported")
                    

    def output_shape(self):
        output_shape_map = {k: v.output_shape() for k, v in self.obs_to_nets.items()}

        if self.return_dict:
            return output_shape_map
        
        feture_dim = 0
        for k, v in output_shape_map.items():
            feture_dim += np.prod(v).astype(int)

        return feture_dim
    
    def apply_augmentation(self, obs_key, obs):
        modality = self.obs_to_modality[obs_key]
        if self.training and modality in self.modality_to_aug.keys():
            return self.modality_to_aug[modality](obs)
        
        return obs

    def forward(self, obs_dict, return_dict=False):
        if self.return_dict or return_dict:
            ret = OrderedDict()
            for k, v in self.obs_to_nets.items():
                obs = self.apply_augmentation(k, obs_dict[k])
                ret[k] = v(obs)
            return ret
            # return {k: v(obs_dict[k]) for k, v in self.obs_to_nets.items()}
        else:
            # Handles 1 dimensional observation
            output = []
            for k, v in self.obs_to_nets.items():
                obs = self.apply_augmentation(k, obs_dict[k])
                out = v(obs)
                if len(out.shape) == 1:
                    out = out.unsqueeze(0)
                output.append(out)
            return torch.cat(output, dim=-1)

class VisionCore(nn.Module):
    def __init__(self,
                 input_shape,
                 backbone_class,
                 pool_class,
                 backbone_kwargs=None,
                 pool_kwargs=None,
                 feature_dim=64):

        super(VisionCore, self).__init__()

        self.input_shape = input_shape
        self.feature_dim = feature_dim

        if backbone_kwargs is None:
            backbone_kwargs = {}
        
        backbone_kwargs['input_channel'] = self.input_shape[0]
        backbone = backbone_class(**backbone_kwargs)
        net_list = [backbone]
        feat_shape = backbone.output_shape(self.input_shape)
        
        # pooling skipped for now
        if pool_class is not None:
            pool_kwargs['input_shape'] = feat_shape
            net_list.append(pool_class(**pool_kwargs))
            feat_shape = net_list[-1].output_shape(feat_shape)

        net_list.append(nn.Flatten(start_dim=1, end_dim=-1))
        if self.feature_dim is not None:
            net_list.append(nn.Linear(np.prod(feat_shape), self.feature_dim))

        self.nets = nn.Sequential(*net_list)
    
    def output_shape(self):
        if self.feature_dim is None:
            return [np.prod(self.nets[-2].output_shape(self.input_shape))]
        return [self.feature_dim]

    def forward(self, inputs):
        inp_shape = inputs.shape
        extra_dims = inp_shape[:len(inp_shape) - len(self.input_shape)]
        
        inputs = inputs.view(-1, *self.input_shape)
        
        out = self.nets(inputs).view(*extra_dims, *self.output_shape())
        return out
    
class IdentityObsEncoder(nn.Module):
    
    def __init__(self, input_shape):
        super(IdentityObsEncoder, self).__init__()
        self.input_shape = input_shape

    def output_shape(self):
        return self.input_shape
        
    def forward(self, obs):
        extra_dims = obs.shape[:len(obs.shape) - len(self.input_shape)]
        return obs.view(*extra_dims, *self.input_shape)
    

class LowDimCore(MLP):

    def __init__(self, input_shape, feature_dim, hidden_units=[32, 32], normalization=None, activation=nn.ReLU(), output_activation=None):
        super(LowDimCore, self).__init__(input_dim=input_shape[0], output_dim=feature_dim, hidden_units=hidden_units, normalization=normalization, activation=activation, output_activation=output_activation)