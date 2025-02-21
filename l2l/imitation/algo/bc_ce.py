import numpy as np
import torch
import torch.nn as nn

from l2l.imitation.algo.base_algo import BaseAlgo
from l2l.imitation.utils.general_utils import AttrDict
from l2l.imitation.models.normalizers import DictNormalizer
from l2l.imitation.models.obs_nets import ObservationEncoder
from l2l.imitation.models.base_nets import MLP
from l2l.imitation.models.distribution_nets import MDN
import torch.distributions as D

from l2l.imitation.utils.obs_utils import process_obs_dict
from l2l.imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from l2l.imitation.utils.file_utils import get_obs_key_to_modality_from_config

class BC_CrossEntropy(BaseAlgo):

    def __init__(self, config):
        super(BC_CrossEntropy, self).__init__()
        
        self.config = config

        policy_config = config.policy_config
        observation_config = config.observation_config
        keys_to_shapes = config.keys_to_shapes
        
        action_dim = keys_to_shapes['ac_dim']

        self.nets = nn.ModuleDict()
        normalizer = DictNormalizer(config.normalization_stats, type='gaussian')
        self.nets["normalizer"] = normalizer
        
        obs_encoder = ObservationEncoder(observation_config, keys_to_shapes['obs_shape'], return_dict=False)
        self.nets["obs_encoder"] = obs_encoder

        self.nets["mlp"] = MLP(
            input_dim=obs_encoder.output_shape(),
            output_dim=policy_config.output_dim,
            hidden_units=policy_config.hidden_units,
            activation=nn.LeakyReLU(0.2),
            output_activation=None
        )
        
        self.nets["action_head"] = MLP(
            input_dim=policy_config.output_dim,
            output_dim=policy_config.action_categories,
            hidden_units=[],
            activation=None,
            output_activation=None
        )

    def get_optimizers_and_schedulers(self, **kwargs):
        optimizer = torch.optim.Adam(self.nets.parameters(), lr=3e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=0.1)
    
        return [optimizer], [lr_scheduler]
        
    def forward(self, batch):
        batch['obs'] = self.nets["normalizer"].normalize(batch['obs'])

        feat = self.nets["obs_encoder"](batch['obs'])
        mlp_out = self.nets["mlp"](feat)
        out = self.nets["action_head"](mlp_out)

        output = {
            "action_logits": out,
            "obs_features": feat
        }

        return output
    
    def compute_loss(self, batch):
        output = self.forward(batch)
        action_logits = output['action_logits']

        losses = AttrDict(total = 0)
        losses.nll = nn.CrossEntropyLoss()(action_logits[:, 0], batch['actions'][:, 0, 0].long())
        
        losses.total += losses.nll

        return losses
    
    def get_loss(self, pred, target):
        return nn.CrossEntropyLoss()(pred, target)

    def preprocess_obs(self, obs):
        obs = process_obs_dict(obs, self.config.keys_to_modality) # divides by 255 and changes hwc->chw
        obs = recursive_dict_list_tuple_apply(
                obs,
                {
                    torch.Tensor: lambda x: x[None].float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x)[None].float().to(self.device),
                    type(None): lambda x: x,
                }
            )
        return obs

    @torch.no_grad()
    def get_action(self, obs):
        self.eval()

        obs = self.preprocess_obs(obs)

        output = self.forward({'obs': obs})
        action = torch.argmax(output['action_logits'], dim=-1, keepdim=True)
        
        self.train() 
        return action[0].cpu().numpy()
        
    def reset(self):
        pass