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

class BCCrossEntropyEnsemble(BaseAlgo):

    def __init__(self, config):
        super(BCCrossEntropyEnsemble, self).__init__()
        
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

        self.nets["ensemble"] = nn.ModuleList([
            MLP(
                input_dim=obs_encoder.output_shape(),
                output_dim=policy_config.action_categories,
                hidden_units=policy_config.hidden_units + [policy_config.output_dim],
                activation=nn.LeakyReLU(0.2),
                output_activation=None
            ) for _ in range(policy_config.n_ensemble)
        ])

        self.n_ensemble = policy_config.n_ensemble

    def get_optimizers_and_schedulers(self, **kwargs):
        optimizer = torch.optim.Adam(self.nets.parameters(), lr=3e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=0.1)
    
        return [optimizer], [lr_scheduler]
        
    def forward(self, batch):
        batch['obs'] = self.nets["normalizer"].normalize(batch['obs'])

        feat = self.nets["obs_encoder"](batch['obs'])
        outputs = []
        for i in range(self.n_ensemble):
            out = self.nets["ensemble"][i](feat)
            outputs.append(out)
    
        output = {
            "action_logits": torch.stack(outputs, dim=0),
            "obs_features": feat
        }

        return output
    
    def compute_loss(self, batch):
        output = self.forward(batch)
        action_logits = output['action_logits']

        action_logits = action_logits.view(-1, *action_logits.shape[2:])
        target_actions = batch['actions'][:, 0, 0].repeat(self.n_ensemble).long()
        
        losses = AttrDict(total = 0)
        losses.nll = nn.CrossEntropyLoss()(action_logits[:, 0], target_actions)
        
        losses.total += losses.nll

        return losses
    
    def get_loss(self, pred, target):    
        action_logits = pred.view(-1, *pred.shape[2:])
        target_actions = target.repeat(self.n_ensemble).long()
        return nn.CrossEntropyLoss()(action_logits, target_actions)

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
        action_logits = torch.mean(output['action_logits'], dim=0)
        action = torch.argmax(action_logits, dim=-1, keepdim=True)
        
        self.train() 
        action = action[0].cpu().numpy()
        return action
        
    def reset(self):
        pass