import numpy as np
import torch
import torch.nn as nn

from l2l.imitation.algo.base_algo import BaseAlgo
from l2l.imitation.utils.general_utils import AttrDict
from l2l.imitation.models.normalizers import DictNormalizer
from l2l.imitation.models.obs_nets import ObservationEncoder
from l2l.imitation.models.base_nets import MLP
from l2l.imitation.models.distribution_nets import MDN
from l2l.imitation.models.transformer_modules import SinusoidalPositionEncoding, TransformerDecoder
import torch.distributions as D

from l2l.imitation.utils.obs_utils import process_obs_dict
from l2l.imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from l2l.imitation.utils.file_utils import get_obs_key_to_modality_from_config

class BCTransformer(BaseAlgo):

    def __init__(self, config):
        super(BCTransformer, self).__init__()
        
        self.config = config

        policy_config = config.policy_config
        observation_config = config.observation_config
        keys_to_shapes = config.keys_to_shapes
        
        action_dim = keys_to_shapes['ac_dim']

        self.nets = nn.ModuleDict()
        normalizer = DictNormalizer(config.normalization_stats, type='gaussian')
        self.nets["normalizer"] = normalizer
        
        # make sure outputs of all obs encoder have same feature dim as transformer's feature dim
        for modality in observation_config.encoder.keys():
            assert len(observation_config.obs[modality]) == 0 or policy_config.token_dim == observation_config.encoder[modality].core_kwargs['feature_dim'], f"Token dim of transformer should be same as feature dim of obs encoder. Expected {policy_config.token_dim}, got {observation_config.encoder[modality].core_kwargs['feature_dim']}"

        # observation encoder
        obs_encoder = ObservationEncoder(observation_config, keys_to_shapes['obs_shape'], return_dict=False)
        self.nets["obs_encoder"] = obs_encoder

        # position encoder
        self.nets["pos_encoder"] = SinusoidalPositionEncoding(input_size=policy_config.token_dim, inv_freq_factor=10)

        # tranformer
        self.nets["transformer"] = TransformerDecoder(
            input_size=policy_config.token_dim,
            num_layers=policy_config.transformer_num_layers,
            num_heads=policy_config.transformer_num_heads,
            head_output_size=policy_config.transformer_head_output_size,
            mlp_hidden_size=policy_config.transformer_mlp_hidden_size,
            dropout=policy_config.transformer_dropout,
        )
        
        
        self.nets["action_head"] = policy_config.action_head(
                    input_size=policy_config.token_dim,
                    output_size=action_dim,
                    has_time_dimension=True,
                    **policy_config.action_head_kwargs
                )

        self.latent_queue = []
        self.horizon = policy_config.horizon

    def get_optimizers_and_schedulers(self, **kwargs):
        assert 'num_epochs' in kwargs, 'num_epochs is required for DiffusionPolicy optimizer initialization'
        assert 'epoch_every_n_steps' in kwargs, 'epoch_every_n_steps is required for DiffusionPolicy optimizer initialization'

        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=3e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs['num_epochs']*kwargs['epoch_every_n_steps'], eta_min=1e-5)

        return [optimizer], [lr_scheduler]
    
    def spatial_encode(self, obs):
        # encode observations
        obs_feat = self.nets["obs_encoder"](obs) # (B, T, num_modality*token_dim)
        B, T = obs_feat.shape[:2]
        obs_feat = obs_feat.view(B, T, -1, self.config.policy_config.token_dim) # (B, T, num_modality, token_dim)

        return obs_feat
    
    def temporal_encode(self, feat):
        B, T = feat.shape[:2]

        # encode position
        pos = self.nets["pos_encoder"](feat)
        feat = feat + pos.unsqueeze(1)

        # transformer
        feature_shape = feat.shape
        self.nets['transformer'].compute_mask(feat.shape)
        feat = feat.view(B, -1, self.config.policy_config.token_dim)
        feat = self.nets['transformer'](feat)
        feat = feat.view(*feature_shape)[:, :, 0] # (B, T, num_modality, token_dim) -> (B, T, token_dim)

        return feat
        
    def forward(self, batch, hidden_state=None, get_hidden_state=False):
        batch['obs'] = self.nets["normalizer"].normalize(batch['obs'])

        obs_feat = self.spatial_encode(batch['obs'])
        feat = self.temporal_encode(obs_feat)

        # action head
        out = self.nets["action_head"](feat)

        output = {
            "action_dist": out,
            "transformer_out": feat,
            "obs_features": obs_feat
        }

        return output
    
    def compute_loss(self, batch):
        output = self.forward(batch)
        action_dist = output['action_dist']

        batch['actions'] = self.nets["normalizer"].normalize_by_key(batch['actions'], 'actions')

        losses = AttrDict(total = 0)
        losses.nll = -action_dist.log_prob(batch['actions']).mean()
        losses.total += losses.nll

        return losses
    
    def preprocess_obs(self, obs):
        obs = process_obs_dict(obs, self.config.keys_to_modality) # divides by 255 and changes hwc->chw
        
        obs = recursive_dict_list_tuple_apply(
                obs,
                {
                    torch.Tensor: lambda x: x[None, None].float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x)[None, None].float().to(self.device),
                    type(None): lambda x: x,
                }
            )
        return obs

    @torch.no_grad()
    def get_action(self, obs):
        self.eval()

        obs = self.preprocess_obs(obs)
        obs = self.nets["normalizer"].normalize(obs)
        
        x = self.spatial_encode(obs) # (B, T, num_modality, token_dim)
        self.latent_queue.append(x)
        if len(self.latent_queue) > self.horizon:
            self.latent_queue.pop(0)
        
        x = torch.cat(self.latent_queue, dim=1)  # (B, T, modalities, token_dim)
        x = self.temporal_encode(x)

        dist = self.nets['action_head'](x[:, -1])
        action = dist.sample()

        action =  self.nets["normalizer"].denormalize_by_key(action, 'actions')

        self.train() 
        return action[0].cpu().numpy()
        
    def reset(self):
        self.latent_queue = []