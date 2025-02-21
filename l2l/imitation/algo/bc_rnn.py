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

class BC_RNN(BaseAlgo):

    def __init__(self, config):
        super(BC_RNN, self).__init__()
        
        self.config = config

        policy_config = config.policy_config
        observation_config = config.observation_config
        keys_to_shapes = config.keys_to_shapes
        
        action_dim = keys_to_shapes['ac_dim']

        self.nets = nn.ModuleDict()
        normalizer = DictNormalizer(config.normalization_stats, type='gaussian')
        self.nets["normalizer"] = normalizer
        
        obs_encoder = ObservationEncoder(observation_config, keys_to_shapes['obs_shape'], return_dict=False)
        self.nets["obs_encoder"] = nn.Sequential(
                    obs_encoder,
                    MLP(
                        input_dim=obs_encoder.output_shape(),
                        output_dim=obs_encoder.output_shape(),
                        hidden_units=[obs_encoder.output_shape()],
                        activation=nn.LeakyReLU(0.2),
                        output_activation=None
                    )
                )

        self.nets["rnn"] = nn.LSTM(
                input_size=obs_encoder.output_shape(),
                hidden_size=policy_config.hidden_dim,
                num_layers=policy_config.num_layers,
                batch_first=True
            )
        
        self.nets["action_head"] = policy_config.action_head(
                    input_size=policy_config.hidden_dim,
                    output_size=action_dim,
                    has_time_dimension=True,
                    **policy_config.action_head_kwargs
                )

        # self.nets["action_head"] = nn.Sequential(
        #         MLP(
        #             input_dim=policy_config.hidden_dim,
        #             output_dim=policy_config.hidden_dim,
        #             hidden_units=[policy_config.hidden_dim],
        #             activation=nn.LeakyReLU(0.2),
        #             output_activation=None
        #         ),
        #         MDN(
        #             input_size=policy_config.hidden_dim,
        #             output_size=action_dim,
        #             num_gaussians=policy_config.num_gaussians,
        #             has_time_dimension=True
        #         )
        # )

        self._rnn_hidden_state = None
        self.horizon = policy_config.horizon
        self._rnn_counter = 0

        self.classifier = None#MLP(
        #     input_dim=policy_config.hidden_dim,#obs_encoder.output_shape(),
        #     output_dim=1,
        #     hidden_units=[policy_config.hidden_dim]*2,#[obs_encoder.output_shape()]*2,
        #     activation=nn.LeakyReLU(0.2),
        #     output_activation=nn.Sigmoid()
        # )

    def get_optimizers_and_schedulers(self, **kwargs):
        optimizer = torch.optim.Adam(self.nets.parameters(), lr=3e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=0.1)

        # classifier optim
        if self.classifier is not None:
            classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=3e-4)
            classifier_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(classifier_optimizer, milestones=[25000], gamma=0.5)
            return [optimizer, classifier_optimizer], [lr_scheduler, classifier_lr_scheduler]
    
        return [optimizer], [lr_scheduler]
        
    def forward(self, batch, hidden_state=None, get_hidden_state=False):
        batch['obs'] = self.nets["normalizer"].normalize(batch['obs'])

        feat = self.nets["obs_encoder"](batch['obs'])
        rnn_out, (h, c) = self.nets["rnn"](feat, hidden_state)
        out = self.nets["action_head"](rnn_out)

        output = {
            "action_dist": out,
            "rnn_out": rnn_out,
            "obs_features": feat
        }

        if get_hidden_state:
            return output, (h, c)
        return output
    
    def compute_loss(self, batch):
        output = self.forward(batch)
        action_dist = output['action_dist']

        batch['actions'] = self.nets["normalizer"].normalize_by_key(batch['actions'], 'actions')

        losses = AttrDict(total = 0)

        if 'idx' in batch:
            log_prob = -action_dist.log_prob(batch['actions']).mean(-1)
            lt_5t_mask = batch['idx']<=5
            gt_5t_mask = batch['idx']>5
            if lt_5t_mask.sum() > 0:
                losses.first_5_timesteps = torch.sum(lt_5t_mask*log_prob)/lt_5t_mask.sum()
            if gt_5t_mask.sum() > 0:
                losses.after_5_timesteps = torch.sum(gt_5t_mask*log_prob)/gt_5t_mask.sum()
        
        losses.nll = -action_dist.log_prob(batch['actions']).mean()
        losses.total += losses.nll
        
        ### classifier loss
        if self.classifier is not None:
            rnn_out = output['rnn_out']#.detach()
            classifier_out = self.classifier(rnn_out)

            target = batch['obs']['reference_cube_color'][..., 0].unsqueeze(-1)
            losses.classifier = nn.BCELoss()(classifier_out, target)
            
            losses.total += losses.classifier
        ###

        return losses
    
    def get_rnn_init_state(self):
        h_0 = torch.zeros(self.config.policy_config.num_layers, self.config.policy_config.hidden_dim, device=self.device)
        c_0 = torch.zeros(self.config.policy_config.num_layers, self.config.policy_config.hidden_dim, device=self.device)

        return (h_0, c_0)
    
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

        if self._rnn_counter % self.horizon == 0:
            self._rnn_hidden_state = self.get_rnn_init_state()

        self._rnn_counter += 1
        output, self._rnn_hidden_state = self.forward({'obs': obs}, hidden_state=self._rnn_hidden_state, get_hidden_state=True)
        action = output['action_dist'].sample()

        action = self.nets["normalizer"].denormalize_by_key(action, 'actions') 

        self.train() 
        return action[0].cpu().numpy()
        
    def reset(self):
        self._rnn_hidden_state = None
        self._rnn_counter = 0