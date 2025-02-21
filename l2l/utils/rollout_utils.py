import numpy as np
import torch
from l2l.utils.general_utils import recursive_map_dict
import copy
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

class SB3MinigridRolloutPolicyWrapper:

    def __init__(self, policy, deterministic=True):
        self.policy = policy
        self.deterministic = deterministic

    def get_action(self, inp):
        inp['ego_grid'] = np.moveaxis(inp['ego_grid'], -1, -3)
        action, _ = self.policy.predict(inp, deterministic=self.deterministic)

        return action.item()
    
    def get_value(self, inp):
        inp['ego_grid'] = np.moveaxis(inp['ego_grid'], -1, -3)
        inp = recursive_map_dict(lambda x: torch.tensor(x, device='cuda')[None], inp)
        _, value, _ = self.policy(inp)

        return value.item()
    
class SB3RobosuiteRolloutPolicyWrapper:

    def __init__(self, policy, deterministic=True):
        self.policy = policy
        self.deterministic = deterministic

    def get_action(self, inp):
        action, _ = self.policy.predict(inp, deterministic=self.deterministic)
        return int(action)
    
    def get_value(self, inp):
        inp = recursive_map_dict(lambda x: torch.tensor(x, device='cuda')[None], inp)
        _, value, _ = self.policy(inp)

        return value.item()

class RobomimicMinimalPolicyWrapper:

    '''
        A minimal policy wrapper essential for running the policy in parallel on multiple threads using SB3
    '''

    def __init__(self, model, obs_normalization_stats=None):
        self.nets = model.nets
        self.device = model.device
        self.obs_normalization_stats = obs_normalization_stats

        self._rnn_horizon = model._rnn_horizon
        self.reset()
        self.set_eval()

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training

        obs_dict = self._prepare_observation(obs_dict)

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

        obs_to_use = obs_dict

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action[0].cpu().numpy()

    def reset(self):
        self._rnn_hidden_state = None
        self._rnn_counter = 0
    
    def set_eval(self):
        self.nets.eval()

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        self.set_eval()
        self.reset()

    def _prepare_observation(self, ob):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
        """
        ob = copy.deepcopy(ob)
        for k in ['agentview_image', 'robot0_eye_in_hand_image']:
            ob[k] = ObsUtils.process_frame(ob[k][::-1], channel_dim=3, scale=255.0)

        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.device)
        ob = TensorUtils.to_float(ob)
        if self.obs_normalization_stats is not None:
            # ensure obs_normalization_stats are torch Tensors on proper device
            obs_normalization_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(self.obs_normalization_stats), self.policy.device))
            # limit normalization to obs keys being used, in case environment includes extra keys
            ob = { k : ob[k] for k in self.policy.global_config.all_obs_keys }
            ob = ObsUtils.normalize_obs(ob, obs_normalization_stats=obs_normalization_stats)
        return ob