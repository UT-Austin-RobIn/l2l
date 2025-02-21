import os
import pickle
import torch

class RoboMimicPolicy:

    def __init__(self, policy, data_stats_path=None, unnormalize_action_dims=None):
        assert (data_stats_path is None) or (unnormalize_action_dims is not None), "If data_stats_path is provided, unnormalize_action_dims must be provided as well."
        self.policy = policy # this is a RolloutPolicy in RoboMimic
        self.data_stats_path = data_stats_path
        # first n dims to unnormalize
        self.unnormalize_action_dims = unnormalize_action_dims
        
        if data_stats_path is not None:
            self.data_stats = self.load_data_stats(data_stats_path)

    def unnormalize_actions(self, action):
        action = action * self.data_stats['std']['actions'][:self.unnormalize_action_dims] + self.data_stats['mean']['actions'][:self.unnormalize_action_dims]
        return action

    def normalize_obs(self, obs):
        for k in obs.keys():
            if k in self.data_stats['mean']:
                obs[k] = (obs[k] - self.data_stats['mean'][k])/self.data_stats['std'][k]
        return obs

    def get_action(self, obs):
        obs = self.normalize_obs(obs) if self.data_stats_path is not None else obs
        action = self.policy(obs)
        if self.data_stats_path is not None:
            action[:self.unnormalize_action_dims] = self.unnormalize_actions(action[:self.unnormalize_action_dims])

        return action
    
    def start_episode(self):
        self.policy.start_episode()

    def load_data_stats(self, data_stats_path):
        filename = os.path.expanduser(data_stats_path)
        with open(filename, 'rb') as f:
            data_stats = pickle.load(f)
        return data_stats

class UncertaintyPolicyWrapper:

    def __init__(self, policy) -> None:
        self.policy = policy

    def forward(self, obs):
        return self.policy.forward(obs)

    def get_loss(self, pred, target):
        return self.policy.get_loss(pred, target)
    
    def encode_obs(self, batch):
        batch = self.policy.nets["normalizer"].normalize(batch)
        encoded_obs = self.policy.nets['obs_encoder'](batch, return_dict=True)

        return encoded_obs

    def get_actions_from_sampled_privileged(self, batch, privileged_batch):
        encoded_obs = self.encode_obs(batch)

        encoded_batch = []
        for priv in privileged_batch:
            obs = []            
            for k, v in encoded_obs.items():
                if k == 'privileged_info': #reference_cube_color':#
                    obs.append(priv)
                else:
                    obs.append(v.clone())
            encoded_batch.append(torch.cat(obs, dim=-1))
        encoded_batch = torch.cat(encoded_batch, dim=0)

        out = self.policy.nets["action_head"](self.policy.nets["mlp"](encoded_batch))
        return out
    
    def get_loss_from_sampled_privileged(self, batch, privileged_batch, gt_actions):
        actions = self.get_actions_from_sampled_privileged(batch, privileged_batch)
        gt_actions = gt_actions.repeat(privileged_batch.shape[0])
        return self.get_loss(actions, gt_actions).item()
        
class UncertaintyVisuomotorPolicyWrapper:

    def __init__(self, policy, privileged_key) -> None:
        self.policy = policy
        self.privileged_key = privileged_key

    def forward(self, obs):
        return self.policy.forward(obs)
    
    def get_action(self, obs):
        return self.policy.get_action(obs)

    def get_loss(self, pred, target):
        return self.policy.get_loss(pred, target)
    
    def encode_obs(self, batch):
        batch = self.policy.nets["normalizer"].normalize(batch)
        encoded_obs = self.policy.nets['obs_encoder'](batch, return_dict=True)

        return encoded_obs

    # customize it for transformer
    @torch.no_grad
    def get_actions_from_sampled_privileged(self, batch, privileged_batch):
        self.policy.eval()
        encoded_obs = self.encode_obs(batch)
        
        encoded_batch = []
        for p in privileged_batch:
            # encoded_obs[self.privileged_key] = self.policy.nets['obs_encoder'].obs_to_nets[self.privileged_key](p[None].float())
            
            obs = []            
            for k, v in encoded_obs.items():
                if k == self.privileged_key: 
                    obs.append(self.policy.nets['obs_encoder'].obs_to_nets[self.privileged_key](p[None].float()))
                else:
                    obs.append(v.clone())

            obs = torch.cat(obs, dim=-1)
            B, T = obs.shape[:2]
            obs = obs.view(B, T, -1, self.policy.config.policy_config.token_dim)
            encoded_batch.append(obs)
        
        encoded_batch = torch.cat(encoded_batch, dim=0)
        feat = self.policy.temporal_encode(encoded_batch)
        out = self.policy.nets['action_head'](feat, return_states=True)
        self.policy.train()
        return out
    
    def get_loss_from_sampled_privileged(self, batch, privileged_batch, gt_action_dist):
        actions = self.get_actions_from_sampled_privileged(batch, privileged_batch)[0].sample()
        return -gt_action_dist.log_prob(actions).mean().item() # TODO: instead of loss, maybe kl divergence is better
        return self.get_loss(actions, gt_actions).item()