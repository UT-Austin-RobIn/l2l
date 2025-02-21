import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space['ego_grid'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space['ego_grid'].sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations['ego_grid']
        return self.linear(self.cnn(obs))

class MinigridFeaturesExtractorOneHot(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        self.w_dim = int(observation_space['agent_pos'].high[0])+1
        self.h_dim = int(observation_space['agent_pos'].high[1])+1
        self.linear = nn.Sequential(
                            nn.Linear(self.h_dim + self.w_dim + 4, features_dim),
                            nn.ReLU(),
                            nn.Linear(features_dim, features_dim),
                            nn.ReLU()
                        )

    def process_obs(self, inputs):
        # print(inputs['agent_pos'].shape, inputs['agent_dir'].shape)
        h_one_hot = F.one_hot(inputs['agent_pos'][..., 0].long(), num_classes=self.h_dim)
        w_one_hot = F.one_hot(inputs['agent_pos'][..., 1].long(), num_classes=self.w_dim)
        orn_one_hot = F.one_hot(inputs['agent_dir'][..., 0].long(), num_classes=4,)

        # print(h_one_hot.shape, orn_one_hot.shape, w_one_hot.shape)
        # return torch.cat((inputs['obs']['agent_pos'], inputs['obs']['agent_dir'][:, None]), dim=-1).float()
        return torch.cat((h_one_hot, w_one_hot, orn_one_hot), dim=-1).float()


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = self.process_obs(observations)
        return self.linear(obs)