import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th


class RobosuiteFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(RobosuiteFeatureExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if key in self.state_keys:  # "proprioception",
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())
            elif key in self.cnn_keys:
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            else:
                continue
                # raise ValueError("Unknown observation key: %s" % key)
            total_concat_size += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key in self.cnn_keys:
                observations[key] = observations[key].permute((0, 3, 1, 2))/255.0
            # print("Extractor", torch.sum(observations[key][:, 1:] - observations['stage_id']).item())
            encoded_tensor_list.append(extractor(observations[key]))
        
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

class KitchenFeatureExtractor(RobosuiteFeatureExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        self.cnn_keys = ["activeview_image", "robot0_eye_in_hand_image"]
        self.state_keys = ["camera_angle", "robot0_eef_pos", "robot0_gripper_qpos", "meatball_cook_status", "butter_melt_status", "is_policy_uncertain"]

        super(KitchenFeatureExtractor, self).__init__(observation_space)

class MultiStageFeatureExtractor(RobosuiteFeatureExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        self.cnn_keys = ["activeview_image", "robot0_eye_in_hand_image"]
        self.state_keys = ["camera_angle", "robot0_eef_pos", "robot0_gripper_qpos", "is_policy_uncertain"]

        super(MultiStageFeatureExtractor, self).__init__(observation_space)

class WalledMultiStageFeatureExtractor(RobosuiteFeatureExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        self.cnn_keys = ["activeview_image", "robot0_eye_in_hand_image"]
        self.state_keys = ["camera_angle", "robot0_eef_pos", "robot0_gripper_qpos", "is_policy_uncertain"]

        super(WalledMultiStageFeatureExtractor, self).__init__(observation_space)

class TwoArmFeatureExtractor(RobosuiteFeatureExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        self.cnn_keys = ["activeview_image", "robot0_eye_in_hand_image"]
        self.state_keys = ["camera_angle", "robot0_eef_pos", "robot0_gripper_qpos", "robot1_eef_pos", "robot1_gripper_qpos", "is_policy_uncertain"]

        super(TwoArmFeatureExtractor, self).__init__(observation_space)

class KitchenFullRLBaselineFeatureExtractor(RobosuiteFeatureExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        self.cnn_keys = ["activeview_image", "robot0_eye_in_hand_image"]
        self.state_keys = ["camera_angle", "robot0_eef_pos", "robot0_gripper_qpos", "meatball_cook_status", "butter_melt_status"]

        super(KitchenFullRLBaselineFeatureExtractor, self).__init__(observation_space)

class MultiStageFullRLBaselineFeatureExtractor(RobosuiteFeatureExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        self.cnn_keys = ["activeview_image", "robot0_eye_in_hand_image"]
        self.state_keys = ["camera_angle", "robot0_eef_pos", "robot0_gripper_qpos"]

        super(MultiStageFullRLBaselineFeatureExtractor, self).__init__(observation_space)

class WalledMultiStageFullRLBaselineFeatureExtractor(RobosuiteFeatureExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        self.cnn_keys = ["activeview_image", "robot0_eye_in_hand_image"]
        self.state_keys = ["camera_angle", "robot0_eef_pos", "robot0_gripper_qpos"]

        super(WalledMultiStageFullRLBaselineFeatureExtractor, self).__init__(observation_space)

class TwoArmFullRLBaselineFeatureFeatureExtractor(RobosuiteFeatureExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        self.cnn_keys = ["activeview_image", "robot0_eye_in_hand_image"]
        self.state_keys = ["camera_angle", "robot0_eef_pos", "robot0_gripper_qpos", "robot1_eef_pos", "robot1_gripper_qpos"]

        super(TwoArmFullRLBaselineFeatureFeatureExtractor, self).__init__(observation_space)