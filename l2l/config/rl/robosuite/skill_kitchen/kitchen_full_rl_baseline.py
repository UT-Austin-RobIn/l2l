from l2l.utils.general_utils import AttrDict
from l2l.envs.robosuite.camera_color_lift import CameraOracleColorLift
from l2l.config.env.robosuite.skill_kitchen import env_config
from stable_baselines3.common.monitor import Monitor
from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper, FullRLBaselineWrapper
from stable_baselines3 import PPO
from l2l.modules.robosuite_feature_extractor import KitchenFullRLBaselineFeatureExtractor

environment_config = AttrDict(
    env_config=env_config,
    wrappers=[RobosuiteGymWrapper, FullRLBaselineWrapper, Monitor],
    num_envs=16,#24
)

model_config = AttrDict(
    model_class=PPO,
    model_config=AttrDict(
        policy="MultiInputPolicy",
        policy_kwargs=dict(
            features_extractor_class=KitchenFullRLBaselineFeatureExtractor,
            # features_extractor_kwargs=dict(features_dim=128),
        ),
        n_steps=300,
        gamma=0.99,
        learning_rate=1e-4,
        # ent_coef=0.03,
        batch_size=64,
        # clip_range=0.05,
        verbose=1
    )
)

train_config = AttrDict(
    n_epochs=1e6,
    rl_save_freq=625
)

sb3_callback_config = None #AttrDict(
#     reward_threshold=3,
#     eval_freq=int(3e2),
#     n_eval_envs=environment_config.num_envs,
# )

evaluator_config = None

config = AttrDict(
    env_config=environment_config,
    model_config=model_config,
    train_config=train_config,
    evaluator_config=evaluator_config,
    sb3_callback_config=sb3_callback_config,
)