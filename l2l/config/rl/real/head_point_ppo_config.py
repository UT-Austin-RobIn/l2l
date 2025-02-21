from l2l.utils.general_utils import AttrDict
from l2l.config.env.real.tiago_head_env import env_config
from stable_baselines3.common.monitor import Monitor
from l2l.wrappers.real_wrappers import TiagoPointHeadWrapper
from stable_baselines3 import PPO
from l2l.modules.real_feature_extractor import RealFeatureExtractor

environment_config = AttrDict(
    env_config=env_config,
    wrappers=[TiagoPointHeadWrapper, Monitor],
    num_envs=1,
)

model_config = AttrDict(
    model_class=PPO,
    model_config=AttrDict(
        policy="MultiInputPolicy",
        policy_kwargs=dict(features_extractor_class=RealFeatureExtractor),
        n_steps=50,
        gamma=0.95,
        learning_rate=1e-4,
        # ent_coef=0.03,
        batch_size=50,
        # clip_range=0.05,
        verbose=1
    )
)

# This is actually env steps
train_config = AttrDict(
    n_epochs=1e5
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