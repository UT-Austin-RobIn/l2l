from l2l.utils.general_utils import AttrDict
from l2l.config.env.real.tiago_head_env import env_config
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from l2l.modules.real_feature_extractor import TeatimeFeatureExtractor
from l2l.modules.image_model import Image2ActionModelLarge
from l2l.wrappers.real_wrappers import SimAndRealUncertaintyAwareWrapper, DisplayStringWrapper, DisplayImageWrapper

environment_config = AttrDict(
    env_config=env_config,
    wrappers=[SimAndRealUncertaintyAwareWrapper, DisplayImageWrapper, Monitor],
    num_envs=1,
)

if env_config.pi_format == "clip":
    if env_config.partitioned_encoding:
        privileged_dim = 512
        discrete_privileged = False
        bc_ckpt = ""
    else:
        privileged_dim = 512
        discrete_privileged = False
        bc_ckpt = ""
elif env_config.pi_format == "onehot":
    privileged_dim = None
    discrete_privileged = True
    bc_ckpt = "/home/shivin/learning2look/experiments/imitation/skill_color_pnp_for_real/weights/weights_ep15.pth"
elif env_config.pi_format == "continuous":  # this is the simple continuous
    privileged_dim = 2
    discrete_privileged = False
    # TODO: add bc_ckpt

rl_model_config = AttrDict(
    model_class=PPO,
    model_kwargs=AttrDict(
        policy="MultiInputPolicy",
        policy_kwargs=dict(features_extractor_class=TeatimeFeatureExtractor),
        n_steps=50,
        gamma=0.95,
        learning_rate=1e-4,
        ent_coef=0.1,
        batch_size=25,
        # clip_range=0.05,
        verbose=1
    )
)

encoder_model_config = AttrDict(
    model_class=Image2ActionModelLarge,
    image_key='tiago_head_image',
    privileged_key='privileged_info',
    image_shape=(3, 224, 224),
    discrete_privileged=discrete_privileged,
    privileged_classes=2,
    privileged_dim=privileged_dim,
    n_ensemble=5,
    bc_ckpt=bc_ckpt,
)

# This is actually env steps
train_config = AttrDict(
    n_epochs=50,
    rl_save_freq=220,
    warmup_rl_steps=3.2e2,
    encoder_warmup_epochs=10,

    rl_steps_per_epoch=5e2,
    encoder_epochs_per_epoch=5,
    encoder_batch_size=32,
)

sb3_callback_config = None #AttrDict(
#     reward_threshold=3,
#     eval_freq=int(3e2),
#     n_eval_envs=environment_config.num_envs,
# )

evaluator_config = None

config = AttrDict(
    env_config=environment_config,
    rl_model_config=rl_model_config,
    encoder_model_config=encoder_model_config,
    train_config=train_config,
    evaluator_config=evaluator_config,
    sb3_callback_config=sb3_callback_config,
)