from l2l.utils.general_utils import AttrDict
from l2l.config.env.real.tiago_head_and_hand_env import env_config
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from l2l.modules.real_feature_extractor import ButtonFeatureExtractor
from l2l.modules.image_model import Image2ActionModelLarge
from l2l.wrappers.real_wrappers import SimAndRealUncertaintyAwareWrapper, TiagoHandAndHeadWrapper, \
    OnOffDisplayImageWrapper, CameraAndHandExplorationBonus

environment_config = AttrDict(
    env_config=env_config,
    wrappers=[TiagoHandAndHeadWrapper, SimAndRealUncertaintyAwareWrapper, OnOffDisplayImageWrapper, 
              Monitor],
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
        policy_kwargs=dict(features_extractor_class=ButtonFeatureExtractor),
        n_steps=64,
        gamma=0.99,
        learning_rate=1e-4,
        ent_coef=0.1,
        batch_size=32,
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
    warmup_rl_steps=128,
    encoder_warmup_epochs=3,

    rl_steps_per_epoch=5e2,
    encoder_epochs_per_epoch=5,
    encoder_batch_size=32,

    load_dataset_file='/home/shivin/learning2look/experiments/real_button_test_5/dataset_3904.pkl',
    load_encoder_file='/home/shivin/learning2look/experiments/real_button_test_5/encoder_weights/weights/weights_ep6.pth'
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