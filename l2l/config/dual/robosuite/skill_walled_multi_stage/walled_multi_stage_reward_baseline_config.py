from l2l.utils.general_utils import AttrDict
from l2l.config.env.robosuite.skill_walled_multi_stage import env_config
from stable_baselines3.common.monitor import Monitor
from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper, FrameStackWrapper, RewardBasedLookingWrapper, CameraExplorationBonus
from stable_baselines3 import PPO
from l2l.modules.robosuite_feature_extractor import WalledMultiStageFeatureExtractor
from l2l.modules.image_model import Image2ActionModel

environment_config = AttrDict(
    env_config=env_config,
    # wrappers=[RobosuiteGymWrapper, Monitor],
    wrappers=[RobosuiteGymWrapper, RewardBasedLookingWrapper, Monitor, CameraExplorationBonus],
    num_envs=16,#24
)

if env_config.pi_format == "clip":
    privileged_dim = 512
    discrete_privileged = False
    bc_ckpt = "" # put path after training
elif env_config.pi_format == "onehot":
    privileged_dim = None
    discrete_privileged = True
    bc_ckpt = "/home/shivin/Desktop/learning2look/experiments_imitation/walled/weights/weights_ep15.pth"
else:
    raise NotImplementedError

rl_model_config = AttrDict(
    model_class=PPO,
    model_kwargs=AttrDict(
        policy="MultiInputPolicy",
        policy_kwargs=dict(
            features_extractor_class=WalledMultiStageFeatureExtractor,
            # features_extractor_kwargs=dict(features_dim=128),
        ),
        n_steps=320,
        gamma=0.99,
        learning_rate=1e-4,
        ent_coef=0.2, # 0.03
        batch_size=64,
        # clip_range=0.05,
        verbose=1
    )
)

encoder_model_config = AttrDict(
    model_class=Image2ActionModel,
    image_key='activeview_image',
    privileged_key='privileged_info',
    image_shape=(3, 84, 84),
    discrete_privileged=discrete_privileged,
    privileged_classes=4,
    privileged_dim=privileged_dim, # 512, # testing naive continuous privileged info
    n_ensemble=5,
    bc_ckpt=bc_ckpt,
    gmm_head_for_pi=True,
)

# This is actually env steps
train_config = AttrDict(
    n_epochs=50,
    rl_save_freq=625,
    warmup_rl_steps=3.2e2,
    encoder_warmup_epochs=10,
    
    rl_steps_per_epoch=2e4,
    encoder_epochs_per_epoch=7,
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