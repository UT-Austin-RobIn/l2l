from l2l.imitation.utils.general_utils import AttrDict
from l2l.imitation.algo.bc_transformer import BCTransformer
from l2l.imitation.models.distribution_nets import MDN, Gaussian
from l2l.imitation.models.image_nets import ResNet18, ResNet34, SpatialSoftmax, R3M, VisionTransformer
from l2l.imitation.models.obs_nets import VisionCore, LowDimCore
from l2l.imitation.utils.dataset import SequenceDatasetMultiFile
import torch.nn as nn
from collections import OrderedDict

train_config = AttrDict(
    output_dir="/home/shivin/imitation/experiments/",
    batch_size=16,
    num_epochs=1000,
    epoch_every_n_steps=500,
    log_every_n_epochs=1,
    val_every_n_epochs=10,
    save_every_n_epochs=25,
    eval_every_n_epochs=25,
    seed=1
)

data_config = AttrDict(
    data=["/home/shivin/imitation/data/multi_stage_n300.h5"],
    dataset_class=SequenceDatasetMultiFile,
    dataset_kwargs=dict(
        seq_length=10,
        pad_seq_length=True,
        frame_stack=1,
        pad_frame_stack=True,
        dataset_keys=['actions'],
        hdf5_cache_mode='low_dim',
        hdf5_use_swmr=True
    ),
    num_workers=2
)

# BC-Transformer GMM
policy_config = AttrDict(
    policy_class=BCTransformer,
    token_dim=64,
    transformer_num_layers=4,
    transformer_num_heads=6,
    transformer_head_output_size=64,
    transformer_mlp_hidden_size=256,
    transformer_dropout=0.1,
    horizon=10,
 
    action_head=MDN,
    action_head_kwargs=dict(
        num_gaussians=5
    )
)

observation_config = AttrDict(
    obs = OrderedDict(
        low_dim = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "privileged_info",
        ],
        rgb = [
            "activeview_image",
            "robot0_eye_in_hand_image",
        ],
        depth = [],
    ),
    obs_keys_to_normalize = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ],
    encoder = AttrDict(
        low_dim = AttrDict(
            core_class=LowDimCore,
            core_kwargs=dict(
                feature_dim=policy_config.token_dim,
                hidden_units=[],
                activation=nn.LeakyReLU(0.2),
                output_activation=None,
            ),
        ),
        rgb = AttrDict(
            core_class=VisionCore,
            core_kwargs=dict(
                backbone_class=ResNet18,
                backbone_kwargs=None,
                feature_dim=policy_config.token_dim,
                pool_class=SpatialSoftmax,
                pool_kwargs=dict(
                    num_kp=32,
                    learnable_temperature=False,
                    temperature=1.0,
                    noise_std=0.0,
                ),
            ),
        ),
        depth = AttrDict(
            core_class=None,
            core_kwargs=dict(),
        ),
    )
)

from l2l.imitation.evaluators.robosuite_evaluator import RobosuiteEvaluator

# from l2l.config.env.robosuite.color_lift import env_config
# evaluator_config = AttrDict(
#     evaluator=RobosuiteEvaluator,
#     env_config = env_config,
#     n_rollouts = 10,
#     max_steps = 60,
# )

# from l2l.config.env.robosuite.camera_color_pnp_imitation import env_config
# evaluator_config = AttrDict(
#     evaluator=RobosuiteEvaluator,
#     env_config = env_config,
#     n_rollouts = 10,
#     max_steps = 200,
# )

from l2l.config.env.robosuite.walled_multi_stage import env_config
evaluator_config = AttrDict(
    evaluator=RobosuiteEvaluator,
    env_config = env_config,
    n_rollouts = 10,
    max_steps = 100,
)


config = AttrDict(
    train_config=train_config,
    data_config=data_config,
    observation_config=observation_config,
    policy_config=policy_config,
    evaluator_config=evaluator_config,
)
