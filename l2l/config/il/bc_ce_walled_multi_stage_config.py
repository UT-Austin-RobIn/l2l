from l2l.imitation.utils.general_utils import AttrDict
from l2l.imitation.algo.bc_ce import BC_CrossEntropy
from l2l.imitation.models.distribution_nets import MDN, Gaussian
from l2l.imitation.models.image_nets import ResNet18, ResNet34, SpatialSoftmax, R3M, VisionTransformer
from l2l.imitation.models.obs_nets import VisionCore, LowDimCore
from l2l.imitation.utils.dataset import SequenceDatasetMultiFile
import torch.nn as nn
from collections import OrderedDict

train_config = AttrDict(
    output_dir="/home/shivin/Desktop/learning2look/experiments_imitation/",
    batch_size=32,
    num_epochs=50,
    epoch_every_n_steps=50,
    log_every_n_epochs=1,
    val_every_n_epochs=5,
    save_every_n_epochs=5,
    eval_every_n_epochs=5,
    seed=1
)

data_config = AttrDict(
    data=["/home/shivin/Desktop/learning2look/data/skill_multi_stage_oh_n200.h5"],
    dataset_class=SequenceDatasetMultiFile,
    dataset_kwargs=dict(
        seq_length=1,
        pad_seq_length=True,
        frame_stack=1,
        pad_frame_stack=True,
        dataset_keys=['actions'],
        hdf5_cache_mode='all',
        hdf5_use_swmr=True
    ),
    num_workers=2
)

# BC Catagorical
policy_config = AttrDict(
    policy_class=BC_CrossEntropy,
    output_dim=512,
    hidden_units=[512, 512],
    action_categories=6, # need to provide from domain knowledge
)

observation_config = AttrDict(
    obs = OrderedDict(
        low_dim = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "privileged_info",
            # "object-state"
        ],
        rgb = [
            # "agentview_image",
            # "sideview_image",
            "robot0_eye_in_hand_image",
            # "activeview_image",
            # "reference_image",
        ],
        depth = [],
    ),
    obs_keys_to_normalize = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        # "object-state",
        # "agentview_image",
        # "robot0_eye_in_hand_image",
    ],
    encoder = AttrDict(
        low_dim = AttrDict(
            core_class=None,#LowDimCore,
            core_kwargs=dict(
                output_dim=16,
                hidden_units=[16],
                activation=nn.LeakyReLU(0.2),
                output_activation=nn.LeakyReLU(0.2),
            ),
        ),
        rgb = AttrDict(
            core_class=VisionCore,
            core_kwargs=dict(
                backbone_class=ResNet18,
                backbone_kwargs=None,
                feature_dim=64,
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
from l2l.config.env.robosuite.skill_walled_multi_stage import env_config
evaluator_config = AttrDict(
    evaluator=RobosuiteEvaluator,
    env_config = env_config,
    n_rollouts = 10,
    max_steps = 10,
)

config = AttrDict(
    train_config=train_config,
    data_config=data_config,
    observation_config=observation_config,
    policy_config=policy_config,
    evaluator_config=evaluator_config,
)