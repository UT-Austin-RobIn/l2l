from l2l.imitation.utils.general_utils import AttrDict
from l2l.imitation.algo.bc_transformer import BCTransformer
from l2l.imitation.models.distribution_nets import MDN, Gaussian
from l2l.imitation.models.image_nets import ResNet18, ResNet34, SpatialSoftmax, R3M, VisionTransformer
from l2l.imitation.models.obs_nets import VisionCore, LowDimCore
from l2l.imitation.utils.dataset import SequenceDatasetMultiFile
from l2l.imitation.models.data_augmentation import BatchWiseImgColorJitterAug, TranslationAug
import torch.nn as nn
from collections import OrderedDict

train_config = AttrDict(
    output_dir="/home/shivin/learning2look/experiments/",
    batch_size=16,
    num_epochs=501,
    epoch_every_n_steps=500,
    log_every_n_epochs=1,
    val_every_n_epochs=5,
    save_every_n_epochs=10,
    eval_every_n_epochs=-1,
    seed=1
)

data_config = AttrDict(
    # data=["/home/shivin/learning2look/data/pick_dataset/pick_cup_n40.h5", "/home/shivin/learning2look/data/pick_dataset/pick_can_n40.h5"],
    # data=["/home/shivin/learning2look/data/apple_dataset/red_apple_n30.h5", "/home/shivin/learning2look/data/apple_dataset/green_apple_n30.h5"],
    data=["/home/shivin/learning2look/data/place_dataset/place_pink_n25.h5", "/home/shivin/learning2look/data/place_dataset/place_green_n25.h5"],
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
            "right",
            "base_delta_pose",
            # "base_velocity",
            "privileged_info",
        ],
        rgb = [
            "tiago_head_image",
        ],
        depth = [
            "tiago_head_depth",
        ],
    ),
    obs_keys_to_normalize = [
        "right",
        "base_delta_pose",
        # "base_velocity",
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
            augmentation=[
                AttrDict(
                    aug_class=TranslationAug,
                    aug_kwargs=dict(
                        translation=15
                    )
                ),
                AttrDict(
                    aug_class=BatchWiseImgColorJitterAug,
                    aug_kwargs=dict(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.3,
                        epsilon=0.1
                    )
                )
            ]
                    
        ),
        depth = AttrDict(
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
            augmentation=[
                AttrDict(
                    aug_class=TranslationAug,
                    aug_kwargs=dict(
                        translation=15
                    )
                )
            ]
        ),
    )
)

config = AttrDict(
    train_config=train_config,
    data_config=data_config,
    observation_config=observation_config,
    policy_config=policy_config,
    evaluator_config=None,
)
