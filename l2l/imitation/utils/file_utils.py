import h5py
import numpy as np
import os
from collections import OrderedDict
from l2l.imitation.utils.obs_utils import get_processed_shape

def get_all_obs_keys_from_config(config):
    all_obs_keys = []
    for modality, obs_keys in config.obs.items():
        all_obs_keys.extend(obs_keys)
    return all_obs_keys

def get_obs_key_to_modality_from_config(config):
    obs_key_to_modality = {}
    for modality, obs_keys in config.obs.items():
        for obs_key in obs_keys:
            obs_key_to_modality[obs_key] = modality
    return obs_key_to_modality

def get_shape_metadata_from_dataset(dataset_path, all_obs_keys, obs_key_to_modality):
    shape_meta = {}

    # read demo file for some metadata
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    demo_id = list(f["data"].keys())[0]
    demo = f[f"data/{demo_id}"]

    # action dimension
    shape_meta['ac_dim'] = f[f"data/{demo_id}/actions"].shape[1]

    obs_shape_meta = {}
    for k in sorted(all_obs_keys):
        obs = demo["obs/{}".format(k)][:]
        if obs_key_to_modality[k] == "low_dim" and len(obs.shape) == 1:
            obs = obs.reshape(-1, 1)
        obs_shape_meta[k] = get_processed_shape(obs_key_to_modality[k], obs.shape[1:])

    shape_meta['obs_shape'] = obs_shape_meta
    f.close()

    return shape_meta