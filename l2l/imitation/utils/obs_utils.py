import numpy as np
import torch

def get_processed_shape(obs_modality, input_shape):

    return list(process_obs(obs=np.zeros(input_shape), modality=obs_modality).shape)

def process_obs(obs, modality):
    if modality == 'low_dim':
        return obs
    elif modality == 'rgb':
        return batch_image_hwc_to_chw(obs)/255.0
    elif modality == 'depth':
        assert obs.max() <= 1, "depth is not normalized"
        return batch_image_hwc_to_chw(obs)
    else:
        raise ValueError(f"Modality {modality} not supported")
    
def process_obs_dict(obs_dict, obs_to_modality):
    processed_obs_dict = {}
    for k, obs in obs_dict.items():
        if k not in obs_to_modality:
            continue
        processed_obs_dict[k] = process_obs(obs, obs_to_modality[k])

    return processed_obs_dict

def batch_image_hwc_to_chw(im):
    start_dims = np.arange(len(im.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    if isinstance(im, np.ndarray):
        return im.transpose(start_dims + [s + 3, s + 1, s + 2])
    else:
        return im.permute(start_dims + [s + 3, s + 1, s + 2])