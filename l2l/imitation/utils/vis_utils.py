import cv2
import torch
import numpy as np

def get_saliency_maps(model, obs):
    action = model.get_action(obs)
    keys_to_modality = model.config.keys_to_modality
    image_keys = []
    for k in keys_to_modality:
        if keys_to_modality[k] == 'rgb':
            image_keys.append(k)

    original_images = {k: obs[k] for k in image_keys}

    obs = model.preprocess_obs(obs)
    
    # comment out following when using BCTransformer
    # for k in obs:
    #     obs[k] = obs[k][:, None]

    model.zero_grad()
    model.eval()

    # generate saliency map
    for k in image_keys:
        obs[k] = obs[k].clone().detach().requires_grad_(True)
    
    # Forward pass
    loss = model.compute_loss({'obs': obs, 'actions': torch.tensor(action).view(1, 1, -1)}).total

    # Backward pass
    loss.backward()

    # Get saliency map
    saliency_dict = {}
    for k in image_keys:
        saliency = obs[k].grad.data.abs()[0, 0].cpu().numpy()

        # Normalize the saliency map for visualization
        min_s = np.min(saliency) #np.min(saliency, axis=(1, 2)).reshape(3, 1, 1)
        max_s = np.max(saliency) #np.max(saliency, axis=(1, 2)).reshape(3, 1, 1)
        saliency = (saliency - min_s) / (max_s - min_s + 1e-12)
        saliency = np.uint8(255 * saliency).transpose(1, 2, 0)  # Scale up to 255

        saliency = np.array(0.5 * saliency + 0.5 * original_images[k].astype(np.uint8)).astype(np.uint8)

        saliency_dict[k] = saliency

    return saliency_dict

def write_video(video_frames, filename, fps=10):
    '''
    video_frames: list of frames (T, C, H, W)
    '''

    import imageio
    for i in range(len(video_frames)):
        video_frames[i] = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
    imageio.mimwrite(filename, video_frames, fps=fps)

