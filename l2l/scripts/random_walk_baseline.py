import numpy as np
import copy
from imitation.algo.bc_ce import BC_CrossEntropy
from imitation.algo.bc_rnn import BC_RNN
from imitation.algo.diffusion_policy import DiffusionPolicy
from imitation.algo.bc_transformer import BCTransformer
import torch
import cv2
from imitation.utils.vis_utils import get_saliency_maps, write_video

from l2l.modules.image_model import Image2ActionModel
from stable_baselines3 import PPO

from importlib.machinery import SourceFileLoader
from l2l.utils.general_utils import get_env_from_config

import matplotlib.pyplot as plt

class RandomWalkBaseline:

    def __init__(self, n_actions):
        self.n_actions = n_actions
        
        self.current_action = np.random.randint(0, n_actions)

    def get_action(self, obs):
        if np.random.rand() < 0.4:
            self.current_action = np.random.randint(0, self.n_actions)

        return self.current_action

def get_action_and_entropy(obs, active_image_encoder):
    action, is_uncertain, uncertainty = active_image_encoder.get_action_and_uncertainty(obs)

    return action, is_uncertain, uncertainty

def get_encoder_path(rl_path):
    rl_ckpt = rl_path.split('/')[-3].split('_')[-1]

    encoder_path = "/".join(rl_path.split('/')[:-3]) + f"/encoder_weights/weights/weights_ep{rl_ckpt}.pth"
    return encoder_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_ckpt', type=str, default='/home/shivin/Desktop/learning2look/experiments/multi_stageskill_dual_cam_with_movement_1/epoch_43/weights/rl_model_905760_steps.zip')
    # parser.add_argument('--rl_ckpt', type=str, default='/home/shivin/Desktop/learning2look/experiments/kitchen_skill_dual_2dof_1/epoch_10/weights/rl_model_245980_steps.zip')
    args = parser.parse_args()
    
    # load policy
    config_path = "l2l/config/dual/robosuite/skill_multi_stage/multi_stage_action_dual_config.py"
    # config_path = "l2l/config/dual/robosuite/skill_kitchen/kitchen_action_dual_config.py"
    conf = SourceFileLoader('conf', config_path).load_module().config
    rl_env = get_env_from_config(conf.env_config)

    import robosuite as suite
    from l2l.config.env.robosuite.skill_multi_stage import env_config
    # from l2l.config.env.robosuite.skill_kitchen import env_config
    # initialize the task
    env = suite.make(
        **env_config
    )
    obs = env.reset()

    
    encoder_path = get_encoder_path(args.rl_ckpt)
    cam_policy = RandomWalkBaseline(env.n_camera_actions)
    active_image_encoder = Image2ActionModel.load_weights(encoder_path)

    # do visualization
    steps = 0
    cam_steps = 0

    video = []
    n_evals = 0
    n_success = 0

    uncertainty_list = []
    cam_uncertainty_list = []
    reward_list = []

    for i in range(100):
        action, is_uncertain, uncertainty = get_action_and_entropy(obs, active_image_encoder)
        print(uncertainty)
        uncertainty_list.append(uncertainty)

        gather_info = False
        
        if is_uncertain:
            gather_info = True
            info_gathering_steps = 0
            cam_steps = 0
        
        while gather_info:
            cam_uncertainty_list.append(uncertainty)
            reward_list.append(active_image_encoder.get_reward(obs))

            cam_act = cam_policy.get_action(obs)
            
            img = np.concatenate([obs['agentview_image']/255, obs['activeview_image']/255], axis=1)
            img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
            video.append(img)
            cv2.imshow('rollout', img)
            cv2.waitKey(10)
            
            obs, _, _, _ = env.rotate_camera(cam_act)

            action, is_uncertain, uncertainty = get_action_and_entropy(obs, active_image_encoder)
            print(uncertainty) 
            
            if not is_uncertain:
                info_gathering_steps += 1
            else:
                info_gathering_steps = 0
            
            if info_gathering_steps >= 1:
                input("Press Enter to continue...")
                gather_info = False 

            cam_steps += 1
            if cam_steps > 100:
                break
              

        img = np.concatenate([obs['agentview_image']/255, obs['activeview_image']/255], axis=1)
        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
        for _ in range(5):
            video.append(img)

        for _ in range(5):
            cv2.imshow('rollout', img)
            cv2.waitKey(5)
        print()

        obs, _, done, _ = env.step(action)

        steps += 1

        if done or steps%100==0:
            obs = env.reset()
            steps = 0

            # create two axes
            fig, ax = plt.subplots(3, 1)
            ax[0].plot(uncertainty_list)
            ax[1].plot(cam_uncertainty_list)
            ax[2].plot(reward_list)
            plt.show()
            uncertainty_list = []
            cam_uncertainty_list = []
            reward_list = []

            n_evals += 1
            if done:
                n_success += 1

            if n_evals > 5:
                break

    print(f"Success rate: {(n_success-1)/(n_evals-1)}")
    write_video(np.array(video).astype(np.float32), "full_system_action.mp4", fps=10)
