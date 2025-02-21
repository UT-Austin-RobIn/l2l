import numpy as np
import copy
# from l2l.imitation.algo.bc_ce import BC_CrossEntropy
# from l2l.imitation.algo.bc_rnn import BC_RNN
# from l2l.imitation.algo.diffusion_policy import DiffusionPolicy
# from l2l.imitation.algo.bc_transformer import BCTransformer
import torch
import os
import cv2
import time
from l2l.imitation.utils.vis_utils import get_saliency_maps, write_video

from l2l.modules.image_model import Image2ActionModel, TransformerImage2ActionModel
from stable_baselines3 import PPO

from importlib.machinery import SourceFileLoader
from l2l.utils.general_utils import get_env_from_config

import matplotlib.pyplot as plt

class RandomWalkBaseline:

    def __init__(self, n_actions):
        self.n_actions = n_actions
        
        self.current_action = np.random.randint(0, n_actions)

    def predict(self, obs, **kwargs):
        if np.random.rand() < 0.4:
            self.current_action = np.random.randint(0, self.n_actions)

        return self.current_action, None

def get_action_and_entropy(obs, active_image_encoder):
    action, is_uncertain, uncertainty = active_image_encoder.get_action_and_uncertainty(obs)

    return action, is_uncertain, uncertainty

def get_encoder_path(rl_path):
    rl_ckpt = rl_path.split('/')[-3].split('_')[-1]

    encoder_path = "/".join(rl_path.split('/')[:-3]) + f"/encoder_weights/weights/weights_ep{rl_ckpt}.pth"
    return encoder_path

def get_action_and_entropy(obs, active_image_encoder):
    action, is_uncertain, uncertainty = active_image_encoder.get_action_and_uncertainty(obs)

    return action, is_uncertain, uncertainty


noise = 0.00
def do_single_rollout(
                    env, 
                    cam_policy, 
                    active_image_encoder, 
                    render, 
                    deterministic, 
                    total_cam_steps,
                    total_cam_steps_per_run,
                    n_stages,
                    info_gathering_steps_break):
    obs = env.reset()
    for k in obs:
        obs[k] = obs[k] + np.random.normal(scale=noise, size=obs[k].shape) if noise > 0.00 else obs[k]
    
    max_steps = 20 # 100 for visuomotor
    cam_steps = 0
    info = {'cam_step_and_uncertainty': [[], []]}
    stage_success = [False for _ in range(n_stages)]
    cur_stage = env.stage_id
    for i in range(max_steps):
        action, is_uncertain, uncertainty = get_action_and_entropy(obs, active_image_encoder)
        # loss = active_image_encoder.get_reward(obs)
        # print(uncertainty, loss)
        while len(info['cam_step_and_uncertainty'][0]) < len(env.recorded_obs):
            info['cam_step_and_uncertainty'][0].append(uncertainty)
            info['cam_step_and_uncertainty'][1].append(0)

        gather_info = False
        
        if is_uncertain:
            gather_info = True
            info_gathering_steps = 0
            cam_steps_per_run = 0
        
        while gather_info:
            _obs = copy.deepcopy(obs)
            _obs['is_policy_uncertain'] = np.array([int(is_uncertain)])
            cam_act, states = cam_policy.predict(_obs, deterministic=deterministic)
            
            obs, _, _, _ = env.rotate_camera(cam_act)
            for k in obs:
                obs[k] = obs[k] + np.random.normal(scale=noise, size=obs[k].shape) if noise > 0.00 else obs[k]
            
            if render:
                img = obs['activeview_image']/255
                img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
                cv2.imshow('rollout', img)
                cv2.waitKey(100)

            action, is_uncertain, uncertainty = get_action_and_entropy(obs, active_image_encoder)
            # loss = active_image_encoder.get_reward(obs)
            # print('\n', cam_act)
            # print(uncertainty, loss)

            info['cam_step_and_uncertainty'][0].append(uncertainty)
            info['cam_step_and_uncertainty'][1].append(1) 
            
            if not is_uncertain:
                info_gathering_steps += 1
            else:
                info_gathering_steps = 0
            
            if info_gathering_steps >= info_gathering_steps_break:
                gather_info = False 

            cam_steps += 1
            cam_steps_per_run += 1
            if cam_steps_per_run > total_cam_steps_per_run:
                break
            if cam_steps > total_cam_steps:
                break
                
        if render:
            img = obs['activeview_image']/255
            img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

            for _ in range(5):
                cv2.imshow('rollout', img)
                cv2.waitKey(5)
            env.render()

        obs, _, done, _ = env.step(action)
        for k in obs:
            obs[k] = obs[k] + np.random.normal(scale=noise, size=obs[k].shape) if noise > 0.00 else obs[k]
        if env.stage_id > cur_stage:
            print(f'stage {i} cam_steps', cam_steps)
            stage_success[cur_stage] = True
            cur_stage = env.stage_id
            # print("Stage success: ", cur_stage)

        if env._check_success():
            print('cam_steps', cam_steps)
            print('env steps', env.total_env_steps_taken)
            stage_success[-1] = True
            return stage_success, info
        
        if env._check_failure():
            print('cam_steps', cam_steps)
            print('env steps', env.total_env_steps_taken)
            return stage_success, info
    
    print('cam_steps', cam_steps)
    print('env steps', env.total_env_steps_taken)        
    return stage_success, info

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--n_rollouts', type=int, default=50)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--info_step_break', type=int)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    
    print("Deterministic: ", args.deterministic)
    print("Eval with noise: ", noise)
    # if args.record:
    save_folder = f"../experiments/recordings/{args.env}/fails"
    os.makedirs(save_folder, exist_ok=True)

    import robosuite as suite
    if args.env == 'kitchen':
        from l2l.config.env.robosuite.skill_kitchen import env_config
        total_cam_steps_per_run = 80
        total_cam_steps = 240
        n_stages = 3
        n_actions = 5
    elif args.env == 'walled':
        from l2l.config.env.robosuite.skill_walled_multi_stage import env_config
        total_cam_steps_per_run = 80
        total_cam_steps = 160
        n_stages = 2
        n_actions = 9
    elif args.env == 'two_arm':
        from l2l.config.env.robosuite.skill_two_arm import env_config
        total_cam_steps_per_run = 50
        total_cam_steps = 50
        n_stages = 1
        n_actions = 9
    elif args.env == 'visuomotor_walled':
        from l2l.config.env.robosuite.walled_multi_stage import env_config
        total_cam_steps_per_run = 80
        total_cam_steps = 160
        n_stages = 2
        n_actions = 9
    else:
        raise NotImplementedError
    
    # initialize the task
    env = suite.make(
        **env_config
    )

    # load all models
    encoder_path = get_encoder_path(args.ckpt)
    # cam_policy = RandomWalkBaseline(n_actions=n_actions)
    cam_policy = PPO.load(args.ckpt, device='cuda')
    # active_image_encoder = TransformerImage2ActionModel.load_weights(encoder_path).to('cuda') # for visuomotor eval
    active_image_encoder = Image2ActionModel.load_weights(encoder_path).to('cuda')

    total_stage_attempts = [0 for _ in range(n_stages)]
    total_stage_success = [0 for _ in range(n_stages)]
    for i in range(args.n_rollouts):
        stage_success, info = do_single_rollout(
                env=env,
                cam_policy=cam_policy,
                active_image_encoder=active_image_encoder,
                render=args.render,
                deterministic=args.deterministic,
                total_cam_steps=total_cam_steps,
                total_cam_steps_per_run=total_cam_steps_per_run,
                n_stages=n_stages,
                info_gathering_steps_break=args.info_step_break
            )

        for j, success in enumerate(stage_success):
            if j == 0 or stage_success[j-1]:
                total_stage_attempts[j] += 1
            if success:
                total_stage_success[j] += 1
        print(f"Rollout {i}: {stage_success}")
        print()

        if not stage_success[-1]: #args.record

            recording = []
            info['activeview_image'] = []
            info['agentview_image'] = []
            info['robot0_eye_in_hand_image'] = []

            save_name = f'{args.env}_{time.asctime().replace(" ", "_")}' 
            save_recording =  os.path.join(save_folder, f'{save_name}.mp4')
            save_np = os.path.join(save_folder, f'{save_name}.npy')
            print("Saving video to", save_recording)
            for ob in env.recorded_obs:
                img = np.concatenate((ob['activeview_image'], ob['agentview_image'], ob['robot0_eye_in_hand_image']), axis=1)/255
                
                recording.append(img)
                info['activeview_image'].append(ob['activeview_image']/255)
                info['agentview_image'].append(ob['agentview_image']/255)
                info['robot0_eye_in_hand_image'].append(ob['robot0_eye_in_hand_image']/255)
                
                # cv2.imshow('rollout', img)
                # cv2.waitKey(10)

            # plt.plot(info['cam_step_and_uncertainty'][0])
            # plt.show()
            write_video((255*np.array(recording)).astype(np.uint8), save_recording, fps=10)
            np.save(save_np, info)

    for i in range(n_stages):
        print(f"Stage {i}: {total_stage_success[i]}/{total_stage_attempts[i]}")