import numpy as np
import copy
import rospy
import torch
import cv2
from imitation.utils.vis_utils import get_saliency_maps, write_video

from l2l.modules.image_model import Image2ActionModelLarge, TransformerImage2ActionModelLarge
from stable_baselines3 import PPO

from importlib.machinery import SourceFileLoader
from l2l.utils.general_utils import get_env_from_config, AttrDict
import matplotlib.pyplot as plt

from telemoma.human_interface.teleop_policy import TeleopPolicy
from telemoma.configs.only_vr import teleop_config

class RandomWalkBaseline:

    def __init__(self, n_actions):
        self.n_actions = n_actions
        
        self.current_action = np.random.randint(0, n_actions)

    def predict(self, obs, **kwargs):
        if np.random.rand() < 0.5:
            self.current_action = np.random.randint(0, self.n_actions)

        return np.array(self.current_action), None

def get_encoder_path(rl_path):
    rl_ckpt = rl_path.split('/')[-3].split('_')[-1]

    encoder_path = "/".join(rl_path.split('/')[:-3]) + f"/encoder_weights/weights/weights_ep{rl_ckpt}.pth"
    return encoder_path

def copy_np_dict(d):
    return {k: v.copy() for k, v in d.items()}


def process_obs_for_cam(obs_look, is_uncertain):
    _obs_look = copy_np_dict(obs_look)
    keys_to_remove = []
    for key in _obs_look:
        if key not in env.observation_space.keys():
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del _obs_look[key]
    _obs_look['is_policy_uncertain'] = np.array([int(is_uncertain)])
    return _obs_look

if __name__ == "__main__":
    # task = 'clock'
    # task = 'person'
    task = 'button'

    from tiago_gym.tiago.tiago_gym import TiagoGym
    from l2l.wrappers.real_wrappers import DisplayImageWrapper, DisplayStringWrapper, RealFullSystemWrapper, TiagoHandAndHeadWrapper, \
        OnOffDisplayImageWrapper, RealFullSystemHandWrapper
    env = TiagoGym(
                frequency=10,
                head_enabled=True,
                base_enabled=True,
                right_arm_enabled=True,
                left_arm_enabled=task=='button',
                right_gripper_type='robotiq2F-140',
                left_gripper_type='robotiq2F-85',)
    if task == 'clock':
        env = DisplayStringWrapper(RealFullSystemWrapper(env))
        rl_ckpt = '/home/shivin/learning2look/experiments/final_real_clock_1/epoch_20/weights/rl_model_11000_steps.zip'
        bc_ckpt = '/home/shivin/learning2look/experiments/real_pick_cup_or_can_n80_with_augmentation_2/weights/weights_ep250.pth'
    elif task == 'person':     
        env = DisplayImageWrapper(RealFullSystemWrapper(env))
        rl_ckpt = '/home/shivin/learning2look/experiments/final_real_person_1_resumed_8/epoch_7/weights/rl_model_4290_steps.zip'
        bc_ckpt = '/home/shivin/learning2look/experiments/real_place_pink_or_green_n50_with_augmentation_1/weights/weights_ep200.pth'
    elif task == 'button':
        env = OnOffDisplayImageWrapper(RealFullSystemHandWrapper(TiagoHandAndHeadWrapper(env)))
        rl_ckpt = '/home/shivin/learning2look/experiments/final_real_button_1/epoch_7/weights/rl_model_4152_steps.zip'
        bc_ckpt = '/home/shivin/learning2look/experiments/real_apple_red_or_green_n60_with_augmentation_1/weights/weights_ep200.pth'
    
    encoder_path = get_encoder_path(rl_ckpt)
    # cam_policy = RandomWalkBaseline(8)
    cam_policy = PPO.load(rl_ckpt, env=None)
    active_image_encoder = TransformerImage2ActionModelLarge(
        config=AttrDict(
            image_key='tiago_head_image',
            privileged_key='privileged_info',
            image_shape=(3, 224, 224),
            discrete_privileged=True,
            privileged_classes=2,
            privileged_dim=None,
            n_ensemble=5,
            bc_ckpt=bc_ckpt
        )
    )
    trained_encoder = Image2ActionModelLarge.load_weights(encoder_path)
    active_image_encoder.nets['obs_encoder'] = trained_encoder.nets['obs_encoder']

    teleop = TeleopPolicy(teleop_config)
    teleop.start()

    def shutdown_helper():
        teleop.stop()
    rospy.on_shutdown(shutdown_helper)

    obs_act, _ = env.reset()
    obs_look = copy_np_dict(obs_act)
    # input("Press enter to start...")

    # do visualization
    cam_steps = 0
    is_uncertain = True

    video = []

    uncertainty_list = []
    cam_uncertainty_list = []
    uncertainty_and_cam_idx = [[], []]

    max_steps=20
    first = True
    while not rospy.is_shutdown():
        
        if True or is_uncertain:
            _, is_uncertain, uncertainty = active_image_encoder.get_action_and_uncertainty_for_real(obs_act=copy_np_dict(obs_act), obs_look=obs_look)
            uncertainty_list.append(uncertainty)
        uncertainty_and_cam_idx[0].append(uncertainty)
        uncertainty_and_cam_idx[1].append(0)

        img = obs_act['tiago_head_image']/255
        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
        video.append(img)
        cv2.imshow('rollout', img)
        cv2.waitKey(1)

        gather_info = False        
        if is_uncertain:
            gather_info = True
            info_gathering_steps = 0
            cam_steps = 0
        
        while first and gather_info and (not rospy.is_shutdown()):
            cam_uncertainty_list.append(uncertainty)

            _obs_look = process_obs_for_cam(obs_look=obs_look, is_uncertain=is_uncertain)
            cam_act, states = cam_policy.predict(_obs_look, deterministic=False)
            print(cam_act)

            obs_look, _, _, _, _ = env.rotate_camera(cam_act)
            _, is_uncertain, uncertainty = active_image_encoder.get_action_and_uncertainty_for_real(obs_act=copy_np_dict(obs_act), obs_look=obs_look)
            
            img = obs_look['tiago_head_image']/255
            img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
            for i in range(5):
                video.append(img)
                uncertainty_and_cam_idx[0].append(uncertainty)
                uncertainty_and_cam_idx[1].append(1)
            cv2.imshow('rollout', img)
            cv2.waitKey(1)
            
            
            print(uncertainty)
            if not is_uncertain:
                info_gathering_steps += 1
            else:
                info_gathering_steps = 0
            
            if info_gathering_steps >= 3:
                # reset head position
                obs_act = env.reset_head()

                # reset policy
                active_image_encoder.policy.reset()
                
                gather_info = False
                break

            cam_steps += 1
            if cam_steps > max_steps:
                env.reset_head()
                active_image_encoder.policy.reset()
                break
        first=False
        
        # it's important that this comes first because of some bug in data collection
        final_action = teleop.get_action(obs_act)
        buttons = final_action.extra['buttons']
        
        if not (buttons.get('RG', False) or buttons.get('LG', False)):
            action = active_image_encoder.get_fast_actions(obs_act=copy_np_dict(obs_act))
            # print('real_priv', obs_act['privileged_info'])
            # print()
            final_action['right'] = action[3:10]
            final_action['base'] = action[:3]
        
        obs_act, _, done, _, _ = env.step(final_action)
        
        if buttons.get('A', False) or buttons.get('B', False):
            break

    teleop.stop()
    
    # create two axes
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(uncertainty_and_cam_idx[0])
    ax[1].plot(uncertainty_and_cam_idx[1])
    # plt.show()
    
    import os
    import time
    save_path = f"../experiments/recordings/{task}/{task}_{time.asctime().replace(' ', '_')}"
    save_video =  save_path + ".mp4"
    save_np = save_path + ".npy"
    
    # write_video(np.array(video).astype(np.float32), save_video, fps=10)
    # np.save(save_np, np.array(uncertainty_and_cam_idx))