import gym
from gymnasium import spaces
import numpy as np
import time
import cv2
from collections import OrderedDict
from tiago_gym.utils.general_utils import AttrDict, copy_np_dict

def merge_dicts(d1, d2):
    for k in d1.keys():
        assert k not in d2, f'found same key ({k}) in both dicts'
        d2[k] = d1[k]
    return d2

class RealFullSystemWrapper(gym.core.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)

        self.cam_frequency = 1

    def add_extra_obs(self, obs):
        obs['privileged_info'] = self.privileged_info.copy()
        for k in obs.keys():
            if 'image' in k or 'depth' in k:
                if 'depth' in k:
                    obs[k] = np.clip(obs[k], 0, 4000)/4000

                obs[k] = obs[k].astype(float)
                if len(obs[k].shape) < 3: # happens in depth
                    obs[k] = obs[k][..., None]
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.privileged_info = np.array([1, 0]) if np.random.random()>0.5 else np.array([0, 1])

        obs = self.add_extra_obs(obs)
        return obs, info
    
    def reset_head(self):
        self.env.tiago.head.reset(self.env.tiago.reset_pose)
        time.sleep(2)
        
        obs = self.env._observation()
        obs = self.add_extra_obs(obs)

        return obs

    def rotate_camera(self, action):
        self.cam_start_time = time.time()
        if action is not None:
            assert isinstance(action, np.ndarray), "Action in wrong format"
            self.env.tiago.step({'head': action})
        
        self.cam_end_time = time.time()
        time.sleep(max(0., 1/self.cam_frequency - (self.cam_end_time-self.cam_start_time)))

        obs = self.env._observation()
        self.env.steps += 1

        obs = self.add_extra_obs(obs)
        return obs, 0, False, False, {}
    
    def step(self, action):
        obs, rew, terminate, truncate, info = self.env.step(action)
        obs = self.add_extra_obs(obs)

        return obs, rew, terminate, truncate, info
    
class RealFullSystemHandWrapper(gym.core.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)

        self.cam_frequency = 1

    def add_extra_obs(self, obs):
        obs['privileged_info'] = self.privileged_info.copy()
        for k in obs.keys():
            if 'image' in k or 'depth' in k:
                if 'depth' in k:
                    obs[k] = np.clip(obs[k], 0, 4000)/4000

                obs[k] = obs[k].astype(float)
                if len(obs[k].shape) < 3: # happens in depth
                    obs[k] = obs[k][..., None]
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        r = np.random.random()
        self.privileged_info = np.array([1, 0]) if r>0.5 else np.array([0, 1])

        obs = self.add_extra_obs(obs)
        return obs, info
    
    def reset_head(self):
        self.env.tiago.head.reset(self.env.tiago.reset_pose)
        time.sleep(2)
        
        obs = self.env.unwrapped._observation()
        obs = self.add_extra_obs(obs)

        return obs

    def rotate_camera(self, action):
        self.cam_start_time = time.time()
        info = {}
        if action is not None:
            assert isinstance(action, np.ndarray), "Action in wrong format"
            info = self.env.step({'head': action})[-1]
        
        self.cam_end_time = time.time()
        time.sleep(max(0., 1/self.cam_frequency - (self.cam_end_time-self.cam_start_time)))

        obs = self.env.unwrapped._observation()
        obs['arm_x_position'] = np.array([self.env.net_movement[0]])
        self.env.steps += 1

        obs = self.add_extra_obs(obs)
        return obs, 0, False, False, info
    
    def step(self, action):
        obs, rew, terminate, truncate, info = self.env.env.step(action)
        obs = self.add_extra_obs(obs)

        return obs, rew, terminate, truncate, info

class SimAndRealRewardWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        from l2l.config.env.robosuite.skill_color_pnp import env_config
        from l2l.utils.general_utils import get_env_from_config
        from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper

        self.sim_env = get_env_from_config(AttrDict(env_config=env_config, wrappers=[RobosuiteGymWrapper]))
    
        self.encoder = None

        self.mode = 'robot'
        self.camera_step = 0
        self.max_camera_steps = 32 #20

        self.new_episode = True
        self.full_obs = None

    @property   
    def observation_space(self):
        ob_space = OrderedDict()
        sim_ob_space = self.sim_env.observation_space
        real_ob_space = self.env.observation_space

        ob_space['privileged_info'] = sim_ob_space['privileged_info']
        for k in real_ob_space:
            ob_space[k] = real_ob_space[k]
        ob_space['is_policy_uncertain'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        return spaces.Dict(ob_space)
        
    @property
    def action_space(self):
        if hasattr(self.env, 'n_looking_actions'):
            return spaces.Discrete(self.env.n_looking_actions)
        return spaces.Discrete(5)
    
    def set_encoder(self, encoder):
        self.encoder = encoder
    
    def update_dict(self, d):
        '''
            Overwrite keys of d1 with those of d2
        '''
        for k in d.keys():
            assert k in self.full_obs.keys(), f'Key {k} not found in dictionary d1'
            self.full_obs[k] = d[k]
    
    def hard_reset(self, **kwargs):
        obs_real, info_real = self.env.reset(**kwargs)
        obs_sim, info_sim = self.sim_env.reset(**kwargs)

        obs = merge_dicts(obs_real, obs_sim)
        info = merge_dicts(info_real, info_sim)

        return obs, info

    def reset(self, **kwargs):
        if self.new_episode:
            self.full_obs, info = self.hard_reset(**kwargs)
            self.camera_step = 0
        else:
            info = {}

        self.mode = 'robot'
        _, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(0, False, False, {})
        if terminated or truncated:
            self.full_obs, info = self.hard_reset(**kwargs)
            self.camera_step = 0
            _, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(0, False, False, {})
        
        self.mode = 'camera'        
        return copy_np_dict(self.full_obs), info

    def rollout_robot_until_uncertain(self, reward, terminated, truncated, info, loss=None):
        while self.mode == 'robot':
            if self.encoder is None:
                output = (len(self.sim_env.unwrapped.skills)-1, False)
                loss = 1
            else:
                output = self.encoder.get_action_and_uncertainty(self.full_obs)
                loss = self.encoder.get_reward(self.full_obs) if loss is None else loss
            robot_action, uncertain = output[0], output[1]
            
            for _ in range(5):
                sim_resized = cv2.resize(self.full_obs['agentview_image'], self.full_obs['tiago_head_image'].shape[:-1])
                cv2.imshow('obs', np.concatenate((sim_resized, self.full_obs['tiago_head_image']), axis=1)/255)
                cv2.waitKey(1)
            
            if loss > 1:
                self.mode = 'camera'
                break
            
            sim_obs, reward, terminated, truncated, _ = self.sim_env.step(robot_action)
            self.full_obs['is_policy_uncertain'] = np.array([int(loss > 0.5)])
            self.update_dict(sim_obs)
            loss = None

            if terminated or truncated:
                break

        return self.full_obs, reward, terminated, truncated, info

    def step(self, action):
        assert self.mode == 'camera', 'set to camera mode before stepping'
        
        real_obs, _, _, _, info = self.env.step({'head': action})
        self.update_dict(real_obs)
        truncated = False
        terminated = False
        loss = self.encoder.get_reward(copy_np_dict(self.full_obs))
        
        
        self.full_obs['is_policy_uncertain'] = np.array([int(loss > 0.5)])

        reward = -0.1
        if loss < 1:
            print('robot rollouts')
            self.mode = 'robot'
            _, _, _, _, _ = self.rollout_robot_until_uncertain(0, False, False, {})

            self.mode = 'camera'
            if self.sim_env._check_success():
                print("rewarded")
                reward = 5
                terminated = True
                truncated = False

        self.camera_step += 1
        if self.camera_step >= self.max_camera_steps:
            terminated = True
            truncated = False
            self.new_episode = True

        sim_resized = cv2.resize(self.full_obs['agentview_image'], self.full_obs['tiago_head_image'].shape[:-1])
        cv2.imshow('obs', np.concatenate((sim_resized, self.full_obs['tiago_head_image']), axis=1)/255)
        cv2.waitKey(1)

        
        print(f'{loss:0,.2f}', self.full_obs['is_policy_uncertain'])
        return copy_np_dict(self.full_obs), reward, terminated, truncated, info
    
class SimAndRealUncertaintyAwareWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        from l2l.config.env.robosuite.skill_color_pnp import env_config
        from l2l.utils.general_utils import get_env_from_config
        from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper

        self.sim_env = get_env_from_config(AttrDict(env_config=env_config, wrappers=[RobosuiteGymWrapper]))
    
        self.encoder = None

        self.mode = 'robot'
        self.camera_step = 0
        self.max_camera_steps = 32 #20
        self.max_camera_steps_per_stage = 32 #20

        self.new_episode = True
        self.full_obs = None

        self.z, self.reward = [], []

    @property   
    def observation_space(self):
        ob_space = OrderedDict()
        sim_ob_space = self.sim_env.observation_space
        real_ob_space = self.env.observation_space

        ob_space['privileged_info'] = sim_ob_space['privileged_info']
        for k in real_ob_space:
            ob_space[k] = real_ob_space[k]
        ob_space['is_policy_uncertain'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        return spaces.Dict(ob_space)
        
    @property
    def action_space(self):
        if hasattr(self.env, 'n_looking_actions'):
            return spaces.Discrete(self.env.n_looking_actions)
        return spaces.Discrete(5)
    
    def set_encoder(self, encoder):
        self.encoder = encoder
    
    def update_dict(self, d):
        '''
            Overwrite keys of d1 with those of d2
        '''
        for k in d.keys():
            assert k in self.full_obs.keys(), f'Key {k} not found in dictionary d1'
            self.full_obs[k] = d[k]
    
    def hard_reset(self, **kwargs):
        obs_real, info_real = self.env.reset(**kwargs)
        obs_sim, info_sim = self.sim_env.reset(**kwargs)

        obs = merge_dicts(obs_real, obs_sim)
        info = merge_dicts(info_real, info_sim)

        return obs, info

    def reset(self, **kwargs):
        if self.new_episode:
            self.full_obs, info = self.hard_reset(**kwargs)
            self.camera_step = 0
        else:
            info = {}

        self.mode = 'robot'
        _, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(0, False, False, {})
        if terminated or truncated:
            self.full_obs, info = self.hard_reset(**kwargs)
            self.camera_step = 0
            _, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(0, False, False, {})
            
        
        self.mode = 'camera'

        if len(self.z)> 0:
            self.z = np.array(self.z)
            mx = np.max(self.z, axis=0)
            mn = np.min(self.z, axis=0)
            print(f"({mx[0]:0,.2f}, {mx[1]:0,.2f}), ({mn[0]:0,.2f}, {mn[1]:0,.2f}), {np.mean(self.reward):0,.2f}")
            
        self.z, self.reward = [], []
        
        return copy_np_dict(self.full_obs), info

    def rollout_robot_until_uncertain(self, reward, terminated, truncated, info):
        while self.mode == 'robot':
            if self.encoder is None:
                output = (len(self.sim_env.unwrapped.skills)-1, False)
                loss = 1
            else:
                output = self.encoder.get_action_and_uncertainty(self.full_obs)
                loss = self.encoder.get_reward(self.full_obs)
            robot_action, uncertain = output[0], output[1]
            
            for _ in range(5):
                sim_resized = cv2.resize(self.full_obs['agentview_image'], self.full_obs['tiago_head_image'].shape[:-1])
                cv2.imshow('obs', np.concatenate((sim_resized, self.full_obs['tiago_head_image']), axis=1)/255)
                cv2.waitKey(1)
            
            if loss > 0.5:
                self.mode = 'camera'
                break
            
            sim_obs, reward, terminated, truncated, _ = self.sim_env.step(robot_action)
            self.update_dict(sim_obs)

            if terminated or truncated:
                break

        self.full_obs['is_policy_uncertain'] = np.array([int(loss > 0.5)])
        return self.full_obs, reward, terminated, truncated, info

    def step(self, action):
        assert self.mode == 'camera', 'set to camera mode before stepping'
        
        real_obs, _, _, _, info = self.env.step({'head': action})
        self.update_dict(real_obs)
        truncated = False
        terminated = False
        loss = self.encoder.get_reward(copy_np_dict(self.full_obs))
        reward = np.clip(1-loss, -1, 1)
        # loss = 0
        # reward = 0
        
        self.z.append(self.full_obs['head'])
        self.reward.append(reward)

        self.camera_step += 1
        if self.camera_step % self.max_camera_steps_per_stage == 0:
            terminated = True
            self.new_episode = False

        if self.camera_step >= self.max_camera_steps:
            terminated = True
            truncated = False
            self.new_episode = True

        sim_resized = cv2.resize(self.full_obs['agentview_image'], self.full_obs['tiago_head_image'].shape[:-1])
        cv2.imshow('obs', np.concatenate((sim_resized, self.full_obs['tiago_head_image']), axis=1)/255)
        cv2.waitKey(1)

        self.full_obs['is_policy_uncertain'] = np.array([int(loss > 0.5)])
        print(f'{loss:0,.2f}', self.full_obs['is_policy_uncertain'])
        return copy_np_dict(self.full_obs), reward, terminated, truncated, info

from l2l.utils.display_utils import FullscreenStringDisplay
class DisplayStringWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.display = FullscreenStringDisplay(update_interval=1, monitor_id=0)

        self.string1 = "8:00 AM"
        self.string2 = "5:15 PM"

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        
        if obs['privileged_info'][0]==1 :
            print(self.string1)
            self.display.display_text(self.string1, relx=0.3, rely=0.5)
        else:
            print(self.string2)
            self.display.display_text(self.string2, relx=0.3, rely=0.5)
        time.sleep(1)
        
        return obs, info

from PIL import Image
from l2l.utils.display_utils import FullscreenImageDisplay
class DisplayImageWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.display = FullscreenImageDisplay(update_interval=1, monitor_id=0)

        # img = np.array(Image.open("/home/shivin/Downloads/hot.png"))
        img = np.array(Image.open("/home/shivin/Downloads/odegaard_blurred.jpg"))
        img = img.transpose(1, 0, 2)
        self.img1 = self.display.get_resized_image(img)

        # img = np.array(Image.open("/home/shivin/Downloads/cold.png"))
        img = np.array(Image.open("/home/shivin/Downloads/mahrez_blurred.jpg"))
        img = img.transpose(1, 0, 2)
        self.img2 = self.display.get_resized_image(img)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        
        if obs['privileged_info'][0]==1 :
            print('odegaard')
            self.display.update_image(self.img1)
        else:
            print('mahrez')
            self.display.update_image(self.img2)
        time.sleep(1)
        
        return obs, info
    
class TiagoPointHeadWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.head_goal = np.array([0.6, 0.6])
        self.max_steps = 10
        self.cur_steps = 0

    @property
    def action_space(self):
        return spaces.Discrete(5)

    def reset(self, *args, **kwargs):
        self.cur_steps = 0
        obs, info = self.env.reset()
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step({'head': action})
        head_joints = obs['head']

        reward =  0.5 - np.linalg.norm(self.head_goal - head_joints)

        self.cur_steps += 1

        truncated = False
        if self.max_steps <= self.cur_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


class TiagoHandAndHeadWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.arm_to_move = 'left'

        self.lower_limits = np.array([-1, -1, -1])
        self.upper_limits = np.array([1, 0, 1])

        self.n_looking_actions = 5 + 3
    
    @property   
    def observation_space(self):
        ob_space = OrderedDict()
        real_ob_space = self.env.observation_space

        for k in real_ob_space:
            ob_space[k] = real_ob_space[k]
        
        ob_space['arm_x_position'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,),) 
        ob_space['screen_on'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,),)
        return spaces.Dict(ob_space)

    @property
    def action_space(self):
        return spaces.Discrete(self.n_looking_actions)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        self.net_movement = np.zeros(3).astype(int)

        obs['arm_x_position'] = np.array([self.net_movement[0]])
        return obs, info

    def execute_arm_movement(self, xyz_delta):
        arm_action = np.zeros(7)
        arm_action[-1] = 1
        arm_action[:3] = xyz_delta
        
        self.env.step({self.arm_to_move: arm_action})
        # print(obs['left'])
        time.sleep(1.5)
    
    def do_arm_skill(self, action):
        assert 5 <= action < 10, f"Arm action {action} not defined"
        turn_display_on = False

        if action!=7:
            if action == 5: # +x
                xyz_delta = np.array([0.1, 0, 0])
                if self.net_movement[0] < self.upper_limits[0]:
                    self.net_movement[0] += 1
                    self.execute_arm_movement(xyz_delta)
            elif action == 6: # -x
                xyz_delta = np.array([-0.1, 0, 0])
                if self.net_movement[0] > self.lower_limits[0]:
                    self.net_movement[0] -= 1
                    self.execute_arm_movement(xyz_delta)
            # elif action == 8: # +y
            #     xyz_delta = np.array([0, 0.1, 0])
            #     if self.net_movement[1] < self.upper_limits[1]:
            #         self.net_movement[1] += 1
            #         self.execute_arm_movement(xyz_delta)
            # elif action == 9: # -y
            #     xyz_delta = np.array([0, -0.1, 0])
            #     if self.net_movement[1] > self.lower_limits[1]:
            #         self.net_movement[1] -= 1
            #         self.execute_arm_movement(xyz_delta)

        xyz_delta = np.array([0, 0, -0.02])
        self.execute_arm_movement(xyz_delta)
        xyz_delta = np.array([0, 0, 0.02])
        self.execute_arm_movement(xyz_delta)

        if self.net_movement[0]==1 and self.net_movement[1]==0:
            turn_display_on = True

        obs = self.env._observation()
        return obs, 0, False, False, {'turn_display_on': turn_display_on}

    def step(self, action):
        action = action['head'] # action in this format so that it's compatible with SimAndReal Wrapper
        if action < 5:
            obs, reward, terminated, truncated, info = self.env.step({'head': action})
            time.sleep(1)
            obs = self.env._observation()
        else:
            obs, reward, terminated, truncated, info = self.do_arm_skill(action)
        obs['arm_x_position'] = np.array([self.net_movement[0]])
        return obs, 0, False, False, info

class OnOffDisplayImageWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.display = FullscreenImageDisplay(update_interval=1, monitor_id=0)

        img = np.array(Image.open("/home/shivin/Downloads/apple.jpg"))
        img = img.transpose(1, 0, 2)
        self.img1 = self.display.get_resized_image(img)

        img = np.array(Image.open("/home/shivin/Downloads/pear.jpg"))
        img = img.transpose(1, 0, 2)
        self.img2 = self.display.get_resized_image(img)

        self.black_img = np.ones_like(self.img2).astype(np.uint8)
        
        self.screen_on = False

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        self.display.update_image(self.black_img)
        
        self.screen_on = False
        obs['screen_on'] = np.array([int(self.screen_on)])
        return obs, info
    
    def rotate_camera(self, action):
        obs, reward, terminated, truncated, info = self.env.rotate_camera(action)
        
        if info.get('turn_display_on', False):
            print("Turning display on")
            self.screen_on = True
            if obs['privileged_info'][0]==1 :
                print('red apple')
                self.display.update_image(self.img1)
            else:
                print('green pear')
                self.display.update_image(self.img2)
            time.sleep(1)
            obs['tiago_head_image'] = np.array(self.env.unwrapped.cameras['tiago_head'].get_img(), dtype=np.float32)

        obs['screen_on'] = np.array([int(self.screen_on)])
        return obs, reward, terminated, truncated, info
        
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # print('priv', obs['privileged_info'])
        if info.get('turn_display_on', False):
            print("Turning display on")
            self.screen_on = True
            if obs['privileged_info'][0]==1 :
                # print('blue on left')
                self.display.update_image(self.img1)
            else:
                # print('blue on right')
                self.display.update_image(self.img2)
            time.sleep(1)
            obs['tiago_head_image'] = np.array(self.env.unwrapped.cameras['tiago_head'].get_img(), dtype=np.float32)

        obs['screen_on'] = np.array([int(self.screen_on)])
        return obs, reward, terminated, truncated, info

import gymnasium
class CameraAndHandExplorationBonus(gymnasium.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}
        self.exploration_bonus = True
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.exploration_bonus:
            state_action = (tuple(((np.array(obs['head']) + 1)*100).astype(np.int16)), 
                            tuple(((np.array(obs['left'][:2]) + 1)*100).astype(np.int16)),
                            action)#, tuple(obs['stage_id']))
            if state_action in self.counts:
                self.counts[state_action] += 1
            else:
                self.counts[state_action] = 1
            
            bonus = 1/np.sqrt(self.counts[state_action])
            reward += bonus
        return obs, reward, terminated, truncated, info