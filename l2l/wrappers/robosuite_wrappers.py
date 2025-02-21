import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env
from collections import OrderedDict
from robosuite.wrappers import Wrapper

class RobosuiteGymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None

    def __init__(self, env):
        super().__init__(env=env)
        self.name = type(self.env).__name__

        self.obs_keys = []
        self.env.spec = None

        obs = self.env.reset()
        for k in obs.keys():
            self.obs_keys.append(k)

        self.modality_dims = {key: obs[key].shape for key in self.obs_keys}

        if self.env.action_space is not None:
            self.action_space = self.env.action_space
        else:
            low, high = self.env.action_spec
            self.action_space = spaces.Box(low, high)

    @property   
    def observation_space(self):
        ob_space = OrderedDict()
        for key in self.obs_keys:
            ob_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=self.modality_dims[key])
        return spaces.Dict(ob_space)
    
    def filter_obs(self, obs):
        return {key: obs[key] for key in self.obs_keys}
    
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("seed must be an integer")
        return self.filter_obs(self.env.reset()), {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # if self.env._check_success():
        #     print('Gym Wrapper - Success!')
        #     done = True
        return self.filter_obs(obs), reward, done, False, info

class FrameStackWrapper(gym.core.Wrapper):

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = {k: [] for k in self.env.observation_space.spaces.keys()}
        
    @property
    def observation_space(self):
        ob_space = OrderedDict()
        for key in self.frames.keys():
            space = self.env.observation_space.spaces[key]
            if len(space.shape) == 1:
                ob_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stack*space.shape[0],))
            elif len(space.shape) == 3:
                ob_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(space.shape[0], space.shape[1], self.num_stack*space.shape[2]))
            else:
                raise ValueError('Invalid shape', key, space)
        return spaces.Dict(ob_space)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for k in self.frames.keys():
            self.frames[k] = [obs[k] for _ in range(self.num_stack)]
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._append_frame(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _append_frame(self, obs):
        for k in self.frames.keys():
            self.frames[k].append(obs[k])
            if len(self.frames[k]) > self.num_stack:
                self.frames[k].pop(0)

    def _get_obs(self):
        # return {k: np.concatenate(self.frames[k], axis=-1) for k in self.frames.keys()}
        obs = {}
        for k in self.frames.keys():
            obs[k] = np.concatenate(self.frames[k], axis=-1)
            # if len(space.shape) == 1:
            #     obs[k] = np.concatenate(self.frames[k], axis=0)
            # elif len(space.shape) == 3:
            #     obs[k] = 
            # else:
            #     raise ValueError('Invalid shape', k, space)
        return obs

class FullRLBaselineWrapper(gym.core.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.reward_type = 'stage' # task or stage
        self.max_steps = 150
        self.cur_steps = 0

        self.seed = None

    @property
    def action_space(self):
        return spaces.Discrete(self.env.unwrapped.n_camera_actions + len(self.env.unwrapped.skills))
    
    def reset(self, **kwargs):
        if self.seed is None:
            self.seed = -1#kwargs.get('seed', None)

        obs, info = self.env.reset(**kwargs)
        self.cur_steps = 0
        self.cur_stage = self.env.unwrapped.stage_id
        return obs, info


    def step(self, action):
        if action < len(self.env.unwrapped.skills):
            obs, _, done, _, info = self.env.step(action)
        else:
            action -= len(self.env.unwrapped.skills)
            obs, _, done, info = self.env.unwrapped.rotate_camera(action)
        self.cur_steps += 1

        reward = -0.1
        if self.reward_type == 'stage':
            new_stage = self.env.unwrapped.stage_id
            if new_stage > self.cur_stage:
                reward = 5
                self.cur_stage = new_stage
        
        done=False
        if self.env.unwrapped._check_success():
            reward = 5
            done = True
        if self.env.unwrapped._check_failure():
            reward = -5
            done = True

        if self.cur_steps >= self.max_steps:
            done = True

        if self.seed == 0:
            print(reward)
            for _ in range(5):
                self.env.render()
            cv2.imshow('obs', obs['activeview_image']/255)
            cv2.waitKey(1)
    
        return obs, reward, done, False, info
    
class RewardBasedLookingWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.encoder = None

        self.mode = 'robot'
        self.camera_step = 0
        self.max_camera_steps = 320
        
        self.new_episode = True
        self.last_obs = None

        self.reward = []

        self.reward_type = 'stage'

        self.seed = None

    @property
    def action_space(self):
        return spaces.Discrete(self.env.unwrapped.n_camera_actions)
    
    def set_encoder(self, encoder):
        self.encoder = encoder
    
    def reset(self, **kwargs):
        if self.seed is None:
            self.seed = -1#kwargs.get('seed', None)

        if self.new_episode:
            obs, info = self.env.reset(**kwargs)
            self.camera_step = 0
        else:
            obs, info = self.last_obs
        
        self.mode = 'robot'
        obs, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(obs, 0, False, False, {})
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
            obs, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(obs, 0, False, False, {})
            self.camera_step = 0
        
        self.cur_stage = self.env.unwrapped.stage_id

        self.mode = 'camera'

        if len(self.reward)> 0:
            print(f"{np.mean(self.reward):0,.2f}")
            
        self.reward = []
        return obs, info

    def rollout_robot_until_uncertain(self, obs, reward, terminated, truncated, info, loss=None):
        while self.mode == 'robot':
            if self.encoder is None:
                output = (len(self.env.unwrapped.skills)-1, False)
                loss = 1
            else:
                output = self.encoder.get_action_and_uncertainty(obs)
                loss = self.encoder.get_reward(obs) if loss is None else loss
            robot_action, uncertain = output[0], output[1]

            if loss > 0.5:
                self.mode = 'camera'
                break
            
            obs, reward, terminated, truncated, info = self.env.step(robot_action)
            obs['is_policy_uncertain'] = np.array([int(loss > 0.5)])
            loss=None

            if self.seed == 0:
                for _ in range(5):
                    self.env.render()

            if terminated or truncated:
                break
        
        return obs, reward, terminated, truncated, info

    def step(self, action):
        assert self.mode == 'camera', 'set to camera mode before stepping'
        
        obs, _, _, info = self.env.unwrapped.rotate_camera(action)
        truncated = False
        terminated = False
        loss = self.encoder.get_reward(obs)

        obs['is_policy_uncertain'] = np.array([int(loss > 0.5)])
        reward = -0.1
        if loss < 0.5:
            self.mode = 'robot'
            _, _, _, _, _ = self.rollout_robot_until_uncertain(obs, 0, False, False, {}, loss=loss)

            if self.reward_type == 'stage':
                new_stage = self.env.unwrapped.stage_id
                if new_stage > self.cur_stage:
                    reward = 10
                    self.cur_stage = new_stage
            if self.env.unwrapped._check_success():
                reward = 10
                terminated = True
                truncated = False

        self.reward.append(reward)
        if self.seed == 0:
            cv2.imshow('obs', obs['activeview_image']/255)
            cv2.waitKey(1)

        self.camera_step += 1
        if self.camera_step >= self.max_camera_steps:
            terminated = True
            truncated = False
            self.new_episode = True

        return obs, reward, terminated, truncated, info

class UncertaintyBasedLookingWrapper(gym.core.Wrapper):


    def __init__(self, env):
        super().__init__(env)
        self.encoder = None

        self.mode = 'robot'
        self.camera_step = 0
        self.max_camera_steps = 320
        self.max_camera_steps_per_stage = 64
        
        self.new_episode = True
        self.last_obs = None

        self.z, self.reward = [], []

        self.seed = None

    @property
    def action_space(self):
        return spaces.Discrete(self.env.unwrapped.n_camera_actions)
    
    def set_encoder(self, encoder):
        self.encoder = encoder
    
    def reset(self, **kwargs):
        if self.seed is None:
            self.seed = -1#kwargs.get('seed', None)

        if self.new_episode:
            obs, info = self.env.reset(**kwargs)
            self.camera_step = 0
        else:
            obs, info = self.last_obs
        
        self.mode = 'robot'
        obs, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(obs, 0, False, False, {})
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
            obs, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(obs, 0, False, False, {})
            self.camera_step = 0

        self.mode = 'camera'

        if len(self.z)> 0:
            print(f"{np.max(self.z):0,.2f}, {np.min(self.z):0,.2f}, {np.mean(self.reward):0,.2f}")
            
        self.z, self.reward = [], []
        return obs, info

    def rollout_robot_until_uncertain(self, obs, reward, terminated, truncated, info):
        while self.mode == 'robot':
            if self.encoder is None:
                output = (len(self.env.unwrapped.skills)-1, False)
                loss = 1
                # visuomotor
                # output = (np.zeros(4), False) #(len(self.env.unwrapped.skills)-1, False)
                # loss = 1e12
            else:
                # skill
                output = self.encoder.get_action_and_uncertainty(obs)
                
                # visuomotor
                # output = (self.encoder.get_gt_action(obs), None)
                loss = self.encoder.get_reward(obs)
            robot_action, uncertain = output[0], output[1]
            # robot_action = gt
            
            if self.seed == 0:
                # print(self.env.unwrapped.id_to_skill[robot_action])
                for _ in range(5):
                    self.env.render()

            # visuomotor
            # if loss > 5e10:
            if loss > 0.5:
                self.mode = 'camera'
                break
                print(loss)
                print(obs['stage_id'])
                input('enter to continue')
            
            obs, reward, terminated, truncated, info = self.env.step(robot_action)

            if self.seed == 0:
                for _ in range(5):
                    self.env.render()

            if terminated or truncated:
                break
        
        obs['is_policy_uncertain'] = np.array([int(loss > 0.5)])
        return obs, reward, terminated, truncated, info

    def step(self, action):
        assert self.mode == 'camera', 'set to camera mode before stepping'
        
        obs, _, _, info = self.env.unwrapped.rotate_camera(action)
        truncated = False
        terminated = False
        loss = self.encoder.get_reward(obs)
        reward = np.clip(1-loss, -1, 1)

        # visuomotor
        # if loss > 5e10:
        #     reward = -1
        # else:
        #     reward = 1

        self.z.append(obs['camera_angle'])
        self.reward.append(reward)
        if self.seed == 0:
            cv2.imshow('obs', obs['activeview_image']/255)
            cv2.waitKey(1)

        self.camera_step += 1
        if self.camera_step % self.max_camera_steps_per_stage == 0:
            terminated = True
            self.new_episode = False
            self.last_obs = (obs, info)

        if self.camera_step >= self.max_camera_steps:
            terminated = True
            truncated = False
            self.new_episode = True

        obs['is_policy_uncertain'] = np.array([int(loss > 0.5)])

        return obs, reward, terminated, truncated, info

class CameraExplorationBonus(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}
        self.exploration_bonus = True
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.exploration_bonus:
            state_action = (tuple(((np.array(obs['camera_angle']) + 1)*100).astype(np.int16)), action)#, tuple(obs['stage_id']))
            if state_action in self.counts:
                self.counts[state_action] += 1
            else:
                self.counts[state_action] = 1
            
            bonus = 1/np.sqrt(self.counts[state_action])
            reward += bonus
        return obs, reward, terminated, truncated, info

class RobosuiteDiscreteControlWrapper(gym.core.Wrapper):
    '''
        Maps discrete integer actions into continuous list of actions.
        If a list action is provided then it sends it directly to the environment without processing.
    '''

    def __init__(self, env):
        super().__init__(env=env)

        self.action_map = {
            0: np.array([0, 0, 0, 0]),
            1: np.array([1, 0, 0, 0]),
            2: np.array([-1, 0, 0, 0]),
            3: np.array([0, 1, 0, 0]),
            4: np.array([0, -1, 0, 0]),
            5: np.array([0, 0, 1, 0]),
            6: np.array([0, 0, -1, 0]),
            7: np.array([0, 0, 0, 1]), # close gripper
            8: np.array([0, 0, 0, -1]) # open gripper
        }

        self.action_dim = len(list(self.action_map.keys()))
        self.max_action = 0.5

    @property
    def action_space(self):
        return spaces.Discrete(self.action_dim)

    def step(self, action):
        if isinstance(action, list) or isinstance(action, np.ndarray):
            return self.env.step(action)
        action_id = action
        action = self.action_map[int(action)]
        if action_id in list(range(1, 7)):
            action = self.max_action * action / np.linalg.norm(action)
        return self.env.step(action)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

class ReachButtonRewardWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.max_steps = 300
        self.step_count = 0
    
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            trunc = True
        
        reward = 0

        from_xyz = obs['robot0_eef_pos']
        to_xyz = obs['Button1_pos'] + np.array([0, 0, 0.1])

        dist = np.linalg.norm(from_xyz - to_xyz)
        reward += (0.1 - dist)
        
        # if dist < 0.1:
        #     reward += 0.3

        if self.env.unwrapped.buttons_on[1]:
            reward += 70
            done = True
        return obs, reward, done, trunc, info
    
    def reset(self, seed=None):
        self.step_count = 0
        return self.env.reset(seed=seed)


class AppendPolicyWrapper(gym.core.Wrapper):

    def __init__(self, env, policy_A_steps=30):
        super().__init__(env)
        self.policy = None
        self.start_A = 0
        self.policy_A_steps = policy_A_steps

        self.max_steps = 400 #850
        self.step_count = 0
    
    def set_policy(self, policy):
        self.policy = policy
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.step_count = 0
        self.start_A = np.random.randint(0, 151)#(self.start_A + 10)%150
        self.policy_A_steps_shifted = self.policy_A_steps + self.start_A

        if self.policy is None:
            return obs

        n_obs = obs[0]
        self.policy.start_episode()
        while self.step_count < self.start_A:
            self.step_count += 1
            a = self.policy.get_action(n_obs)
            # print('reset', self.step_count)
            n_obs, reward, terminated, truncated, info = self.env.step(a)

            if terminated or truncated:
                # self.start_A = 0
                # self.policy_A_steps_shifted = self.policy_A_steps
                return self.env.reset(**kwargs)
            
        # self.start_A = np.random.randint(0, 151)#(self.start_A + 10)%150
        # self.policy_A_steps_shifted = self.policy_A_steps + self.start_A
        return obs

    def step(self, action):
        """Steps through the environment with `action`."""
        assert self.policy is not None, 'Please set a policy before running the environment'
        
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward = 0
        self.step_count += 1
        # print('step', self.step_count)
        if self.step_count >= self.policy_A_steps_shifted:
            # run the IL policy
            n_obs = obs
            self.policy.start_episode()
            while not (terminated or truncated):
                a = self.policy.get_action(n_obs)
                n_obs, n_reward, terminated, truncated, n_info = self.env.step(a)
                
                #
                # print('B policy', time.time(), a)
                # cv2.imshow('obs', cv2.cvtColor(self.render(), cv2.COLOR_BGR2RGB))
                # cv2.waitKey(30)

                if terminated:
                    # if n_reward == 0:
                    #     reward = 10
                    # else:
                    reward = -5

                if truncated or self.max_steps < self.step_count:
                    truncated = True
                    reward = 5

                self.step_count += 1
                # print('end', self.step_count)

        return obs, reward, terminated, truncated, info