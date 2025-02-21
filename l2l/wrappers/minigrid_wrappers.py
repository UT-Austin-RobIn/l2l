import numpy as np
from gymnasium import spaces
from gymnasium.core import Wrapper, ObservationWrapper
from collections import OrderedDict
import cv2, time

class CompleteStateWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "ego_grid": env.observation_space["image"],
                # "agent_pos": spaces.MultiDiscrete([self.env.height, self.env.width]),
                "agent_pos": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.env.height-1, self.env.width-1]),
                    shape=(2,)
                ),
                "agent_dir": spaces.Box(
                    low=0,
                    high=3,
                    shape=(1,)
                ),
            }
        )

    def observation(self, obs):
        new_obs = OrderedDict({
            "ego_grid": obs['image'],
            "agent_pos": self.env.unwrapped.agent_pos,
            "agent_dir": [self.env.unwrapped.agent_dir]
        })

        return new_obs

class AppendPolicyWrapper(Wrapper):

    def __init__(self, env, policy_A_steps=30):
        super().__init__(env)
        self.policy = None
        self.start_A = 0
        self.policy_A_steps = policy_A_steps
    
    def set_policy(self, policy):
        self.policy = policy
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.start_A = np.random.randint(0, 15)
        self.policy_A_steps_shifted = self.policy_A_steps + self.start_A

        if self.policy is None:
            return obs

        n_obs = obs[0]
        while self.env.unwrapped.step_count < self.start_A:
            a = self.policy.get_action(n_obs)
            n_obs, reward, terminated, truncated, info = self.env.step(a)

            if terminated or truncated:
                # self.start_A = 0
                # self.policy_A_steps_shifted = self.policy_A_steps
                return self.env.reset(**kwargs)
            
        # self.start_A += 5
        # self.policy_A_steps_shifted = self.policy_A_steps + self.start_A
        return obs

    def step(self, action):
        """Steps through the environment with `action`."""
        assert self.policy is not None, 'Please set a policy before running the environment'
        
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward = -1 if (terminated or truncated) else 0
        if self.env.unwrapped.step_count >= self.policy_A_steps_shifted:
            # run the IL policy
            n_obs = obs
            while not (terminated or truncated):
                a = self.policy.get_action(n_obs)
                n_obs, n_reward, terminated, truncated, n_info = self.env.step(a)
                
                # if self.render_B_policy:
                #     print('B policy', time.time(), a)
                #     cv2.imshow('obs', cv2.cvtColor(self.render(), cv2.COLOR_BGR2RGB))
                #     cv2.waitKey(30)
                if terminated:
                    if n_reward == 0:
                        reward = 1
                    else:
                        reward = -1
                if truncated:
                    reward = 1

        return obs, reward, terminated, truncated, info
