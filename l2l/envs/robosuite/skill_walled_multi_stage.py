import cv2
import numpy as np
from collections import OrderedDict

from gymnasium import spaces

from l2l.envs.robosuite.walled_multi_stage import WalledMultiStage
from l2l.envs.robosuite.skill_base import SkillBase

class SkillWalledMultiStage(WalledMultiStage, SkillBase):

    def __init__(self, **kwargs):
        WalledMultiStage.__init__(self, **kwargs)
        SkillBase.__init__(self, self.action_spec)

    def reset(self):
        obs = super().reset()
        self.reset_skills()
    
        return obs
    
    @property
    def action_space(self):
        return spaces.MultiDiscrete([len(self.skills)])
    
    def create_skills(self):
        self.skills = OrderedDict()
        
        self.skills['move_to_blue_block'] = [
                ['move', 'cube_blue_pos', np.array([0, 0, 0.05])],
                ['move', 'cube_blue_pos', np.array([0, 0, 0])],
            ]
        
        self.skills['move_to_wood_block'] = [
                ['move', 'cube_wood_pos', np.array([0, 0, 0.05])],
                ['move', 'cube_wood_pos', np.array([0, 0, 0])],
            ]
        
        self.skills['pick_block'] = [
                ['grasp', [0, 0, 0, 1], 5],
                ['move', 'delta', np.array([0, 0, 0.05])],
            ]
        
        self.skills['move_to_red'] = [
                ['move', 'serving_region_red_pos', np.array([0, 0, 0.05])],
            ]
        
        self.skills['move_to_green'] = [
                ['move', 'serving_region_green_pos', np.array([0, 0, 0.05])],
            ]
        
        self.skills['place_block'] = [
                ['grasp', [0, 0, 0, -1], 5],
            ]
        
        self.skills["wait"] = [
            ["wait", None, 1],
        ]
    
    def get_optimal_skill_sequence(self):
        pick_block =  'move_to_blue_block' if self.pick_blue else 'move_to_wood_block'
        place_region = 'move_to_red' if self.place_on_red else 'move_to_green'
        
        return [pick_block, 'pick_block', place_region, 'place_block']

    def step(self, skill_id):
        skill_id = int(skill_id)
        assert len(self.skills) > skill_id >= 0, f"Invalid skill_id: {skill_id}"

        self.skill_step += 1

        skill_name = self.id_to_skill[skill_id]
        return self.execute_skill(skill_name)
    
    def execute_skill(self, skill_name):
        render = False
        current_skill = self.skills[skill_name]

        obs = self.get_processed_obs()
        reward = 0
        done = False
        info = {}

        for skill in current_skill:
            if skill[0] == 'move':
                from_xyz = obs['robot0_eef_pos']
                to_xyz = self.get_goal_pos(obs, skill)
                
                max_loops = 100
                while max_loops > 0 and np.linalg.norm(from_xyz - to_xyz) > 0.02:
                    action = np.zeros(4)
                    action[:3] = self.move(from_xyz, to_xyz)
                    action[3] = self.last_gripper_act
                    obs, reward, done, info = super().step(action)
                    from_xyz = obs['robot0_eef_pos']
                    max_loops -= 1

                    if render:
                        self.render()
                
                if max_loops == 0:
                    print("Failed to reach goal")

            elif skill[0] == 'grasp':
                n_grasp_steps = skill[2]
                for _ in range(max(1, n_grasp_steps)):
                    action = skill[1]
                    self.last_gripper_act = action[-1]
                    obs, reward, done, info = super().step(action)
            elif skill[0] == 'wait':
                action = np.zeros(4)
                action[3] = self.last_gripper_act
                for _ in range(skill[2]):
                    obs, reward, done, info = super().step(action)
            else:
                raise NotImplementedError
        return obs, reward, done or (self.skill_step > self.horizon), info

if __name__=="__main__":
    from l2l.config.env.robosuite.skill_walled_multi_stage import env_config
    import robosuite as suite
    import cv2
    import matplotlib.pyplot as plt

    from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper, UncertaintyBasedLookingWrapper
    from stable_baselines3 import PPO

    from l2l.config.dual.robosuite.skill_walled_multi_stage.walled_multi_stage_action_dual_config import encoder_model_config

    def get_encoder_path(rl_path):
        rl_ckpt = rl_path.split('/')[-3].split('_')[-1]

        encoder_path = "/".join(rl_path.split('/')[:-3]) + f"/encoder_weights/weights/weights_ep{rl_ckpt}.pth"
        return encoder_path
    
    

    # initialize the task
    # env = RobosuiteGymWrapper(suite.make(**env_config))
    # env = UncertaintyBasedLookingWrapper(RobosuiteGymWrapper(suite.make(**env_config)))
    
    # rl_ckpt = '/home/shivin/Desktop/learning2look/experiments/dual_multi_stage_large_actions_no_stage_id_2/epoch_18/weights/rl_model_456800_steps.zip'
    # encoder_path = get_encoder_path(rl_ckpt)
    # cam_policy = PPO.load(rl_ckpt, env=env)
    # enc = encoder_model_config.model_class.load_weights(encoder_path)
    
    # env.set_encoder(enc)
    
    # obs, _ = env.reset()

    # from collections import OrderedDict
    # bins = OrderedDict()
    # bin_counts = OrderedDict()
    # for i in range(20):
    #     bins[-10 + i] = 0
    #     bin_counts[-10 + i] = 0

    # print(bins.keys())

    # action = 2
    # for i in range(1000):
    #     # action = env.action_space.sample()
    #     action, states = cam_policy.predict(
    #                         obs,  # type: ignore[arg-type]
    #                         deterministic=True,
    #                     )
    #     # if obs['camera_angle']>0.9 or obs['camera_angle']<-0.9:
    #     #     action = action%2 + 1
    #     #     print(action)
        
    #     # if i % 2 == 0:
    #     #     action = int(input("enter action:"))
    #     # print(i)

    #     obs, reward, done, _, _ = env.step(action)

    #     # for _ in range(10):
    #     #     loss = env.encoder.get_reward(obs)
    #     #     reward = np.clip(2*(0.5-loss), -1, 1)
    #     #     bins[int((obs['camera_angle'])*10)] += reward
    #     #     bin_counts[int((obs['camera_angle'])*10)] += 1
    #     # print(obs['camera_angle'], int((obs['camera_angle'] + 1)*100), reward)

    #     img = obs['activeview_image'].astype(np.float32)/255
    #     # env.render()
    #     cv2.imshow('img', img)
    #     cv2.waitKey(1)

    #     # if obs["camera_angle"] < -0.95:
    #     if done:
    #         # print(env.uncertainty)
    #         # plt.plot(env.uncertainty)
    #         # plt.show()
            
    #         # print(bin_counts)
    #         # x = [bins[k]/bin_counts[k] if bin_counts[k] > 0 else 0 for k in bins.keys()]
    #         # plt.bar(list(bins.keys()), x)
    #         # plt.show()

    #         env.reset()

    env = RobosuiteGymWrapper(suite.make(**env_config))
    obs, _ = env.reset()
    for _ in range(10):
        env.render()

    steps = 0
    for i in range(10000):
        robot_action = env.get_optimal_action(obs)
        print(obs['privileged_info'])
        robot_action = int(input('enter action:'))
        # robot_action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(robot_action)

        # for i in range(10):
        #     camera_action = np.random.choice(env.unwrapped.n_camera_actions)
        #     obs, _, _, _ = env.unwrapped.rotate_camera(camera_action)
        #     img = np.concatenate([obs['agentview_image']/255, obs['activeview_image']/255], axis=1)
        #     cv2.imshow('img', img)
        #     cv2.waitKey(1)
        #     print(obs['camera_pos'])
        steps += 1

        for _ in range(10):
            env.render()
        
        # env.render()
        # if i % 20 == 0:
        #     camera_action = int(input())

        if env._check_success():
            print("Success")
            obs, _ = env.reset()
            steps = 0
        
        if env._check_failure():
            print("Failure")
            obs, _ = env.reset()
            steps = 0

        # if done:
        #     env.reset()
        #     steps = 0