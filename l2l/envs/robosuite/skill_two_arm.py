import cv2
import numpy as np
from collections import OrderedDict

from gymnasium import spaces

from l2l.envs.robosuite.two_arm import TwoArm
from l2l.envs.robosuite.skill_base import SkillBase

class SkillTwoArm(TwoArm, SkillBase):

    def __init__(self, **kwargs):
        TwoArm.__init__(self, **kwargs)
        SkillBase.__init__(self, self.action_spec)
        self.create_camera_skills()
        self.cam_skill_step = 1

        self.n_camera_actions= 5 + len(self.cam_skills)

    def reset(self):
        obs = super().reset()
        self.reset_skills()
        return obs

    def reset_skills(self):
        super().reset_skills()
        self.last_gripper2_act = -1
        self.drawer1_is_open = False
        self.drawer2_is_open = False
        self.free_hand = True
    
    @property
    def action_space(self):
        return spaces.MultiDiscrete([len(self.skills)])
    
    def create_skills(self):
        self.skills = OrderedDict()

        self.skills['move_to_nut'] = [
            ['move', 'nut_pos', np.array([-0.075, 0, 0.05])],
            ['move', 'nut_pos', np.array([-0.075, 0, -0.005])],
        ]
        
        self.skills['pick_nut'] = [
                ['grasp', [0, 0, 0, 1], 5],
                ['move', 'delta', np.array([0, 0, 0.2])],
            ]
        
        self.skills['move_to_blue'] = [
                ['move', 'peg1_pos', np.array([-0.075, 0, 0.2])],
                ['move', 'peg1_pos', np.array([-0.075, 0, 0.0])]
            ]
        
        self.skills['move_to_green'] = [
                ['move', 'peg2_pos', np.array([-0.075, 0, 0.2])],
                ['move', 'peg2_pos', np.array([-0.075, 0, 0.0])]
            ]
        
        self.skills['place_block'] = [
                ['grasp', [0, 0, 0, -1], 5],
            ]
        
        self.skills["wait"] = [
            ["wait", None, 1],
        ]

    def create_camera_skills(self):
        self.cam_skills = OrderedDict()
        self.cam_skills['open_drawer1'] = [
            ['move', 'drawer1_pos', np.array([0., -0.25, 0.2])],
            ['move', 'drawer1_pos', np.array([0., -0.25, 0.1])],
            ['grasp', [0,0,0,1], 5],
            ['move', 'delta', np.array([0., -0.2, 0.0])],
            ['grasp', [0,0,0,-1], 5],
            ['move', 'delta', np.array([0., 0., 0.2])],
        ]
        self.cam_skills['open_drawer2'] = [
            ['move', 'drawer2_pos', np.array([0., -0.25, 0.2])],
            ['move', 'drawer2_pos', np.array([0., -0.25, 0.1])],
            ['grasp', [0,0,0,1], 5],
            ['move', 'delta', np.array([0., -0.2, 0.0])],
            ['grasp', [0,0,0,-1], 5],
            ['move', 'delta', np.array([0., 0., 0.2])],
        ]
        # self.cam_skills['pick_blue_block'] = [
        #         ['move', 'cube_blue_pos', np.array([0, 0, 0.1])],
        #         ['move', 'cube_blue_pos', np.array([0, 0, 0])],
        #         ['grasp', [0, 0, 0, 1], 5],
        #         ['move', 'delta', np.array([0, 0, 0.1])],
        #     ]
        
        self.cam_skills['pick_wood_block'] = [
                ['move', 'cube_wood_pos', np.array([0, 0, 0.1])],
                ['move', 'cube_wood_pos', np.array([0, 0, 0])],
                ['grasp', [0, 0, 0, 1], 5],
                ['move', 'delta', np.array([0, 0, 0.1])],
            ]
        
        self.cam_skills['place_on_drawer1'] = [
                ['move', 'drawer1_pos', np.array([0, 0, 0.25])],
                ['grasp', [0, 0, 0, -1], 5],
            ]
        
        # self.cam_skills['place_on_drawer2'] = [
        #         ['move', 'drawer2_pos', np.array([0, 0, 0.25])],
        #         ['grasp', [0, 0, 0, -1], 5],
        #     ]

        self.cam_id_to_skill = {i: skill for i, skill in enumerate(self.cam_skills.keys())}
    
    def get_optimal_skill_sequence(self):
        pick_block =  'move_to_nut'
        place_region = 'move_to_blue' if self.place_on_blue else 'move_to_green'
        
        return [pick_block, 'pick_nut', place_region, 'place_block']

    def step(self, skill_id):

        skill_id = int(skill_id)
        assert len(self.skills) > skill_id >= 0, f"Invalid skill_id: {skill_id}"

        self.skill_step += 1

        skill_name = self.id_to_skill[skill_id]
        return self.execute_skill(skill_name)

    def rotate_camera(self, camera_action):

        # maybe rotate cam
        if camera_action <= 4:
            return super().rotate_camera(camera_action)

        # else, cam action is a skill 
        # account for first 4 actions being primitive 
        camera_action = int(camera_action) - 5
        assert len(self.cam_skills) > camera_action >= 0, f"Invalid camera skill_id: {camera_action}"
        
        # execute skill
        self.cam_skill_step += 1
        cam_skill_name = self.cam_id_to_skill[camera_action]
        return self.execute_cam_skill(cam_skill_name)
    
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
                    print(f"Could not reach goal: {skill[1]}")

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

    def execute_cam_skill(self, skill_name):
        render = False
        dummy_act = np.zeros(4)
        dummy_act[-1] = self.last_gripper2_act
        if skill_name =='open_drawer1':
            if self.drawer1_is_open or not self.free_hand:
                return super().step2(dummy_act)
        
        if skill_name =='open_drawer2':
            if self.drawer2_is_open or not self.free_hand:
                return super().step2(dummy_act)

        if "pick" in skill_name: 
            if not self.free_hand:
                return super().step2(dummy_act)

        current_skill = self.cam_skills[skill_name]

        obs = self.get_processed_obs()
        reward = 0
        done = False
        info = {}

        for skill in current_skill:

            if skill[0] == 'move':
                from_xyz = obs['robot1_eef_pos']
                to_xyz = self.get_goal_pos(obs, skill, xyz_key='robot1_eef_pos')
                max_loops = 100
                while max_loops > 0 and np.linalg.norm(from_xyz - to_xyz) > 0.02:
                    action = np.zeros(4)
                    action[:3] = self.move(from_xyz, to_xyz)
                    action[3] = self.last_gripper2_act
                    obs, reward, done, info = super().step2(action)
                    from_xyz = obs['robot1_eef_pos']
                    max_loops -= 1

                    if render:
                        self.render()
                if max_loops == 0:
                    print(f"Could not reach goal: {skill[1]}")

            elif skill[0] == 'grasp':
                n_grasp_steps = skill[2]
                for _ in range(max(1, n_grasp_steps)):
                    action = skill[1]
                    self.last_gripper2_act = action[-1]
                    obs, reward, done, info = super().step2(action)

            elif skill[0] == 'wait':
                action = np.zeros(4)
                action[3] = self.last_gripper2_act
                for _ in range(skill[2]):
                    obs, reward, done, info = super().step2(action)
            else:
                raise NotImplementedError

        if skill_name =='open_drawer1':
            self.drawer1_is_open = True 
        if skill_name == 'open_drawer2':
            self.drawer2_is_open = True

        if "pick" in skill_name:
            self.free_hand = False
        if "place" in skill_name: 
            self.free_hand = True

        return obs, reward, done or (self.cam_skill_step > self.horizon), info

if __name__=="__main__":
    # from l2l.config.env.robosuite.skill_multi_stage import env_config
    # import robosuite as suite
    # import cv2
    # import matplotlib.pyplot as plt

    # from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper, UncertaintyBasedLookingWrapper
    # from stable_baselines3 import PPO

    # from l2l.config.dual.robosuite.skill_multi_stage.multi_stage_action_dual_config import encoder_model_config


    from l2l.config.env.robosuite.skill_two_arm import env_config
    import robosuite as suite
    import cv2

    from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper
    
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
