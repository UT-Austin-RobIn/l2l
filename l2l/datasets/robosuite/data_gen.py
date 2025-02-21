import robosuite as suite
import numpy as np
from l2l.utils.general_utils import recursive_map_dict
from l2l.datasets import BaseDataGenerator
import cv2, copy

# domain = 'ms'
# domain = 'kitchen'
# domain = 'two_arm'
domain = 'color_pnp_language'

def robosuite_generate_trajectory(env, action_generator, episode_length=None, noise=0, render=False):

    if domain == 'kitchen':
        trajectory_dict = {
                'obs': {
                    # 'agentview_image': [],
                    # 'sideview_image': [],
                    # 'activeview_image': [],
                    # 'reference_image': [],
                    'robot0_eye_in_hand_image': [],
                    'robot0_eef_pos': [],
                    'robot0_eef_quat': [],
                    'robot0_gripper_qpos': [],
                    # 'reference_cube_color': [],
                    'privileged_info': [],
                    'gt_privileged_info': [],  # we can reuse this to generate other forms of privileged info
                    'object-state': [],
                    'butter_melt_status': [],
                    'meatball_cook_status': [],
                },
                'actions': [],
                'dones': [],
                'rewards': []
            }
    elif domain == 'ms':
        trajectory_dict = {
                'obs': {
                    # 'agentview_image': [],
                    # 'sideview_image': [],
                    # 'activeview_image': [],
                    # 'reference_image': [],
                    'robot0_eye_in_hand_image': [],
                    'robot0_eef_pos': [],
                    'robot0_eef_quat': [],
                    'robot0_gripper_qpos': [],
                    # 'reference_cube_color': [],
                    'privileged_info': [],
                    'gt_privileged_info': [],  # we can reuse this to generate other forms of privileged info
                    'object-state': [],
                },
                'actions': [],
                'dones': [],
                'rewards': []
            }
    elif domain == 'two_arm':
        trajectory_dict = {
                'obs': {
                    'robot0_eye_in_hand_image': [],
                    'robot0_eef_pos': [],
                    'robot0_eef_quat': [],
                    'robot0_gripper_qpos': [],
                    'robot1_eef_pos': [],
                    'robot1_gripper_qpos': [],
                    'privileged_info': [],
                    'object-state': [],
                },
                'actions': [],
                'dones': [],
                'rewards': []
            }
    elif domain == 'color_pnp_language':
        trajectory_dict = {
                'obs': {
                    'robot0_eye_in_hand_image': [],
                    'robot0_eef_pos': [],
                    'robot0_eef_quat': [],
                    'robot0_gripper_qpos': [],
                    'privileged_info': [],
                    'gt_privileged_info': [],
                    'object-state': [],
                },
                'actions': [],
                'dones': [],
                'rewards': []
            }
    else:
        raise ValueError('Invalid domain')
    action_meta = []
    new_api = False

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
        new_api = True

    for i in range(episode_length):
        for k in trajectory_dict['obs'].keys():
            if 'image' in k:
                trajectory_dict['obs'][k].append(obs[k])
            else:
                trajectory_dict['obs'][k].append(obs[k])
        
        action_info = action_generator(copy.deepcopy(obs), noise = noise)
        # print(action_info['action'][-1])
        if new_api:
            obs, reward, done, trunc, info = env.step(action_info['action'])
        else:
            obs, reward, done, info = env.step(action_info['action'])

        # if action_info['meta'].get('policy_id', 1) != 1:
        #     action_info['action'] = np.zeros(4) 
        trajectory_dict['actions'].append(action_info['action'])
        trajectory_dict['dones'].append(done)
        trajectory_dict['rewards'].append(reward)

        action_meta.append(action_info['meta'])

        if render:
            # cv2.imshow('obs', np.concatenate((obs['activeview_image'], obs['reference_image']), axis=0)/255)
            # cv2.imshow('obs', np.concatenate((obs['agentview_image'], obs['sideview_image']), axis=0).astype(float)/255)
            # cv2.waitKey(1)
            env.render()

        try:
            is_success = env._check_success()
        except:
            is_success = env.unwrapped._check_success()
        if done != is_success:
            print('WARNING: done is not equal to task success')
        if done:
            # print("Success")
            break

    return {
                'trajectory': recursive_map_dict(np.array, trajectory_dict),
                'meta': action_meta,
            }

class RobosuiteDataGenerator(BaseDataGenerator):

    def __init__(self, env_config) -> None:
        self.env_config = env_config
        self.env = suite.make(**env_config)

        class ActionGenerator:

            def __init__(self, env):
                self.env = env
            
            def __call__(self, obs, *args, **kwargs):
                return {'action': self.env.get_optimal_action(obs, noise=kwargs['noise']), 'meta': {}}
            
        self.action_generator = ActionGenerator(self.env)

    def generate_trajectory(self, episode_length=None, noise=0, render=False):
        return robosuite_generate_trajectory(self.env, self.action_generator, episode_length, noise, render)['trajectory']

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_traj", type=int, help="no. of trajectories to save")
    parser.add_argument("--episode_length", default=None, type=int, help="max episode length")
    parser.add_argument("--noise", default=0, type=float, help="noise in actions")
    parser.add_argument("--save_path", default=None, type=str, help="path to save file")
    parser.add_argument("--render", action="store_true", help="pass flag to render environment while data generation")
    args = parser.parse_args()

    if domain == 'ms':
        from l2l.config.env.robosuite.skill_multi_stage import env_config
    elif domain == 'kitchen':
        from l2l.config.env.robosuite.skill_kitchen import env_config
    elif domain == 'two_arm':
        from l2l.config.env.robosuite.skill_two_arm import env_config
    elif domain == 'color_pnp_language':
        from l2l.config.env.robosuite.skill_color_pnp_language import env_config
    else:
        raise ValueError('Invalid domain')

    obj = RobosuiteDataGenerator(env_config=env_config)
    obj.make_dataset(
        num_traj = args.num_traj,
        episode_length = args.episode_length,
        noise = args.noise,
        save_path = args.save_path,
        render = args.render,
        env_args = env_config,
    )