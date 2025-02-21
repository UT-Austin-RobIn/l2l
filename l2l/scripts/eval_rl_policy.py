import os
import time
import cv2
import copy
import numpy as np
import imageio

from stable_baselines3 import PPO
import robosuite as suite
from l2l.utils.general_utils import get_env_from_config

from l2l.envs.robosuite.camera_color_pnp import CameraOracleColorPnP  # necessary otherwise sb3 doesn't find environment
from importlib.machinery import SourceFileLoader


def do_single_rollout(
            env,
            policy,
            max_steps,
            n_stages,
            deterministic,
            render
        ):

    obs, info = env.reset()
    stage_success = [False for _ in range(n_stages)]
    cur_stage = env.stage_id
    while env.total_env_steps_taken < max_steps:
        action, states = policy.predict(obs, deterministic=deterministic)

        obs, _, _, _, _ = env.step(action)

        if render:
            img = cv2.resize(obs['activeview_image'], (252, 252))/ 255
            cv2.imshow('obs', img)
            cv2.waitKey(1)
            env.render()

        if env.stage_id > cur_stage:
            print(f'stage {i} env_steps', env.total_env_steps_taken)
            stage_success[cur_stage] = True
            cur_stage = env.stage_id

        if env.unwrapped._check_success():
            print('env steps', env.total_env_steps_taken)
            print("Success")
            stage_success[-1] = True
            return stage_success
        if env.unwrapped._check_failure():
            print('env steps', env.total_env_steps_taken)
            print("Failure")
            return stage_success
    
    print('env steps', env.total_env_steps_taken)
    print("Timeout")
    return stage_success


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, help="max steps while rolling out policy")
    parser.add_argument("--ckpt", type=str, help="path to model checkpoint")
    parser.add_argument("--n_rollouts", default=50, type=int, help="no. of trajectories to save")
    parser.add_argument("--render", action="store_true", help="render rollout")
    parser.add_argument("--deterministic", action="store_true", help="use deterministic policy")
    # parser.add_argument("--save_vid", action="store_true", help="create a video of rollout")
    args = parser.parse_args()

    print("Deterministic: ", args.deterministic)
    
    import robosuite as suite
    if args.env == 'kitchen':
        from l2l.config.env.robosuite.skill_kitchen import env_config
        max_steps = 700
        n_stages = 3
        config_path = "/home/shivin/Desktop/learning2look/l2l/l2l/config/rl/robosuite/skill_kitchen/kitchen_full_rl_baseline.py"

    elif args.env == 'walled':
        from l2l.config.env.robosuite.skill_walled_multi_stage import env_config
        max_steps = 500
        n_stages = 2
        config_path = "/home/shivin/Desktop/learning2look/l2l/l2l/config/rl/robosuite/skill_walled_multi_stage/walled_multi_stage_full_rl_baseline.py"

    elif args.env == 'two_arm':
        from l2l.config.env.robosuite.skill_two_arm import env_config
        max_steps = 650
        n_stages = 1
        config_path = "/home/shivin/Desktop/learning2look/l2l/l2l/config/rl/robosuite/skill_two_arm/two_arm_full_rl_baseline.py"

    else:
        raise NotImplementedError
    
    # initialize the task
    # Load environment
    conf = SourceFileLoader('conf', config_path).load_module().config

    env = get_env_from_config(conf.env_config)
    obs, info = env.reset()

    # load model
    policy = PPO.load(args.ckpt, env=env)


    total_stage_attempts = [0 for _ in range(n_stages)]
    total_stage_success = [0 for _ in range(n_stages)]
    for i in range(args.n_rollouts):
        stage_success = do_single_rollout(
                            env=env,
                            policy=policy,
                            max_steps=max_steps,
                            n_stages=n_stages,
                            deterministic=args.deterministic,
                            render=args.render,
                        )
        
        for j, success in enumerate(stage_success):
            if j == 0 or stage_success[j-1]:
                total_stage_attempts[j] += 1
            if success:
                total_stage_success[j] += 1
        print(f"Rollout {i}: {stage_success}")
        print()

    for i in range(n_stages):
        print(f"Stage {i}: {total_stage_success[i]}/{total_stage_attempts[i]}")