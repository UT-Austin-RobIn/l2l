import cv2
from l2l.utils.general_utils import get_env_from_config


def rollout(env, policy, max_steps=1000, render=False):
    inputs = env.reset()[0]

    reward_total = 0
    for i in range(max_steps):
        # inputs = recursive_map_dict(lambda x: torch.tensor(x, device='cuda')[None], inputs)
        act = policy.get_action(inputs)
        inputs, reward, done, trunc, _ = env.step(act)
        # print(act, reward)
        
        reward_total += reward
        if done or trunc:
            print(i, max_steps)
            break
        
        if render:
            env.render()
        # cv2.imshow('obs', cv2.cvtColor(env.render(), cv2.COLOR_BGR2RGB))
        # cv2.waitKey(30)
    print(reward_total)
    print()

class RobosuiteRolloutEvaluator:

    def __init__(self, eval_config, logger, *args, **kwargs) -> None:
        self.config = eval_config
        self.logger = logger
        self.env = get_env_from_config(self.config.env_config)
        self.count = 0

    def evaluate(self, policy, step=None):
        if 'policy_wrapper' in self.config and self.config.policy_wrapper is not None:
            policy = self.config.policy_wrapper(policy, deterministic=True)

        for i in range(self.config.n_rollouts):
            print(f'Rollout {i}')
            rollout(self.env, policy, max_steps=1000, render=self.config.render)
