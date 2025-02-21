import numpy as np
from tqdm import tqdm
import robosuite as suite

class RobosuiteEvaluator:

    def __init__(self, eval_config):
        self.env = suite.make(**eval_config.env_config)

        self.n_rollouts = eval_config.n_rollouts
        self.max_steps = eval_config.max_steps

    def evaluate(self, policy):
        print("\nEvaluating policy...")

        success = np.zeros(self.n_rollouts)
        timsteps = np.full((self.n_rollouts,), self.max_steps)
        for n in tqdm(range(self.n_rollouts)):
            obs = self.env.reset()
            policy.reset()

            for steps in range(self.max_steps):
                action = policy.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                
                # self.env.render()
                if done:
                    success[n] = 1
                    timsteps[n] = steps
                    break        
        
        eval_info = {
            'success_rate': np.mean(success),
            'mean_timesteps': np.mean(timsteps),
        }

        print(f"Success rate: {eval_info['success_rate']}")
        print(f"Mean timesteps: {eval_info['mean_timesteps']}\n")

        return eval_info 