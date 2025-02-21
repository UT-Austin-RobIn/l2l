import os
from importlib.machinery import SourceFileLoader
from l2l.utils.general_utils import get_env_from_config
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from l2l.envs.robosuite.skill_multi_stage import SkillMultiStage
from l2l.envs.robosuite.skill_kitchen import SkillKitchenEnv
from l2l.envs.robosuite.skill_walled_multi_stage import SkillWalledMultiStage
from l2l.envs.robosuite.skill_two_arm import SkillTwoArm

LOG=True
WANDB_PROJECT_NAME = 'l2l-rl'
WANDB_ENTITY_NAME = 'ut-robin'

class RLTrainer:

    def __init__(self, config_path, exp_name):
        self.config_path = config_path
        self.exp_name = exp_name
        self.load_config(config_path)

        if LOG:
            self.run = wandb.init(
                project=WANDB_PROJECT_NAME,
                entity=WANDB_ENTITY_NAME,
                name=exp_name,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            )
        else:
            print("Logging is disabled")

        self.log_path = os.path.join("../experiments", self.exp_name)

        self.env = self.get_parallel_envs(self.env_config.num_envs)
        self.setup_model()

        self.evaluator = None

    def train(self, n_epochs):
        self.model.learn(
            n_epochs,
            callback=CallbackList([
                    CheckpointCallback(
                        save_freq=self.train_config.rl_save_freq,
                        save_path=os.path.join(self.log_path, 'weights'),
                    ),
                    WandbCallback(),
                ]) if LOG else None,
            )
        
        # if self.evaluator is not None:
        #     self.evaluate()
    
    def evaluate(self):
        return self.evaluator.evaluate(self.model.policy)

    def get_parallel_envs(self, num_envs):
        def make_env(seed):
            def _init():
                return get_env_from_config(self.env_config, seed)

            return _init

        return SubprocVecEnv([make_env(i) for i in range(num_envs)])

    def setup_model(self):
        self.model = self.model_config.model_class(**self.model_config.model_config,
                                                   env=self.env,
                                                   tensorboard_log=f"runs/{self.run.id}" if LOG else None,)

    def load_config(self, config_path):
        self.conf = SourceFileLoader('conf', config_path).load_module().config
        
        self.env_config = self.conf.env_config
        self.model_config = self.conf.model_config
        self.train_config = self.conf.train_config
        self.evaluator_config = self.conf.evaluator_config
        self.callback_config = self.conf.sb3_callback_config


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--exp_name", type=str, default="test_run",
                        help="unique name of the current run(include details of the architecture. eg. SimpleC2R_64x3_relu_run1)")

    args = parser.parse_args()

    trainer = RLTrainer(config_path=args.config_path, exp_name=args.exp_name)
    trainer.train(n_epochs=trainer.train_config.n_epochs)


