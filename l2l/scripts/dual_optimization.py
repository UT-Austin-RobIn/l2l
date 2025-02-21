import os
from importlib.machinery import SourceFileLoader
from l2l.utils.general_utils import get_env_from_config
import wandb
import pickle
import torch
import pickle
from torch.utils.data import DataLoader
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList, CheckpointCallback

from l2l.envs.robosuite.skill_kitchen import SkillKitchenEnv
from l2l.envs.robosuite.skill_walled_multi_stage import SkillWalledMultiStage
from l2l.envs.robosuite.skill_two_arm import SkillTwoArm

from l2l.imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from l2l.utils.general_utils import AttrDict
from l2l.utils.buffer_dataset import BufferDataset
from l2l.utils.sb3_utils import StoreRolloutBufferCallback, StoreReplayBufferCallback

DEVICE='cuda'

LOG = True
WANDB_PROJECT_NAME = 'l2l-rl'
WANDB_ENTITY_NAME = 'ut-robin'

class DualTrainer:

    def __init__(self, config_path, exp_name, resume_rl_ckpt=None):
        self.config_path = config_path
        self.exp_name = exp_name
        self.resume_rl_ckpt = resume_rl_ckpt
        self.load_config(config_path)

        if LOG:
            self.run = wandb.init(
                project=WANDB_PROJECT_NAME,
                entity=WANDB_ENTITY_NAME,
                name=exp_name,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            )

        self.log_path = os.path.join("../experiments", self.exp_name)

        self.env = self.get_parallel_envs(self.env_config.num_envs) if self.env_config.num_envs > 1 else get_env_from_config(self.env_config)
        self.setup_models()

        self.evaluator = None

    def train(self, n_epochs):

        # fill replay buffer
        print("Filling replay buffer")
        self.rl_model.env.env_method('set_encoder', self.encoder)
        self.rl_model.learn(
                self.train_config.warmup_rl_steps, callback=self.store_buffer_callback(self.dataset)
            ) 

        for epoch in range(int(n_epochs)):
            
            print("Training epoch", epoch)

            print("Training encoder...")
            dataloader = DataLoader(self.dataset, batch_size=self.train_config.encoder_batch_size, shuffle=True)
            
            for encoder_epoch in range(self.train_config.encoder_warmup_epochs if epoch==0 else self.train_config.encoder_epochs_per_epoch):
                epoch_info = AttrDict(
                    losses=AttrDict(),
                )
                for idx, batch in enumerate(dataloader):
                    step_info = self.train_on_batch(batch)

                    for k in step_info.losses:
                        if k not in epoch_info.losses:
                            epoch_info.losses[k] = 0
                        epoch_info.losses[k] += step_info.losses[k].item() / len(dataloader)

                print(f'\nepoch {epoch}, encoder_epoch {encoder_epoch}')
                print('Losses')
                for loss in epoch_info.losses.keys():
                    print(f'\t{loss}: {epoch_info.losses[loss]}', end='\n\n')

            save_path = os.path.join(self.log_path, 'encoder_weights')
            os.makedirs(save_path, exist_ok=True)
            print(f"==> Saving checkpoint at epoch {epoch}")
            self.encoder.save_weights(epoch, save_path)

            self.rl_model.env.env_method('set_encoder', self.encoder)

            self.rl_model.learn(
                self.train_config.rl_steps_per_epoch,
                reset_num_timesteps=False,
                callback=CallbackList([
                    self.store_buffer_callback(self.dataset),
                    CheckpointCallback(
                        save_freq=self.train_config.rl_save_freq,
                        save_path=os.path.join(self.log_path, f'epoch_{epoch}', 'weights'),
                    ),
                    WandbCallback(),
                ]) if LOG else None,
            )

            if self.evaluator is not None:
                self.evaluate()

            # save dataset
            # with open(os.path.join(self.log_path, f'dataset_{len(self.dataset)}.pkl'), 'wb') as f:
            #     pickle.dump(self.dataset, f)

    def train_on_batch(self, batch):
        batch = recursive_dict_list_tuple_apply(batch, {torch.Tensor: lambda x: x.to(DEVICE).float()})
            
        for i in range(len(self.encoder_optimizers)):
            self.encoder_optimizers[i].zero_grad()
        
        losses = self.encoder.compute_loss(batch)
        losses.total.backward()

        if self.encoder_model_config.get('grad_clip', False):
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)

        for i in range(len(self.encoder_optimizers)):
            self.encoder_optimizers[i].step()

            if self.encoder_lr_schedulers[i] is not None:
                self.encoder_lr_schedulers[i].step()

        step_info = AttrDict(
            losses=losses,
        )
        return step_info

    def evaluate(self):
        return self.evaluator.evaluate(self.model.policy)

    def get_parallel_envs(self, num_envs):
        def make_env(seed):
            def _init():
                return get_env_from_config(self.env_config, seed)

            return _init

        return SubprocVecEnv([make_env(i) for i in range(num_envs)])

    def setup_rl_model(self):
        if self.resume_rl_ckpt:
            self.rl_model = self.rl_model_config.model_class.load(self.resume_rl_ckpt,
                                                    env=self.env,)
        else:
            self.rl_model = self.rl_model_config.model_class(**self.rl_model_config.model_kwargs,
                                                    env=self.env,
                                                    tensorboard_log=f"runs/{self.run.id}" if LOG else None,)
        if self.rl_model.__class__.__name__ == 'PPO':
            self.store_buffer_callback = StoreRolloutBufferCallback
        else:
            raise NotImplementedError

    def setup_models(self):
        self.setup_rl_model()
        
        if self.resume_rl_ckpt:
            encoder_path = get_encoder_path(self.resume_rl_ckpt)
            self.encoder = self.encoder_model_config.model_class.load_weights(encoder_path).float().to(DEVICE)
        elif self.train_config.get('load_encoder_file', None) is not None:
            self.encoder = self.encoder_model_config.model_class.load_weights(self.train_config.load_encoder_file).float().to(DEVICE)
        else:
            self.encoder = self.encoder_model_config.model_class(self.encoder_model_config).float().to(DEVICE)
        self.encoder_optimizers, self.encoder_lr_schedulers = self.encoder.get_optimizers_and_schedulers()

        if self.train_config.get('load_dataset_file', None) is not None:
            print(f"Loading dataset {self.train_config.load_dataset_file}")
            self.dataset = pickle.load(open(self.train_config.load_dataset_file, 'rb'))
        else:
            self.dataset = BufferDataset(device=DEVICE, max_len=int(5e4))

        print(self.encoder)

    def load_config(self, config_path):
        self.conf = SourceFileLoader('conf', config_path).load_module().config
        
        self.env_config = self.conf.env_config
        self.rl_model_config = self.conf.rl_model_config
        self.encoder_model_config = self.conf.encoder_model_config
        self.train_config = self.conf.train_config
        self.evaluator_config = self.conf.evaluator_config
        self.callback_config = self.conf.sb3_callback_config

def get_encoder_path(rl_path):
    rl_ckpt = rl_path.split('/')[-3].split('_')[-1]

    encoder_path = "/".join(rl_path.split('/')[:-3]) + f"/encoder_weights/weights/weights_ep{rl_ckpt}.pth"
    return encoder_path

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--exp_name", type=str, default="test_run",
                        help="unique name of the current run")
    parser.add_argument("--resume_rl_ckpt", type=str, default=None,
                        help="if provided, continues training from this ckpt")
    args = parser.parse_args()

    trainer = DualTrainer(config_path=args.config, exp_name=args.exp_name, resume_rl_ckpt=args.resume_rl_ckpt)
    trainer.train(n_epochs=trainer.train_config.n_epochs)


