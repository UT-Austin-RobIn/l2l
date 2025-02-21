import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from l2l.imitation.utils.general_utils import AttrDict
from l2l.imitation.utils.obs_utils import process_obs_dict
from l2l.imitation.utils.log_utils import log_value_in_dict, WandBLogger
from l2l.imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from l2l.imitation.utils.file_utils import get_all_obs_keys_from_config, get_shape_metadata_from_dataset, get_obs_key_to_modality_from_config
from importlib.machinery import SourceFileLoader

DEVICE = 'cuda'

LOG = True
WANDB_PROJECT_NAME = 'imitation'
WANDB_ENTITY_NAME = 'ut-robin'

class Trainer:

    def __init__(self, config_path, exp_name):
        self.config_path = config_path
        self.exp_name = exp_name
        self.load_config(config_path)
        
        self.create_log()
        self.setup_data()
        self.setup_model()

        self.evaluator = None
        if self.evaluator_config:
            self.evaluator = self.evaluator_config.evaluator(self.evaluator_config)

    def train(self, n_epochs):
        self.model.train()
        for epoch in range(n_epochs):
            epoch_info = self.train_epoch(epoch)
            self.model.post_epoch_update()
            
            if epoch % self.train_config.log_every_n_epochs == 0 :
                losses = epoch_info.losses
                if self.logger is not None:
                    self.logger.log_scalar_dict(losses, step=epoch, phase='train/losses')
                    self.logger.log_scalar(epoch_info.gradient_norm, 'gradients/mean_norm', epoch, 'train')
                    self.logger.log_scalar(epoch_info.weight_norm, 'weights/mean_norm', epoch, 'train')

                print(f'\nepoch {epoch}')
                print('Losses')
                for loss in losses.keys():
                    print(f'\t{loss}: {losses[loss]}', end='\n\n') 

            if self.train_config.val_every_n_epochs > 0 and epoch > 0 and epoch % self.train_config.val_every_n_epochs == 0:
                val_info = self.validate()
                if self.logger is not None:
                    self.logger.log_scalar_dict(val_info, step=epoch, phase='val')
                
            if self.train_config.eval_every_n_epochs > 0 and epoch > 0 and epoch % self.train_config.eval_every_n_epochs == 0:
                eval_info = self.evaluate()
                if self.logger is not None:
                    self.logger.log_scalar_dict(eval_info, step=epoch, phase='eval')

            if self.train_config.save_every_n_epochs > 0 and epoch > 0 and epoch % self.train_config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)

    def train_epoch(self, epoch):
        
        data_loader_iter = iter(self.train_loader)
        epoch_info = AttrDict(
            losses=AttrDict(),
            gradient_norm=0,
            weight_norm=0
        )
        for batch_idx in tqdm(range(self.train_config.epoch_every_n_steps)):

            try:
                batch = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(self.train_loader)
                batch = next(data_loader_iter)

            batch['obs'] = process_obs_dict(batch['obs'], self.obs_key_to_modality)
            batch = recursive_dict_list_tuple_apply(batch, {torch.Tensor: lambda x: x.to(DEVICE).float()})
            
            # batchnorm doesn't work with batch size 1 
            if batch['actions'].shape[0] == 1:
                continue
            
            for i in range(len(self.optimizers)):
                self.optimizers[i].zero_grad()

            losses = self.model.compute_loss(batch)
            losses.total.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            for i in range(len(self.optimizers)):
                self.optimizers[i].step()
                
                if self.lr_schedulers[i] is not None:
                    self.lr_schedulers[i].step()

            self.model.post_step_update()

            # logging
            for k in losses:
                if k not in epoch_info.losses:
                    epoch_info.losses[k] = 0
                epoch_info.losses[k] += float(losses[k].item())/self.train_config.epoch_every_n_steps

            epoch_info.gradient_norm += torch.mean(torch.stack([torch.norm(p.grad.data) for p in self.model.parameters() if p.grad is not None]))/self.train_config.epoch_every_n_steps
            epoch_info.weight_norm += torch.mean(torch.stack([torch.norm(p.data) for p in self.model.parameters() if p.grad is not None]))/self.train_config.epoch_every_n_steps

        return epoch_info

    def validate(self):
        print("\nValidating policy...")
        self.model.eval()
        with torch.no_grad():
            val_loss = AttrDict()

            for batch in tqdm(self.val_loader):
                batch['obs'] = process_obs_dict(batch['obs'], self.obs_key_to_modality)
                batch = recursive_dict_list_tuple_apply(batch, {torch.Tensor: lambda x: x.to(DEVICE).float()})

                losses = self.model.compute_loss(batch)

            # logging
            for k in losses:
                if k not in val_loss:
                    val_loss[k] = 0
                val_loss[k] += float(losses[k].item())/len(self.val_loader)

            print('#'*20, '\nValidation Losses')
            for k in val_loss.keys():
                print(f'\t{k}: {val_loss[k]}', end='\n\n') 
            print('#'*20)

        self.model.train()
        return val_loss

    def evaluate(self):
        return self.evaluator.evaluate(self.model)

    def create_log(self):
        self.logger = None
        self.log_path = None

        if LOG:
            self.log_path = os.path.join(self.train_config.output_dir, self.exp_name)
            # TODO: create weights directory and fix the model save path
            os.makedirs(self.log_path, exist_ok=True)

            self.logger = WandBLogger(self.exp_name, WANDB_PROJECT_NAME, WANDB_ENTITY_NAME, self.log_path, self.conf)

    def save_checkpoint(self, epoch):
        if self.log_path is not None:
            print(f"==> Saving checkpoint at epoch {epoch}")
            self.model.save_weights(epoch, self.log_path)

    def load_config(self, config_path):
        self.conf = SourceFileLoader('conf', config_path).load_module().config
        
        self.train_config = self.conf.train_config
        self.data_config = self.conf.data_config
        self.policy_config = self.conf.policy_config
        self.observation_config = self.conf.observation_config

        self.obs_keys = get_all_obs_keys_from_config(self.observation_config)
        self.obs_key_to_modality = get_obs_key_to_modality_from_config(self.observation_config)
        self.shape_meta = get_shape_metadata_from_dataset(self.data_config.data[0], all_obs_keys=self.obs_keys, obs_key_to_modality=self.obs_key_to_modality)
        
        self.evaluator_config = self.conf.evaluator_config if 'evaluator_config' in self.conf else None

    def setup_data(self):
        self.train_dataset = self.data_config.dataset_class(
                                data_paths=self.data_config.data,
                                obs_keys_to_modality=self.obs_key_to_modality,
                                obs_keys_to_normalize=self.observation_config.obs_keys_to_normalize,
                                split='train',
                                **self.data_config.dataset_kwargs)
        self.normalization_stats = self.train_dataset.get_normalization_stats()

        if self.train_config.val_every_n_epochs > 0:
            self.val_dataset = self.data_config.dataset_class(
                                data_paths=self.data_config.data,
                                obs_keys_to_modality=self.obs_key_to_modality,
                                obs_keys_to_normalize=self.observation_config.obs_keys_to_normalize,
                                split='val',
                                **self.data_config.dataset_kwargs)
            
        self.setup_dataloader()
    
    def setup_dataloader(self):
        self.train_loader = DataLoader(
                                self.train_dataset, 
                                batch_size=self.train_config.batch_size,
                                shuffle=True,
                                num_workers=self.data_config.num_workers)
        
        if self.train_config.val_every_n_epochs > 0:
            self.val_loader = DataLoader(
                                self.train_dataset, 
                                batch_size=self.train_config.batch_size,
                                shuffle=True,
                                num_workers=self.data_config.num_workers)

    def setup_model(self):
        model_config = AttrDict(
            policy_config=self.policy_config,
            observation_config=self.observation_config,
            keys_to_shapes=self.shape_meta,
            keys_to_modality=self.obs_key_to_modality,
            normalization_stats=self.normalization_stats
        )
        self.model = self.policy_config.policy_class(model_config)
        self.model.to(DEVICE)
        self.model.train()
        print(self.model)

        self.optimizers, self.lr_schedulers = self.model.get_optimizers_and_schedulers(
            num_epochs=self.train_config.num_epochs,
            epoch_every_n_steps=self.train_config.epoch_every_n_steps
        )
        assert len(self.optimizers) == len(self.lr_schedulers), 'Number of optimizers and learning rate schedulers should be same'

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to the config file")
    parser.add_argument("--exp_name", type=str, help="unique name of the current run(include details of the architecture. eg. SimpleC2R_64x3_relu_run1)")
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config, exp_name=args.exp_name)
    trainer.train(n_epochs=trainer.train_config.num_epochs)
