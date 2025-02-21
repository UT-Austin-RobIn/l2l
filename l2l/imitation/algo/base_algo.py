import torch
import torch.nn as nn
import os

class BaseAlgo(nn.Module):
    def __init__(self):
        super().__init__()
    
        self._dummy_variable = nn.Parameter(requires_grad=False)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def build_network(self):
        raise NotImplementedError
    
    def forward(self, inputs):
        raise NotImplementedError

    def compute_loss(self, inputs, outputs):
        raise NotImplementedError
    
    def post_epoch_update(self):
        pass

    def post_step_update(self):
        pass

    def get_optimizers_and_schedulers(self, **kwargs):
        raise NotImplementedError

    # def log_outputs(self, logger, losses, step, phase):
    #     # self._log_losses(logger, losses, step, phase)

    #     if phase == 'train':
    #         self.log_gradients(logger, step, phase)
    #         self.log_weights(logger, step, phase)
    
    # def _log_losses(self, logger, losses, step, phase):
    #     for name, loss in losses.items():
    #         logger.log_scalar(loss, name + '_loss', step, phase)

    # def log_gradients(self, logger, step, phase):
    #     grad_norms = list([torch.norm(p.grad.data) for p in self.parameters() if p.grad is not None])
    #     if len(grad_norms) == 0:
    #         return
    #     grad_norms = torch.stack(grad_norms)

    #     logger.log_scalar(grad_norms.mean(), 'gradients/mean_norm', step, phase)
    #     logger.log_scalar(grad_norms.abs().max(), 'gradients/max_norm', step, phase)

    # def log_weights(self, logger, step, phase):
    #     weights = list([torch.norm(p.data) for p in self.parameters() if p.grad is not None])
    #     if len(weights) == 0:
    #         return
    #     weights = torch.stack(weights)

    #     logger.log_scalar(weights.mean(), 'weights/mean_norm', step, phase)
    #     logger.log_scalar(weights.abs().max(), 'weights/max_norm', step, phase)

    def save_weights(self, epoch, path):
        path = os.path.join(path, 'weights')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "weights_ep{}.pth".format(epoch))

        model_kwargs = {'config': self.config, 'device': self.device}
        torch.save([model_kwargs, self.state_dict()], path)
    
    @staticmethod
    def load_weights(path):
        if os.path.exists(path):
            kwargs, state = torch.load(path)
            config = kwargs['config']
            device = kwargs['device']
            
            model = config.policy_config.policy_class(config)
            model.load_state_dict(state)
            return model
        else:
            print("File not found: {}".format(path))
            return False
