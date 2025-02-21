import os
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, logger):
        super().__init__()
        self.config = None
        self.logger = logger
        self.device = None
    
    def build_network(self):
        raise NotImplementedError
    
    def forward(self, inputs):
        raise NotImplementedError

    def loss(self, outputs, inputs):
        raise NotImplementedError
    
    def post_epoch_step(self):
        pass

    def log_outputs(self, outputs, inputs, losses, step, phase):
        self._log_losses(losses, step, phase)

        if phase == 'train':
            self.log_gradients(step, phase)
            self.log_weights(step, phase)

        for module in self.modules():
            if hasattr(module, '_log_outputs'):
                module._log_outputs(outputs, inputs, losses, step, phase)
    
    def _log_losses(self, losses, step, phase):
        for name, loss in losses.items():
            self.logger.log_scalar(loss, name + '_loss', step, phase)

    def log_gradients(self, step, phase):
        grad_norms = list([torch.norm(p.grad.data) for p in self.parameters() if p.grad is not None])
        if len(grad_norms) == 0:
            return
        grad_norms = torch.stack(grad_norms)

        self.logger.log_scalar(grad_norms.mean(), 'gradients/mean_norm', step, phase)
        self.logger.log_scalar(grad_norms.abs().max(), 'gradients/max_norm', step, phase)

    def log_weights(self, step, phase):
        weights = list([torch.norm(p.data) for p in self.parameters() if p.grad is not None])
        if len(weights) == 0:
            return
        weights = torch.stack(weights)

        self.logger.log_scalar(weights.mean(), 'weights/mean_norm', step, phase)
        self.logger.log_scalar(weights.abs().max(), 'weights/max_norm', step, phase)

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
            model = kwargs.model_class(**kwargs)
            model.load_state_dict(state)
            return model
        else:
            print("File not found: {}".format(path))
            return False

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation=nn.ReLU, final_activation=None, batch_norm=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
    
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, output_dim))
    
        if final_activation is not None:
            layers.append(final_activation())

        self.net = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.net(input)
    
class Discriminator(nn.Module):

    def __init__(self, config, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.config = config
        self.device = device

        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_classes = config['n_classes']

        self.net = MLP(
            input_dim = self.input_dim,
            output_dim = self.n_classes,
            hidden_dim = self.hidden_dim,
            n_layers = 2,
            activation = nn.LeakyReLU,
            final_activation = None,
            batch_norm = True,
        )
        
        self.to(self.device)

    def forward(self, input):
        out = input
        out = self.net(out)
        return out
    
    def loss(self, output, target):
        target = target.to(self.device).long()
        loss = nn.CrossEntropyLoss()(output, target)
        return loss