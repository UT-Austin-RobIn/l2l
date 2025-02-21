import numpy as np
import torch.nn as nn

class MLP(nn.Module):

    def __init__(
                    self, 
                    input_dim,
                    output_dim,
                    hidden_units=[32, 32],
                    normalization=None,
                    activation=nn.ReLU(),
                    output_activation=None):
        super(MLP, self).__init__()
        self.output_dim = output_dim

        layer_dims = [input_dim] + hidden_units + [output_dim]

        net_list = []
        for i in range(1, len(layer_dims)):
            net_list.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
            if i < len(layer_dims) - 1:
                if normalization is not None:
                    net_list.append(normalization(layer_dims[i]))
                net_list.append(activation)
            else:
                if output_activation is not None:
                    net_list.append(output_activation)
        
        self.net = nn.Sequential(*net_list)
    
    def output_shape(self):
        return self.output_dim
    
    def forward(self, inputs):
        return self.net(inputs)
    

