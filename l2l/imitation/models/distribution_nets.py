import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

class Gaussian(nn.Module):

    def __init__(self, input_size, output_size, has_time_dimension=True) -> None:
        super(Gaussian, self).__init__()
        self.input_size = input_size 
        self.output_size = output_size
        self.has_time_dimension = has_time_dimension

        self.mu = nn.Linear(input_size, output_size)
        self.sigma = nn.Linear(input_size, output_size)

    def forward(self, inputs, return_states=False):
        mean = self.mu(inputs)
        sigma = self.sigma(inputs)

        mean = 3*torch.tanh(mean)

        if not self.training:
            sigma = torch.ones_like(sigma) * 1e-6
        else:
            sigma = F.softplus(sigma) + 0.0001

        gaussian = D.Normal(loc=mean, scale=sigma)
        if self.has_time_dimension:
            gaussian = D.Independent(gaussian, 1)

        if return_states:
            return gaussian, (mean, sigma)
        return gaussian

class MDN(nn.Module):
    """A mixture density network layer"""
    def __init__(self, input_size, output_size, num_gaussians, has_time_dimension=True):
        super(MDN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_gaussians = num_gaussians
        self.has_time_dimension = has_time_dimension

        self.pi = nn.Linear(input_size, num_gaussians)
        self.sigma = nn.Linear(input_size, output_size * num_gaussians)
        self.mu = nn.Linear(input_size, output_size * num_gaussians)

    def forward(self, inputs, return_states=False):
        pi_logits = self.pi(inputs)
        means = self.mu(inputs)
        sigma = self.sigma(inputs)

        means = means.view(*means.shape[:-1], self.num_gaussians, self.output_size)
        sigma = sigma.view(*sigma.shape[:-1], self.num_gaussians, self.output_size)

        means = 3*torch.tanh(means) # multiply by 2 so that it covers sufficient standard normal range

        if not self.training:
            sigma = torch.ones_like(sigma) * 1e-6
        else:
            sigma = F.softplus(sigma) + 0.0001
        
        component_distribution = D.Normal(loc=means, scale=sigma)
        if self.has_time_dimension:
            component_distribution = D.Independent(component_distribution, 1)
        
        gmm = D.MixtureSameFamily(mixture_distribution=D.Categorical(logits=pi_logits), 
                                  component_distribution=component_distribution)
        
        if return_states:
            return gmm, (pi_logits, means, sigma)
        return gmm