import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from policies.policy import Policy, weight_init
from mycategorical import Categorical2
import numpy as np
def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1

    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)

class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """

    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        print('Layer sizes', input_size,hidden_sizes,output_size)
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'],
                      bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

class CaviaMLPPolicy(Policy, nn.Module):
    """CAVIA network based on a multi-layer perceptron (MLP), with a
    `Normal` distribution output, with trainable standard deviation. This
    policy network can be used on tasks with continuous action spaces (eg.
    `HalfCheetahDir`).
    """

    def __init__(self, input_size, output_size, device, hidden_sizes=(), num_context_params=10,
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(CaviaMLPPolicy, self).__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1
        self.context_params = []

        layer_sizes = (input_size,) + hidden_sizes
        self.add_module('layer{0}'.format(1), nn.Linear(layer_sizes[0] + num_context_params, layer_sizes[1]))
        for i in range(2, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.num_context_params = num_context_params
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, params=None):


        if params is None:
            params = OrderedDict(self.named_parameters())


        output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
                           dim=len(input.shape) - 1)


        for i in range(1, self.num_layers):
            output = F.linear(output.float(), weight=params['layer{0}.weight'.format(i)].float(),
                              bias=params['layer{0}.bias'.format(i)].float())
            output = self.nonlinearity(output)


        mu = F.linear(output.float(), weight=params['mu.weight'].float(), bias=params['mu.bias'].float())
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

    def update_params(self, loss, step_size, first_order=False, params=None):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """


        grads = torch.autograd.grad(loss, self.context_params, create_graph=not first_order)[0]


        if not first_order:
            self.context_params = self.context_params - step_size * grads
        else:
            self.context_params = self.context_params - step_size * grads.detach()

        return OrderedDict(self.named_parameters())

    def reset_context(self):
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)



class Cavia_CategoricalMLPPolicy(Policy, nn.Module):
    def __init__(self, input_size, output_size, device, hidden_sizes=(), num_context_params=10,
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(Cavia_CategoricalMLPPolicy, self).__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity

        self.num_layers = len(hidden_sizes) + 1
        self.context_params = []

        layer_sizes = (input_size,) + hidden_sizes + (output_size,)
        self.add_module('layer{0}'.format(1), nn.Linear(layer_sizes[0] + num_context_params, layer_sizes[1]))

        for i in range(2, self.num_layers + 1):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))


        self.num_context_params = num_context_params
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

        self.apply(weight_init)

    def forward(self, input, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
                           dim=len(input.shape) - 1)


        for i in range(1, self.num_layers):
            output = F.linear(output.float(), weight=params['layer{0}.weight'.format(i)].float(),
                              bias=params['layer{0}.bias'.format(i)].float())
            output = self.nonlinearity(output)


        logits = F.linear(output.float(),
                          weight=params['layer{0}.weight'.format(self.num_layers)].float(),
                          bias=params['layer{0}.bias'.format(self.num_layers)].float())

        return Categorical2(logits=logits)





    def update_params(self, loss, step_size, first_order=False, params=None):

        grads = torch.autograd.grad(loss, self.context_params, create_graph=not first_order)[0]

        if not first_order:
            self.context_params = self.context_params - step_size * grads
        else:
            self.context_params = self.context_params - step_size * grads.detach()
        return OrderedDict(self.named_parameters())

    def reset_context(self):
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)