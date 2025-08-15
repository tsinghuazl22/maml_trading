from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from mycategorical import Categorical2
import torch
from policies.policy import Policy, weight_init
import torch as th
'''
根据自己的理解重新写了下多层前向网络的定义:
通过add_module注册网络的时候其实可以把非线性层加上,源代码没加上,反而在前向传播的时候才加上（???），程序可读性不强
源代码这样写的原因:注册只注册线性层,因为当前向传播传入参数的时候,这个参数如何赋予到之前注册的网络当中，它这里因为只注册线性层，
而nn.Linear(input=?,weight=?,bias=?)这样赋予参数还是比较方便的

既然这样，前面注册线性层网络，就是为了能够确定参数的形式，对于有参数的网络，必须要注册
'''
































class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int,
                 last_layer_dim_pi: int = 32,
                 last_layer_dim_vf: int = 32,
                 ):
        super(FeatureExtractor, self).__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU()
        )
        self.policy_net = nn.Sequential(
            nn.Linear(128, self.latent_dim_pi),
            nn.LeakyReLU(),
            nn.Linear()
        )
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.latent_dim_vf),
            nn.LeakyReLU()
        )

    def forward(self, features: th.Tensor):
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.shared_net(features))



class CategoricalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a
    `Categorical` distribution output. This policy network can be used on tasks
    with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
    """

    def __init__(self, input_size, output_size,
                 hidden_sizes=(), nonlinearity=F.leaky_relu):
        super(CategoricalMLPPolicy, self).__init__(
            input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes + (output_size,)
        for i in range(1, self.num_layers + 1):

            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))


        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output.float(),
                              weight=params['layer{0}.weight'.format(i)].float(),
                              bias=params['layer{0}.bias'.format(i)].float())
            output = self.nonlinearity(output)

        logits = F.linear(output.float(),
                          weight=params['layer{0}.weight'.format(self.num_layers)].float(),
                          bias=params['layer{0}.bias'.format(self.num_layers)].float())



        return Categorical2(logits=logits)
