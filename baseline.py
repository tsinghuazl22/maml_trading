
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class critic(nn.Module):
    def __init__(self, input_size, hidden_sizes, out_size, reg_coeff=1e-5,
                 nonlinearity=F.leaky_relu):
        super(critic, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.hidden_sizes = hidden_sizes
        self.out_size = out_size
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes + (out_size,)
        for i in range(1, self.num_layers + 1):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input.observations
        for i in range(1, self.num_layers):
            output = F.linear(output.float(),
                              weight=params['layer{0}.weight'.format(i)].float(),
                              bias=params['layer{0}.bias'.format(i)].float())
            output = self.nonlinearity(output)

        values = F.linear(output.float(),
                          weight=params['layer{0}.weight'.format(self.num_layers)].float(),
                          bias=params['layer{0}.bias'.format(self.num_layers)].float())
        return values

    def update_critic(self, critic_loss, step_size, first_order=False, params=None):
        if params is None:
            params = [param for name, param in self.named_parameters()]
        else:
            params = [param for name, param in params.items()]

        grads = torch.autograd.grad(critic_loss, params, create_graph=not first_order)
        updated_params = OrderedDict()

        for (name, param), param, grad in zip(self.named_parameters(), params, grads):
            updated_params[name] = param - step_size * grad

        return updated_params


class CriticBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CriticBaseline, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ValueNetworkBaseLine(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlinearity=F.leaky_relu):
        super(ValueNetworkBaseLine, self).__init__()
        self.input_size = input_size
        self.num_layers = len(hidden_size) + 1
        self.nonlinearity = nonlinearity
        self.num_epochs = 100
        layer_sizes = (input_size,) + hidden_size + (output_size,)
        for i in range(1, self.num_layers + 1):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.apply(weight_init)

    def forward(self, input):
        params = OrderedDict(self.named_parameters())
        output = input.observations
        for i in range(1, self.num_layers):
            output = F.linear(output.float(),
                              weight=params['layer{0}.weight'.format(i)].float(),
                              bias=params['layer{0}.bias'.format(i)].float())
            output = self.nonlinearity(output)

        values = F.linear(output.float(),
                          weight=params['layer{0}.weight'.format(self.num_layers)].float(),
                          bias=params['layer{0}.bias'.format(self.num_layers)].float())
        return values

    def fit(self, episodes):
        returns = episodes.returns
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        for _ in range(self.num_epochs):
            values = self.forward(episodes).squeeze()
            loss = criterion(values, returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class ValueNetworkBaseLine_timevary(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlinearity=F.leaky_relu,
                 num_epochs=50, lr=0.1):
        super(ValueNetworkBaseLine_timevary, self).__init__()
        self.lr = lr
        self.num_epochs = num_epochs
        self.input_size = input_size
        self.num_layers = len(hidden_size) + 1
        self.nonlinearity = nonlinearity
        self.feature_size = 2 * self.input_size + 4

        layer_sizes = (self.feature_size,) + hidden_size + (output_size,)
        for i in range(1, self.num_layers + 1):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.apply(weight_init)

    def forward(self, input):
        params = OrderedDict(self.named_parameters())
        output = self._feature(input)
        for i in range(1, self.num_layers):
            output = F.linear(output.float(),
                              weight=params['layer{0}.weight'.format(i)].float(),
                              bias=params['layer{0}.bias'.format(i)].float())
            output = self.nonlinearity(output)

        values = F.linear(output.float(),
                          weight=params['layer{0}.weight'.format(self.num_layers)].float(),
                          bias=params['layer{0}.bias'.format(self.num_layers)].float())
        return values

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2)
        observations = episodes.observations * ones
        cum_sum = torch.cumsum(ones, dim=0) * ones
        al = cum_sum / 100.00
        return torch.cat([observations, observations ** 2, al, al ** 2, al ** 3, ones], dim=2)

    def fit(self, episodes):
        returns = episodes.returns.view(-1, 1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.num_epochs):
            values = self.forward(episodes).view(-1, 1)
            loss = criterion(values, returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class LinearFeatureBaseline(nn.Module):
    """改进的线性特征基线，解决矩阵不可逆问题"""

    def __init__(self, input_size, reg_coeff=1e-5, max_reg_increase=1000):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.max_reg_increase = max_reg_increase
        self.linear = nn.Linear(self.feature_size, 1, bias=False)
        self.linear.weight.data.zero_()

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2)
        observations = episodes.observations * ones
        cum_sum = torch.cumsum(ones, dim=0) * ones
        al = cum_sum / 100.00
        observations = (observations - observations.mean()) / (observations.std() + 1e-8)
        return torch.cat([observations, observations ** 2, al, al ** 2, al ** 3, ones], dim=2)

    def fit(self, episodes):
        featmat = self._feature(episodes).view(-1, self.feature_size)
        returns = episodes.returns.view(-1, 1)
        xtx = torch.matmul(featmat.t(), featmat)
        cond = torch.linalg.cond(xtx) if xtx.numel() > 0 else 0
        reg_coeff = self._reg_coeff
        if cond > 1e6:
            reg_coeff = max(reg_coeff, 1e-3)

        eye = torch.eye(self.feature_size, dtype=torch.float32,
                        device=self.linear.weight.device)
        success = False
        for _ in range(10):
            try:
                xtx_reg = xtx + reg_coeff * eye
                xtx_reg = (xtx_reg + xtx_reg.t()) / 2

                try:
                    L = torch.linalg.cholesky(xtx_reg)
                    coeffs = torch.cholesky_solve(torch.matmul(featmat.t(), returns), L)
                except:
                    U, S, Vh = torch.linalg.svd(xtx_reg)
                    S_inv = torch.diag(1.0 / S)
                    xtx_inv = Vh.t() @ S_inv @ U.t()
                    coeffs = xtx_inv @ torch.matmul(featmat.t(), returns)

                success = True
                break
            except RuntimeError:
                reg_coeff *= 2
                if reg_coeff > self._reg_coeff * self.max_reg_increase:
                    break

        if not success:
            raise RuntimeError(
                f'无法求解`LinearFeatureBaseline`中的正规方程。矩阵X^T*X（X为设计矩阵）不是满秩的，'
                f'即使使用最大正则化系数: {reg_coeff}。条件数: {cond}'
            )

        self.linear.weight.data = coeffs.data.t()
        self.used_reg_coeff = reg_coeff

    def fit_offpolicy(self, episodes):
        featmat = self._feature(episodes).view(-1, self.feature_size)
        returns = episodes.returns.view(-1, 1)
        weights = episodes.weight.view(-1, 1)

        reg_coeff = self._reg_coeff
        eye = torch.eye(self.feature_size, dtype=torch.float32,
                        device=self.linear.weight.device)

        for _ in range(10):
            try:
                xtx = torch.matmul(featmat.t() * weights, featmat)
                xty = torch.matmul(featmat.t(), weights * returns)
                coeffs = torch.linalg.lstsq(xty, xtx + reg_coeff * eye).solution
                break
            except RuntimeError:
                reg_coeff *= 2
                if reg_coeff > self._reg_coeff * self.max_reg_increase:
                    raise RuntimeError(
                        f'无法求解离线策略的正规方程，最大正则化系数: {reg_coeff}'
                    )

        self.linear.weight.data = coeffs.data.t()

    def forward(self, episodes):
        features = self._feature(episodes)
        return self.linear(features)


if __name__ == "__main__":
    net = ValueNetworkBaseLine(input_size=23, hidden_size=(128, 64, 32), output_size=1)
    input = torch.randn(1, 23)
    out = net(input)
    print(out)
