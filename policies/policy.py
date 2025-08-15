from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import numpy as np



def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1

    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def progrocess(self, x):
        p = 10
        indicator = (x.abs() >= np.exp(-p)).to(torch.float32)
        x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
        x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
        return torch.mul(x_proc1, x_proc2)

    def preprocess_grad_loss(self, grads):
        tuple_lst = []
        for i in range(len(grads)):
            tuple_lst.append(self.progrocess(grads[i]))
        return tuple(tuple_lst)

    def transfer_params(self, learner_w_grad, cI):


        self.load_state_dict(learner_w_grad.state_dict())

        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx + wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx + blen].view_as(m._parameters['bias']).clone()
                    idx += blen


    def reset_batch_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.constant(m.weight, 0.5)
                nn.init.constant(m.bias, 0)

    def get_flat_params(self):
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    '''
    copy_表示 只需要复制参数的值，不复制其梯度，梯度还是保留原来的梯度 
    即：a.copy_(b),a复制了b的数值，但是a的梯度信息没有改变'''
    def copy_flat_params(self, cI):
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx + plen].view_as(p))
            idx += plen

    def update_params_preprocess_grad(self, loss, step_size, first_order=False, params=None):
        if params is None:
            params = [param for name, param in self.named_parameters()]
        else:
            params = [param for name, param in params.items()]
        grads = torch.autograd.grad(loss, params, create_graph=not first_order)
        grads = self.preprocess_grad_loss(grads)
        updated_params = OrderedDict()
        for (name, param), param, grad in zip(self.named_parameters(), params, grads):
            updated_params[name] = param - step_size * grad

        return updated_params


    def update_params(self, loss, step_size, first_order=False, params=None):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """

        if params is None:
            params = [param for name, param in self.named_parameters()]
        else:
            params = [param for name, param in params.items()]

        grads = torch.autograd.grad(loss, params, create_graph=not first_order)
        updated_params = OrderedDict()

        for (name, param), param, grad in zip(self.named_parameters(), params, grads):
            updated_params[name] = param - step_size * grad

        return updated_params

    def reset_context(self):
        pass
