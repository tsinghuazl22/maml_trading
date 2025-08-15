import time
import torch
import torch.nn as nn

from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from rl_utils.optimization import conjugate_gradient
from rl_utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)
import numpy as np
from envs.subproc_vec_env import SubprocVecEnv
from envs.trading_env import TradingEnv
import multiprocessing as mp
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import threading
from episode import BatchEpisodes,BatchEpisodes2
import scipy.stats as st
import os
import utils
import copy
def get_returns(episodes_per_task):
    returns = []
    for task_idx in range(len(episodes_per_task)):
        curr_returns = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)
            curr_returns.append(ret)
        returns.append(torch.stack(curr_returns, dim=1))
    returns = torch.stack(returns)
    returns = returns.reshape((-1, returns.shape[-1]))
    return returns

def total_rewards(episodes_per_task, interval=False):
    returns = get_returns(episodes_per_task).cpu().numpy()
    mean = np.mean(returns, axis=0)
    conf_int = st.t.interval(0.95, len(mean) - 1, loc=mean, scale=st.sem(returns, axis=0))
    conf_int = mean - conf_int
    if interval:
        return mean, conf_int[0]
    else:
        return mean

class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes
    (before and after the one-step adaptation), compute the inner loss, compute
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan,
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan,
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """

    def __init__(self, sampler, policy, baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)

    def inner_loss_l2_reg(self, episodes, params=None):
        """l2 正则化的损失函数
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)
        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)
        reg_coeff = 0.0001
        reg_loss = 0.0
        if params is None:
            for name, param in self.policy.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, 2) ** 2
        else:
            for name, param in params.items():
                if 'weight' in name:
                    reg_loss += torch.norm(param, 2) ** 2
        loss += reg_coeff * reg_loss



        return loss

    def inner_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner
        loss is REINFORCE with baseline [2], computed on advantages estimated
        with Generalized Advantage Estimation (GAE, [3]).
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)
        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)
        return loss
    def inner_loss_offpolicy(self, args, episodes, params=None, old_pi=None):
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)
        pi = self.policy(episodes.observations, params=params)
        if old_pi is None:
            old_pi = detach_distribution(pi)
        log_ratio = (pi.log_prob(episodes.actions) - old_pi.log_prob(episodes.actions))
        if log_ratio.dim() > 2:
            log_ratio = torch.sum(log_ratio, dim=2)
        ratio = torch.exp(log_ratio)


        loss1 = ratio * advantages
        loss2 = torch.clamp(ratio, 1 - args.eps_clip, args.eps_clip) * advantages
        actor_loss = -torch.min(loss1, loss2).mean()







        entropy = torch.mean(
            - torch.exp(pi.log_prob(episodes.actions)) * pi.log_prob(episodes.actions))

        loss = actor_loss - args.entropy_wt * entropy
        return loss



    def adapt_offpolicy(self, args, epsiodes, first_order=False, params=None, lr=None, old_pi=None):
        if lr is None:
            lr = self.fast_lr
        self.baseline.fit(epsiodes)
        loss = self.inner_loss_offpolicy(args, epsiodes, params=params, old_pi=old_pi)
        params = self.policy.update_params(loss, step_size=lr, first_order=first_order,
                                           params=params)
        return params, loss

    def adpat_preprocess_grad(self, episodes, first_order=False, params=None, lr=None):
        if lr is None:
            lr = self.fast_lr
        self.baseline.fit(episodes)
        loss = self.inner_loss(episodes, params=params)
        params = self.policy.update_params_preprocess_grad(loss, step_size=lr, first_order=first_order,
                                           params=params)
        return params, loss

    def adapt(self, episodes, first_order=False, params=None, lr=None):
        """Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        if lr is None:
            lr = self.fast_lr
        self.baseline.fit(episodes)

        loss = self.inner_loss(episodes, params=params)


        params = self.policy.update_params(loss, step_size=lr, first_order=first_order,
                                           params=params)

        return params, loss


    def sample_time(self, args, tasks, first_order=False):
        episodes = []
        losses = []
        task_num = 1
        start1 = time.time()
        for task in tasks:
            self.policy.reset_context()
            support_episodes_lst = []
            quary_episodes_lst = []
            support_losses_lst = []
            for j in range(args.support_episode_num):
                params = None
                cur_support_data = task[j * args.episode_window: (j + 1) * args.episode_window]
                cur_support_data.index = cur_support_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                               batch_size=args.fast_batch_size, deterministic=False)
                support_episodes_lst.append(cur_train_episodes)
                params, loss = self.adapt(cur_train_episodes, first_order=first_order, params=params)
                support_losses_lst.append(loss.item())

                cur_valid_data = task[(j + 1) * args.episode_window: (j + 2) * args.episode_window]
                cur_valid_data.index = cur_valid_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_valid_data, episode_len=len(cur_valid_data))
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                           batch_size=args.fast_batch_size,
                                                           deterministic=False)
                quary_episodes_lst.append(valid_episodes)

            episodes.append((support_episodes_lst, quary_episodes_lst))
            losses.append((sum(support_losses_lst)/len(support_losses_lst)))

            end1 = time.time()
            print("task{} cost {} seconds".format(task_num, end1 - start1))
            task_num = task_num + 1

        return episodes, losses

    def sample_time_sequence_offpilicy(self, args, tasks, first_order=False, shuffle=False, deter_sample=True):
        episodes = []
        losses = []
        for task in tasks:
            params = None
            support_episodes_lst = []
            support_losses_lst = []
            train_data_lst = []
            for i in range(args.support_episode_num):
                temp_data = task.iloc[i * args.episode_window:(i + 1) * args.episode_window]
                temp_data.index = temp_data.datadate.factorize()[0]
                train_data_lst.append(temp_data)
            if shuffle == True:
                np.random.shuffle(train_data_lst)
                np.random.shuffle(train_data_lst)
            for j in range(len(train_data_lst)):
                cur_support_data = train_data_lst[j]
                cur_support_data.index = cur_support_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                if deter_sample == True:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                cur_train_episodes = self.sampler.sample_dummy(self.policy, params=None, gamma=self.gamma,
                                                               batch_size=args.fast_batch_size, deterministic=False)
                support_episodes_lst.append(cur_train_episodes)
                cur_old_pi = self.policy(cur_train_episodes.observations, params=None)
                cur_old_pi = detach_distribution(cur_old_pi)
                params, loss = self.adapt_offpolicy(args, cur_train_episodes, first_order=first_order, params=params,
                                                    lr=self.fast_lr, old_pi=cur_old_pi)
                support_losses_lst.append(loss.item())
            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_new_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data))
            if deter_sample:
                utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
            valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                       batch_size=args.fast_batch_size * args.support_episode_num,
                                                       deterministic=False)
            episodes.append((support_episodes_lst, valid_episodes))

            losses.append((sum(support_losses_lst) / len(support_losses_lst)))


        return episodes, losses

    def sample_time_sequence_preprocess_grad(self, args, tasks, first_order=False, shuffle=False, change_rate=False):
        """ 产生样本/轨迹,保证每个episode内随机数相同，使得样本完全由 策略+行情决定 """
        episodes = []
        losses = []

        for task in tasks:
            params = None
            self.policy.reset_context()
            support_episodes_lst = []
            support_losses_lst = []
            train_data_lst = []
            for i in range(args.support_episode_num):
                temp_data = task.iloc[i * args.episode_window:(i + 1) * args.episode_window]
                temp_data.index = temp_data.datadate.factorize()[0]
                train_data_lst.append(temp_data)
            if shuffle == True:
                np.random.shuffle(train_data_lst)
                np.random.shuffle(train_data_lst)

            start_lr = args.fast_lr
            for j in range(len(train_data_lst)):
                cur_support_data = train_data_lst[j]
                cur_support_data.index = cur_support_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                               batch_size=args.fast_batch_size, deterministic=False)
                support_episodes_lst.append(cur_train_episodes)
                params, loss = self.adpat_preprocess_grad(cur_train_episodes, first_order=first_order,
                                                          params=params, lr=start_lr)
                if change_rate:
                    start_lr = start_lr * 0.5
                support_losses_lst.append(loss.item())

            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_new_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data))
            utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
            valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                       batch_size=args.fast_batch_size * args.support_episode_num,
                                                       deterministic=False)
            episodes.append((support_episodes_lst, valid_episodes))

            losses.append((sum(support_losses_lst) / len(support_losses_lst)))


        return episodes, losses



    def sample_time_sequence(self, args, tasks, first_order=False, shuffle=False, deter_sample=True):
        """ 产生样本/轨迹,保证每个episode内随机数相同，使得样本完全由 策略+行情决定 """
        episodes = []
        losses = []

        for task in tasks:
            params = None
            self.policy.reset_context()
            support_episodes_lst = []
            support_losses_lst = []
            train_data_lst = []
            for i in range(args.support_episode_num):
                temp_data = task.iloc[i * args.episode_window:(i + 1) * args.episode_window]
                temp_data.index = temp_data.datadate.factorize()[0]
                train_data_lst.append(temp_data)
            if shuffle == True:
                np.random.shuffle(train_data_lst)
                np.random.shuffle(train_data_lst)

            for j in range(len(train_data_lst)):
                cur_support_data = train_data_lst[j]
                cur_support_data.index = cur_support_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                if deter_sample:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                               batch_size=args.fast_batch_size, deterministic=False)
                support_episodes_lst.append(cur_train_episodes)
                params, loss = self.adapt(cur_train_episodes, first_order=first_order, params=params, lr=self.fast_lr)
                support_losses_lst.append(loss.item())

            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_new_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data))
            if deter_sample:
                utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
            valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                       batch_size=args.fast_batch_size * args.support_episode_num,
                                                       deterministic=False)
            episodes.append((support_episodes_lst, valid_episodes))

            losses.append((sum(support_losses_lst) / len(support_losses_lst)))


        return episodes, losses

    def sample_notime_task(self, args, task_chains, first_order=False, deter_sample=False):
        episodes = []
        losses = []
        for i in range(len(task_chains)):
            print("task chain{}".format(i))
            task_chain = task_chains[i]
            for task_num in range(args.task_chain_len):
                cur_task = task_chain[task_num * args.episode_window * (args.task_episode_num - 1):
                                      task_num * args.episode_window * (args.task_episode_num - 1) +
                                      args.episode_window * args.task_episode_num]
                cur_task.index = cur_task.datadate.factorize()[0]
                params = None
                train_episodes = self.sample_task_support_episodes(args, cur_task, params,
                                                                   batch_per_peisode=args.fast_batch_size,
                                                                   deterministic=False, deter_sample=deter_sample)
                params, loss = self.adapt(train_episodes, first_order=first_order, params=params)
                valid_data = cur_task.iloc[args.episode_window * args.support_episode_num:]
                valid_data.index = valid_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data))
                if deter_sample == True:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                           batch_size=args.fast_batch_size * args.support_episode_num,
                                                           deterministic=False)
                episodes.append((train_episodes, valid_episodes))
                if task_num == args.task_chain_len - 1:
                    losses.append(loss.item())

            return episodes, losses

    def sample_time_task_and_episode(self, args, task_chains, first_order=False, deter_sample=False):
        episodes = []
        losses = []
        for i in range(len(task_chains)):
            print("task chain{}".format(i))
            params = None
            task_chain = task_chains[i]
            for task_num in range(args.task_chain_len):
                cur_task = task_chain[task_num * args.episode_window * (args.task_episode_num - 1):
                                      task_num * args.episode_window * (args.task_episode_num - 1) +
                                      args.episode_window * args.task_episode_num]
                cur_task.index = cur_task.datadate.factorize()[0]
                train_episodes = self.sample_task_support_episodes(args, cur_task, params,
                                                                   batch_per_peisode=args.fast_batch_size,
                                                                   deterministic=False, deter_sample=deter_sample)
                params, loss = self.adapt(train_episodes, first_order=first_order, params=params)
                valid_data = cur_task.iloc[args.episode_window * args.support_episode_num:]
                valid_data.index = valid_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data))
                if deter_sample == True:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                           batch_size=args.fast_batch_size * args.support_episode_num,
                                                           deterministic=False)
                episodes.append((train_episodes, valid_episodes))
                if task_num == args.task_chain_len - 1:
                    losses.append(loss.item())

        return episodes, losses

    def sample_time_task(self, args, task_chains, first_order=False, deter_sample=False):
        episodes = []
        losses = []
        for i in range(len(task_chains)):
            print("task chain{}".format(i))
            params = None
            task_chain = task_chains[i]
            for task_num in range(args.task_chain_len):
                cur_task = task_chain[task_num * args.episode_window * (args.task_episode_num - 1):
                                      task_num * args.episode_window * (args.task_episode_num - 1) +
                                      args.episode_window * args.task_episode_num]
                cur_task.index = cur_task.datadate.factorize()[0]
                train_episodes = self.sample_task_support_episodes(args, cur_task, params,
                                                                   batch_per_peisode=args.fast_batch_size,
                                                                   deterministic=False, deter_sample=deter_sample)
                params, loss = self.adapt(train_episodes, first_order=first_order, params=params)
                valid_data = cur_task.iloc[args.episode_window * args.support_episode_num:]
                valid_data.index = valid_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data))
                if deter_sample == True:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                           batch_size=args.fast_batch_size * args.support_episode_num,
                                                           deterministic=False)
                episodes.append((train_episodes, valid_episodes))
                if task_num == args.task_chain_len - 1:
                    losses.append(loss.item())

        return episodes, losses

    def sample_for_each_task(self, args, task, first_order=False, deter_sample=True):
        ''' 单个任务的采样函数(协程函数)
        返回支撑集轨迹，查询集轨迹，任务内循环损失函数
        '''
        params = None
        train_episodes = self.sample_support_episodes(args, self.policy, task,
                                                      batch_per_peisode=args.fast_batch_size,
                                                      deterministic=False, deter_sample=deter_sample)
        params, loss = self.adapt(train_episodes, first_order=first_order, params=params)
        valid_data = task.iloc[args.episode_window * args.support_episode_num:]
        valid_data.index = valid_data.datadate.factorize()[0]
        self.sampler.create_new_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data))
        if deter_sample == True:
            utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
        valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                   batch_size=args.fast_batch_size * args.support_episode_num,
                                                   deterministic=False)
        return (train_episodes, valid_episodes), loss.item()

    def sample_bing(self, args, tasks, first_order=False, deter_sample=False):
        episodes = []
        losses = []
        with mp.Pool(processes=5) as pool:
            results = [pool.apply_async(self.sample_for_each_task, (args, task, first_order, deter_sample))
                       for task in tasks]
            for result in results:
                train_val_episodes, loss = result.get()
                episodes.append(train_val_episodes)
                losses.append(loss)
        return episodes, losses

    def sample_origin(self, args, tasks, first_order=False, deter_sample=False):
        """训练时,支撑集每个episode采样K条轨迹，3个episode采样3*K条轨迹;
           测试时,查询集的episode采样3*K条轨迹
        """
        episodes = []
        losses = []

        for task in tasks:
            params = None
            all_episodes = self.sample_support_episodes(args, self.policy, task,
                                                        batch_per_peisode=args.fast_batch_size,
                                                        deterministic=False, deter_sample=deter_sample)

            params, loss = self.adapt(all_episodes, first_order=True, params=params)
            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_new_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data))
            if deter_sample == True:
                utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
            valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                       batch_size=args.fast_batch_size * args.support_episode_num,
                                                       deterministic=False)
            episodes.append((all_episodes, valid_episodes))
            losses.append(loss.item())



        return episodes, losses











































































































    def test_trading_task(self, args, task_chain, num_steps, batch_size, halve_lr, last_episode_action, deter_sample):
        params = None
        for task_num in range(args.task_chain_len):
            if task_num == args.task_chain_len - 1:
                cur_task = task_chain[task_num * args.episode_window * (args.task_episode_num - 1):]
            else:
                cur_task = task_chain[task_num * args.episode_window * (args.task_episode_num - 1):
                                      task_num * args.episode_window * (args.task_episode_num - 1) +
                                      args.episode_window * args.task_episode_num]
            cur_task.index = cur_task.datadate.factorize()[0]
            for i in range(1, num_steps + 1):
                if i == 1 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                train_episodes = self.sample_task_support_episodes(args, cur_task, params,
                                                                   batch_per_peisode=batch_size,
                                                                   deterministic=False, deter_sample=deter_sample)
                params, loss = self.adapt(train_episodes, first_order=True, params=params, lr=lr)

            if task_num == args.task_chain_len - 1:
                valid_data = cur_task.iloc[args.episode_window * args.support_episode_num:]
                valid_data.index = valid_data.datadate.factorize()[0]
                self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                          last_episode_action=last_episode_action)
                if deter_sample == True:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)

                valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma, batch_size=1,
                                                           deterministic=True)
                return valid_episodes.actions, valid_episodes.rewards


    def test_trading(self, args, task_data, num_steps, batch_size, halve_lr, last_episode_action, deter_sample):
        """实际交易：串行环境交易
        """
        params = None
        self.policy.reset_context()
        train_data = task_data.iloc[0:args.episode_window * args.support_episode_num]
        train_data.index = train_data.datadate.factorize()[0]
        for i in range(1, num_steps + 1):
            support_episodes_lst = []
            for j in range(args.support_episode_num):
                cur_support_data = task_data[j*args.episode_window: (j+1)*args.episode_window]
                cur_support_data.index = cur_support_data.datadate.factorize()[0]

                self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                if deter_sample == True:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                               batch_size=batch_size, deterministic=False)
                support_episodes_lst.append(cur_train_episodes)
            all_episodes = BatchEpisodes2(batch_size=batch_size*args.support_episode_num, gamma=args.gamma,
                                          device=self.device)
            temp_observations_lst = []
            temp_actions_lst = []
            temp_rewards_lst = []
            for tt in range(args.support_episode_num):
                temp_observations_lst += support_episodes_lst[tt]._observations_list
                temp_actions_lst += support_episodes_lst[tt]._actions_list
                temp_rewards_lst += support_episodes_lst[tt]._rewards_list
            all_episodes._observations_list = temp_observations_lst
            all_episodes._actions_list = temp_actions_lst
            all_episodes._rewards_list = temp_rewards_lst
            if i == 1:
                lr = self.fast_lr
            else:
                lr = self.fast_lr / 2
            params, loss = self.adapt(all_episodes, first_order=True, params=params, lr=lr)

        valid_data = task_data.iloc[args.episode_window * args.support_episode_num:]
        valid_data.index = valid_data.datadate.factorize()[0]
        self.sampler.create_test_DummyVecEnv_envs(args.seed, episode_data=valid_data, episode_len=len(valid_data),
                                      last_episode_action=last_episode_action)
        if deter_sample == True:
            utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)

        valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma, batch_size=1,
                                                   deterministic=True)

        self.policy.reset_context()
        return valid_episodes.actions, valid_episodes.rewards


    def test_trading_time_offpolicy(self, args, task_data, num_steps, batch_size, halve_lr, last_episode_action,
                                    deter_sample=True):
        params = None
        for i in range(1, num_steps + 1):

            if i == 1 and halve_lr:
                lr = self.fast_lr / 2
            else:
                lr = self.fast_lr

            support_data_lst = []
            for tt in range(args.support_episode_num):
                temp_data = task_data.iloc[tt * args.episode_window:(tt + 1) * args.episode_window]
                temp_data.index = temp_data.datadate.factorize()[0]
                support_data_lst.append(temp_data)


            start_lr = lr
            cur_support_data = support_data_lst[0]
            cur_support_data.index = cur_support_data.datadate.factorize()[0]
            self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
            if deter_sample:
                utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)

            cur_train_episodes = self.sampler.sample_dummy(self.policy, params=None, gamma=self.gamma,
                                                           batch_size=batch_size, deterministic=False)
            cur_old_pi = self.policy(cur_train_episodes.observations)
            cur_old_pi = detach_distribution(cur_old_pi)
            params, loss = self.adapt_offpolicy(args, cur_train_episodes, first_order=True, params=params,
                                                lr=start_lr, old_pi=cur_old_pi)

            for j in range(1, len(support_data_lst)):
                cur_support_data = support_data_lst[j]
                cur_support_data.index = cur_support_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                if deter_sample == True:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                cur_train_episodes = self.sampler.sample_dummy(self.policy, params=None, gamma=self.gamma,
                                                               batch_size=batch_size, deterministic=False)
                cur_old_pi = self.policy(cur_train_episodes.observations)
                cur_old_pi = detach_distribution(cur_old_pi)

                params, loss = self.adapt_offpolicy(args, cur_train_episodes, first_order=True, params=params,
                                                    lr=start_lr, old_pi=cur_old_pi)
        valid_data = task_data.iloc[args.episode_window * args.support_episode_num:]
        valid_data.index = valid_data.datadate.factorize()[0]
        self.sampler.create_test_DummyVecEnv_envs(args.seed, episode_data=valid_data, episode_len=len(valid_data),
                                                  last_episode_action=last_episode_action)
        if deter_sample == True:
            utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
        valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma, batch_size=1,
                                                   deterministic=True)

        self.policy.reset_context()
        return valid_episodes.actions, valid_episodes.rewards


    def test_trading_time(self, args, task_data, num_steps, batch_size, halve_lr, last_episode_action,
                          deter_sample=True, shuffle=False):
        params = None
        support_data_lst = []
        for tt in range(args.support_episode_num):
            cur_support_data = task_data[tt * args.episode_window: (tt + 1) * args.episode_window]
            cur_support_data.index = cur_support_data.datadate.factorize()[0]
            support_data_lst.append(cur_support_data)
        if shuffle == True:
            np.random.shuffle(support_data_lst)
            np.random.shuffle(support_data_lst)
            np.random.shuffle(support_data_lst)
        for i in range(1, num_steps + 1):
            if i == 2 and halve_lr:
                lr = self.fast_lr / 2
            else:
                lr = self.fast_lr






            for j in range(len(support_data_lst)):
                cur_support_data = support_data_lst[j]
                cur_support_data.index = cur_support_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                if deter_sample:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                               batch_size=batch_size, deterministic=False)
                params, loss = self.adapt(cur_train_episodes, first_order=True, params=params, lr=lr)
        valid_data = task_data.iloc[args.episode_window * args.support_episode_num:]
        valid_data.index = valid_data.datadate.factorize()[0]
        self.sampler.create_test_DummyVecEnv_envs(args.seed, episode_data=valid_data, episode_len=len(valid_data),
                                                  last_episode_action=last_episode_action)
        if deter_sample:
            utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
        valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma, batch_size=1,
                                                   deterministic=True)

        self.policy.reset_context()
        return valid_episodes.actions, valid_episodes.rewards


    def test_trading_time2(self, args, task_data, num_steps, batch_size, halve_lr, last_episode_action,
                           deter_sample=True, shuffle=False, half_mode="half_accord_episode"):
        '''考虑episode间关系，多次更新时是每个episode多次更新后在下个episode使用，而不是123--123这样'''
        ''' 即：每个episode多次更新'''
        params = None
        support_data_lst = []
        for tt in range(args.support_episode_num):
            cur_support_data = task_data[tt * args.episode_window: (tt + 1) * args.episode_window]
            cur_support_data.index = cur_support_data.datadate.factorize()[0]
            support_data_lst.append(cur_support_data)
        if shuffle == True:
            np.random.shuffle(support_data_lst)
            np.random.shuffle(support_data_lst)
            np.random.shuffle(support_data_lst)

        for j in range(len(support_data_lst)):
            if half_mode == "half_accord_episode":
                if j == 0:
                    lr = self.fast_lr
                else:
                    lr = self.fast_lr / 2
            cur_support_data = support_data_lst[j]
            self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
            for i in range(1, num_steps + 1):
                if half_mode == "half_in_episode":
                    if i == 1:
                        lr = self.fast_lr
                    else:
                        lr = self.fast_lr / 2
                if half_mode == "half_1episode_1update":
                    if j == 0 and i == 1:
                        lr = self.fast_lr
                    else:
                        lr = self.fast_lr / 2

                if deter_sample:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                               batch_size=batch_size, deterministic=False)
                params, loss = self.adapt(cur_train_episodes, first_order=True, params=params, lr=lr)
        valid_data = task_data.iloc[args.episode_window * args.support_episode_num:]
        valid_data.index = valid_data.datadate.factorize()[0]
        self.sampler.create_test_DummyVecEnv_envs(args.seed, episode_data=valid_data,
                                                  episode_len=len(valid_data),
                                                  last_episode_action=last_episode_action)
        if deter_sample:
            utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
        valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma, batch_size=1,
                                                   deterministic=True)
        return valid_episodes.actions, valid_episodes.rewards


    def test_muti_times_time(self, args, tasks, num_steps, batch_size, halve_lr):
        all_quary_episode_lst = []

        for task in tasks:
            params_lst = [None]
            params = None
            for i in range(1, num_steps + 1):
                if i == 1 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                cur_support_data = task[(args.support_episode_num-1)*args.episode_window:(args.support_episode_num)*args.episode_window]
                cur_support_data.index = cur_support_data.datadate.factorize()[0]
                self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                               batch_size=batch_size, deterministic=False)
                params, loss = self.adapt(cur_train_episodes, first_order=True, params=params, lr=lr)
                params_lst.append(params)
            cur_val_episodes = []
            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                      last_episode_action=0)
            for j in range(len(params_lst)):
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params_lst[j], gamma=self.gamma,
                                                           batch_size=1, deterministic=True)
                cur_val_episodes.append(valid_episodes)
            all_quary_episode_lst.append(cur_val_episodes)


        return all_quary_episode_lst


    def test_muti_times_task(self, args, val_taskchains, num_steps, batch_size, halve_lr, deter_sample=False):
        all_quary_episodes_lst = []
        for task_chain in val_taskchains:
            params = None

            for task_num in range(args.task_chain_len):
                cur_task = task_chain[task_num * args.episode_window * (args.task_episode_num - 1):
                                      task_num * args.episode_window * (args.task_episode_num - 1) +
                                      args.episode_window * args.task_episode_num]
                cur_task.index = cur_task.datadate.factorize()[0]
                params_lst = [None]
                for i in range(1, num_steps + 1):
                    if i == 1 and halve_lr:
                        lr = self.fast_lr / 2
                    else:
                        lr = self.fast_lr
                    train_episodes = self.sample_task_support_episodes(args, cur_task, params,
                                                                       batch_per_peisode=batch_size,
                                                                       deterministic=False,
                                                                       deter_sample=deter_sample)
                    params, loss = self.adapt(train_episodes, first_order=True, params=params, lr=lr)
                    params_lst.append(params)

                if task_num == args.task_chain_len - 1:
                    valid_data = cur_task.iloc[args.episode_window * args.support_episode_num:]
                    valid_data.index = valid_data.datadate.factorize()[0]
                    self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                              last_episode_action=0)
                    cur_val_episodes = []
                    for j in range(len(params_lst)):
                        if deter_sample == True:
                            utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                        valid_episodes = self.sampler.sample_dummy(self.policy, params=params_lst[j], gamma=self.gamma,
                                                                   batch_size=1, deterministic=True)
                        cur_val_episodes.append(valid_episodes)

                    all_quary_episodes_lst.append(cur_val_episodes)
        return all_quary_episodes_lst

    def test_muti_times(self, args, tasks, num_steps, batch_size, halve_lr, deter_sample=False):
        all_quary_episode_lst = []

        for task in tasks:
            params_lst = [None]
            params = None
            for i in range(1, num_steps + 1):
                if i == 2 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                support_episodes_lst = []
                for j in range(args.support_episode_num):
                    cur_support_data = task[j * args.episode_window: (j + 1) * args.episode_window]
                    cur_support_data.index = cur_support_data.datadate.factorize()[0]
                    self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                    if deter_sample == True:
                        utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                    cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                                   batch_size=batch_size, deterministic=False)
                    support_episodes_lst.append(cur_train_episodes)
                all_episodes = BatchEpisodes2(batch_size=batch_size*args.support_episode_num, gamma=args.gamma, device=self.device)
                temp_observations_lst = []
                temp_actions_lst = []
                temp_rewards_lst = []
                for tt in range(args.support_episode_num):
                    temp_observations_lst += support_episodes_lst[tt]._observations_list
                    temp_actions_lst += support_episodes_lst[tt]._actions_list
                    temp_rewards_lst += support_episodes_lst[tt]._rewards_list
                all_episodes._observations_list = temp_observations_lst
                all_episodes._actions_list = temp_actions_lst
                all_episodes._rewards_list = temp_rewards_lst
                params, loss = self.adapt(all_episodes, first_order=True, params=params, lr=lr)
                params_lst.append(params)
            cur_val_episodes = []
            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                      last_episode_action=0)
            for j in range(len(params_lst)):
                if deter_sample == True:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params_lst[j], gamma=self.gamma,
                                                           batch_size=1, deterministic=True)
                cur_val_episodes.append(valid_episodes)
            all_quary_episode_lst.append(cur_val_episodes)


        return all_quary_episode_lst

    def test_muti_times_sequence_offpolicy(self, args, tasks, num_steps, batch_size, halve_lr,
                                           shuffle=False, deter_sample=True):
        all_quary_episode_lst = []
        task_num = 1
        start1 = time.time()
        for task in tasks:
            params_lst = [None]
            params = None
            for i in range(1, num_steps + 1):
                if i == 2 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                support_data_lst = []
                for tt in range(args.support_episode_num):
                    temp_data = task.iloc[tt * args.episode_window:(tt + 1) * args.episode_window]
                    temp_data.index = temp_data.datadate.factorize()[0]
                    support_data_lst.append(temp_data)
                if shuffle == True:
                    np.random.shuffle(support_data_lst)
                    np.random.shuffle(support_data_lst)
                    np.random.shuffle(support_data_lst)

                for j in range(len(support_data_lst)):
                    cur_support_data = support_data_lst[j]
                    cur_support_data.index = cur_support_data.datadate.factorize()[0]
                    self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data,
                                                             episode_len=len(cur_support_data))
                    if deter_sample == True:
                        utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                    cur_train_episodes = self.sampler.sample_dummy(self.policy, params=None, gamma=self.gamma,
                                                                   batch_size=batch_size, deterministic=False)
                    cur_old_pi = self.policy(cur_train_episodes.observations, params=None)
                    cur_old_pi = detach_distribution(cur_old_pi)

                    params, loss = self.adapt_offpolicy(args, cur_train_episodes, first_order=True, params=params,
                                                        lr=lr, old_pi=cur_old_pi)
                params_lst.append(params)
            cur_val_episodes = []
            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                      last_episode_action=0)
            for j in range(len(params_lst)):
                if deter_sample == True:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params_lst[j], gamma=self.gamma,
                                                           batch_size=1, deterministic=True)
                cur_val_episodes.append(valid_episodes)

            all_quary_episode_lst.append(cur_val_episodes)
            end1 = time.time()
            task_num = task_num + 1
            print("val task{} cost {} seconds".format(task_num, end1-start1))
        return all_quary_episode_lst

    def test_muti_times_sequence_preprocess_grad(self, args, tasks, num_steps, batch_size, halve_lr, shuffle=False):
        all_quary_episode_lst = []

        for task in tasks:
            params_lst = [None]
            params = None
            for i in range(1, num_steps + 1):
                if i == 1 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                support_data_lst = []
                for tt in range(args.support_episode_num):
                    temp_data = task.iloc[tt * args.episode_window:(tt + 1) * args.episode_window]
                    temp_data.index = temp_data.datadate.factorize()[0]
                    support_data_lst.append(temp_data)

                if shuffle == True:
                    np.random.shuffle(support_data_lst)
                    np.random.shuffle(support_data_lst)
                    np.random.shuffle(support_data_lst)

                for j in range(len(support_data_lst)):
                    cur_support_data = support_data_lst[j]
                    cur_support_data.index = cur_support_data.datadate.factorize()[0]
                    self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data,
                                                             episode_len=len(cur_support_data))
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                    cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                                   batch_size=batch_size, deterministic=False)
                    params, loss = self.adpat_preprocess_grad(cur_train_episodes, first_order=True, params=params, lr=lr)
                params_lst.append(params)
            cur_val_episodes = []
            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                      last_episode_action=0)
            for j in range(len(params_lst)):
                utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params_lst[j], gamma=self.gamma,
                                                           batch_size=1, deterministic=True)
                cur_val_episodes.append(valid_episodes)
            all_quary_episode_lst.append(cur_val_episodes)


        return all_quary_episode_lst

    def test_muti_times_sequence_cavia(self, args, tasks, num_steps, batch_size, halve_lr, shuffle=False, deter_sample=True):
        all_quary_episode_lst = []

        for task in tasks:
            self.policy.reset_context()
            params = None
            cur_val_episodes = []

            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                      last_episode_action=0)
            if deter_sample:
                utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
            valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                       batch_size=1, deterministic=True)
            cur_val_episodes.append(valid_episodes)


            for i in range(1, 4 + 1):
                if i == 1 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                support_data_lst = []
                for tt in range(args.support_episode_num):
                    temp_data = task.iloc[tt * args.episode_window:(tt + 1) * args.episode_window]
                    temp_data.index = temp_data.datadate.factorize()[0]
                    support_data_lst.append(temp_data)
                if shuffle == True:
                    np.random.shuffle(support_data_lst)
                    np.random.shuffle(support_data_lst)
                    np.random.shuffle(support_data_lst)
                for j in range(len(support_data_lst)):
                    cur_support_data = support_data_lst[j]
                    cur_support_data.index = cur_support_data.datadate.factorize()[0]
                    self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data,
                                                             episode_len=len(cur_support_data))
                    if deter_sample:
                        utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                    cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                                   batch_size=batch_size, deterministic=False)
                    params, loss = self.adapt(cur_train_episodes, first_order=True, params=params, lr=lr)

                valid_data = task.iloc[args.episode_window * args.support_episode_num:]
                valid_data.index = valid_data.datadate.factorize()[0]
                self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                          last_episode_action=0)
                if deter_sample:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                           batch_size=1, deterministic=True)
                cur_val_episodes.append(valid_episodes)

            all_quary_episode_lst.append(cur_val_episodes)


        return all_quary_episode_lst


    def test_muti_times_sequence(self, args, tasks, num_steps, batch_size, halve_lr, shuffle=False, deter_sample=True):
        all_quary_episode_lst = []
        for task in tasks:
            params_lst = [None]
            params = None
            for i in range(1, num_steps + 1):
                if i == 2 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                support_data_lst = []
                for tt in range(args.support_episode_num):
                    temp_data = task.iloc[tt * args.episode_window:(tt + 1) * args.episode_window]
                    temp_data.index = temp_data.datadate.factorize()[0]
                    support_data_lst.append(temp_data)

                if shuffle == True:
                    np.random.shuffle(support_data_lst)
                    np.random.shuffle(support_data_lst)
                    np.random.shuffle(support_data_lst)

                for j in range(len(support_data_lst)):
                    cur_support_data = support_data_lst[j]
                    self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data,
                                                             episode_len=len(cur_support_data))
                    if deter_sample:
                        utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                    cur_train_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                                   batch_size=batch_size, deterministic=False)
                    params, loss = self.adapt(cur_train_episodes, first_order=True, params=params, lr=lr)
                params_lst.append(params)
            cur_val_episodes = []
            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                      last_episode_action=0)
            for j in range(len(params_lst)):
                if deter_sample:
                    utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params_lst[j], gamma=self.gamma,
                                                           batch_size=1, deterministic=True)
                cur_val_episodes.append(valid_episodes)
            all_quary_episode_lst.append(cur_val_episodes)
        return all_quary_episode_lst

    def test_muti_times_jincheng(self, args, tasks, num_steps, batch_size, halve_lr):
        all_quary_episode_lst = []
        for task in tasks:
            params_lst = []
            params = None
            for i in range(1, num_steps+1):
                if i == 1 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                support_episodes_lst = []
                for j in range(args.support_episode_num):
                    cur_support_data = task[j * args.episode_window: (j + 1) * args.episode_window]
                    cur_support_data.index = cur_support_data.datadate.factorize()[0]
                    self.sampler.close()
                    self.sampler.create_new_envs(args.seed, cur_support_data, episode_len=len(cur_support_data))
                    cur_train_episodes = self.sampler.sample(self.policy, params=params, gamma=self.gamma,
                                                             batch_size=batch_size, deterministic=False)
                    support_episodes_lst.append(cur_train_episodes)
                all_episodes = BatchEpisodes(batch_size=batch_size*args.support_episode_num, gamma=args.gamma, device=self.device)
                temp_observations_lst = []
                temp_actions_lst = []
                temp_rewards_lst = []
                for tt in range(args.support_episode_num):
                    temp_observations_lst += support_episodes_lst[tt]._observations_list
                    temp_actions_lst += support_episodes_lst[tt]._actions_list
                    temp_rewards_lst += support_episodes_lst[tt]._rewards_list
                all_episodes._observations_list = temp_observations_lst
                all_episodes._actions_list = temp_actions_lst
                all_episodes._rewards_list = temp_rewards_lst
                params, loss = self.adapt(all_episodes, first_order=True, params=params, lr=lr)
                params_lst.append(params)
            cur_val_episodes = []
            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.close()
            self.sampler.create_test_envs(args.seed, valid_data, episode_len=len(valid_data), last_episode_action=0)
            for j in range(len(params_lst)):
                valid_episodes = self.sampler.sample_for_test(self.policy, params=params_lst[j], gamma=self.gamma,
                                                              batch_size=1, deterministic=True)
                cur_val_episodes.append(valid_episodes)
            all_quary_episode_lst.append(cur_val_episodes)
        return all_quary_episode_lst

    def test_my22_cavia(self, args, tasks, num_steps, batch_size, halve_lr):
        """等待修改.........
        """
        support_episodes_per_task = []
        quary_episodes_per_task = []
        task_num = 1
        for task in tasks:
            self.policy.reset_context()
            self.sampler.reset_task(task)
            params = None
            train_data = task.iloc[0:args.episode_window * args.support_episode_num]
            train_data.index = train_data.datadate.factorize()[0]
            self.sampler.close()
            self.sampler.create_new_envs(args.seed, train_data, episode_len=len(train_data))

            train_episodes = self.sampler.sample(self.policy, gamma=self.gamma, params=params, batch_size=batch_size,
                                                 deterministic=False)
            curr_support_episodes = [train_episodes]
            curr_quary_episodes = []
            context_params_lst = []
            curr_params = []
            for i in range(1, num_steps + 1):

                if i == 1 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                params, loss = self.adapt(train_episodes, first_order=True, params=params, lr=lr)
                context_params_lst.append(self.policy.context_params)
                curr_params.append(params)

                train_episodes = self.sampler.sample(self.policy, gamma=self.gamma, params=params,
                                                     batch_size=batch_size, deterministic=False)
                curr_support_episodes.append(train_episodes)

            support_episodes_per_task.append(curr_support_episodes)

            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            self.sampler.close()
            self.sampler.create_test_envs(args.seed, valid_data, episode_len=len(valid_data), last_episode_action=0)

            for j in range(len(context_params_lst)):
                self.policy.context_params = context_params_lst[j]
                valid_episodes = self.sampler.sample_for_test(self.policy, params=curr_params[j], gamma=self.gamma,
                                                              batch_size=1, deterministic=True)
                curr_quary_episodes.append(valid_episodes)

            quary_episodes_per_task.append(curr_quary_episodes)

            task_num = task_num + 1

        self.policy.reset_context()
        return support_episodes_per_task, quary_episodes_per_task


    def sample_support_episodes(self, args, cur_policy, task, batch_per_peisode, params=None, deterministic=False,
                                deter_sample=False):
        """对任务task的支撑集采样(lstm专属)
        """
        support_episodes_lst = []
        for j in range(args.support_episode_num):
            cur_support_data = task[j * args.episode_window: (j + 1) * args.episode_window]
            cur_support_data.index = cur_support_data.datadate.factorize()[0]
            self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data,
                                                     episode_len=len(cur_support_data))
            if deter_sample == True:
                utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
            cur_support_episodes = self.sampler.sample_dummy(cur_policy, params=params, gamma=self.gamma,
                                                             batch_size=batch_per_peisode, deterministic=deterministic)
            support_episodes_lst.append(cur_support_episodes)

        train_episodes = BatchEpisodes2(batch_size=batch_per_peisode * args.support_episode_num, gamma=args.gamma,
                                        device=self.device)
        temp_observations_lst = []
        temp_actions_lst = []
        temp_rewards_lst = []
        for tt in range(args.support_episode_num):
            temp_observations_lst += support_episodes_lst[tt]._observations_list
            temp_actions_lst += support_episodes_lst[tt]._actions_list
            temp_rewards_lst += support_episodes_lst[tt]._rewards_list
        train_episodes._observations_list = temp_observations_lst
        train_episodes._actions_list = temp_actions_lst
        train_episodes._rewards_list = temp_rewards_lst

        return train_episodes



    def sample_task_support_episodes(self, args, task, params, batch_per_peisode, deterministic=False,
                                     deter_sample=True):
        """对任务task的支撑集采样(将支撑集 数据 多个episode的轨迹 拼接)
        """
        support_episodes_lst = []
        for j in range(args.support_episode_num):
            cur_support_data = task[j * args.episode_window: (j + 1) * args.episode_window]
            cur_support_data.index = cur_support_data.datadate.factorize()[0]
            self.sampler.create_new_DummyVecEnv_envs(args.seed, cur_support_data,
                                                     episode_len=len(cur_support_data))
            if deter_sample == True:
                utils.set_torch_seed(args.seed, cudnn=args.make_deterministic)
            cur_support_episodes = self.sampler.sample_dummy(self.policy, params=params, gamma=self.gamma,
                                                             batch_size=batch_per_peisode, deterministic=deterministic)
            support_episodes_lst.append(cur_support_episodes)
        train_episodes = BatchEpisodes2(batch_size=batch_per_peisode * args.support_episode_num, gamma=args.gamma,
                                        device=self.device)
        temp_observations_lst = []
        temp_actions_lst = []
        temp_rewards_lst = []
        for tt in range(args.support_episode_num):
            temp_observations_lst += support_episodes_lst[tt]._observations_list
            temp_actions_lst += support_episodes_lst[tt]._actions_list
            temp_rewards_lst += support_episodes_lst[tt]._rewards_list
        train_episodes._observations_list = temp_observations_lst
        train_episodes._actions_list = temp_actions_lst
        train_episodes._rewards_list = temp_rewards_lst

        return train_episodes

    def test_my22(self, args, tasks, num_steps, batch_size, halve_lr, result_dir, out_name):
        """测试：返回支撑集初始参数采样轨迹 + num_steps更新分别的轨迹,
        返回查询集初始参数采样轨迹 + num_steps更新分别的轨迹,
        具体更新方法：先采样再更新，因此最后更新参数也会采样轨迹
        """
        support_episodes_all_task = []
        quary_episodes_all_task = []
        last_episode_action_lst = [0] * (num_steps + 1)
        record_every_update_support_sum = [0] * (args.num_test_steps + 1)
        record_every_update_quary_sum = [0] * (args.num_test_steps + 1)
        for num in range(len(tasks)):
            task = tasks[num]
            self.policy.reset_context()

            params = None
            train_episodes = self.sample_task_support_episodes(args, task, params, batch_per_peisode=batch_size,
                                                               deterministic=False)
            cur_support_episodes = [train_episodes]
            cur_quary_episodes = []
            params_lst = [params]
            for i in range(1, num_steps + 1):

                if i == 1 and halve_lr:
                    lr = self.fast_lr / 2
                else:
                    lr = self.fast_lr
                params, loss = self.adapt(train_episodes, first_order=True, params=params, lr=lr)
                params_lst.append(params)
                train_episodes = self.sample_task_support_episodes(args, task, params, batch_per_peisode=batch_size,
                                                                   deterministic=True)
                cur_support_episodes.append(train_episodes)

            support_episodes_all_task.append(cur_support_episodes)


            valid_data = task.iloc[args.episode_window * args.support_episode_num:]
            valid_data.index = valid_data.datadate.factorize()[0]
            for j in range(len(params_lst)):
                self.sampler.create_test_DummyVecEnv_envs(args.seed, valid_data, episode_len=len(valid_data),
                                                          last_episode_action=last_episode_action_lst[j])
                valid_episodes = self.sampler.sample_dummy(self.policy, params=params_lst[j], gamma=self.gamma,
                                                           batch_size=1, deterministic=True)
                cur_quary_episodes.append(valid_episodes)
                last_episode_action_lst[j] = int(valid_episodes.actions[-1] - 1)
            quary_episodes_all_task.append(cur_quary_episodes)
            support_returns = total_rewards([cur_support_episodes])
            for tt in range(args.num_test_steps + 1):
                print("iter_" + str(tt) + " , " + "support_return:", support_returns[tt])
                record_every_update_support_sum[tt] += support_returns[tt]
            quary_returns = total_rewards([cur_quary_episodes])
            for tt in range(args.num_test_steps + 1):
                print("iter_" + str(tt) + " , " + "quary_returns:", quary_returns[tt])
                record_every_update_quary_sum[tt] += quary_returns[tt]
            print("support sum until now:", record_every_update_support_sum)
            print("quary sum until now:", record_every_update_quary_sum)
            if num == 0:
                f = open(os.path.join(result_dir, out_name + ".csv"), mode="w")
            else:
                f = open(os.path.join(result_dir, out_name + ".csv"), mode="a+")
            f.write("support" + str(num) + ",")
            for ele in record_every_update_support_sum:
                f.write(str(ele) + ",")
            f.write("\n")
            f.write("quary" + str(num) + ",")
            for ele in record_every_update_quary_sum:
                f.write(str(ele) + ",")
            f.write("\n")
            f.close()

        return record_every_update_support_sum, record_every_update_quary_sum

    def hessian_vector_product_time_sequence(self, kl, damping=1e-2):
        """
        Hessian-vector product, based on the Perlmutter method.
        """
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector):

            grad_kl_v = torch.dot(flat_grad_kl.float(), vector.float())
            grad2s = torch.autograd.grad(grad_kl_v.float(), self.policy.parameters(), retain_graph=True)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def hessian_vector_product_time_sequence_offpolicy(self, kl, damping=1e-2):
        """
        Hessian-vector product, based on the Perlmutter method.
        """
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector):

            grad_kl_v = torch.dot(flat_grad_kl.float(), vector.float())
            grad2s = torch.autograd.grad(grad_kl_v.float(), self.policy.parameters(), retain_graph=True)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            self.policy.reset_context()
            params, _ = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))


    def kl_divergence_time(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes_lst, valid_episodes_lst), old_pi in zip(episodes, old_pis):
            params = None
            self.policy.reset_context()
            for j in range(len(train_episodes_lst)):
                train_episodes = train_episodes_lst[j]
                valid_episodes = valid_episodes_lst[j]
                params, _ = self.adapt(train_episodes, params=params)

                pi = self.policy(valid_episodes.observations, params=params)

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
                kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def kl_divergence_time_sequence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episode_lst, valid_episodes), old_pi in zip(episodes, old_pis):
            self.policy.reset_context()
            params = None
            for train_episodes in train_episode_lst:
                params, _ = self.adapt(train_episodes, params=params)

            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def kl_divergence_time_sequence_offpolicy(self, args, episodes, old_pis=None, change_rate=False):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes_lst, valid_episodes), old_pi in zip(episodes, old_pis):
            self.policy.reset_context()
            params = None
            start_lr = self.fast_lr

            cur_episodes = train_episodes_lst[0]
            cur_inner_old_pi = self.policy(cur_episodes.observations)
            cur_inner_old_pi = detach_distribution(cur_inner_old_pi)

            params, _ = self.adapt_offpolicy(args, cur_episodes, first_order=False, params=params,
                                             lr=start_lr, old_pi=cur_inner_old_pi)
            if change_rate:
                start_lr = start_lr * 0.5
            for j in range(1, len(train_episodes_lst)):
                cur_episodes = train_episodes_lst[j]
                cur_inner_old_pi = self.policy(cur_episodes.observations)
                cur_inner_old_pi = detach_distribution(cur_inner_old_pi)
                params, _ = self.adapt_offpolicy(args, cur_episodes, first_order=False, params=params,
                                                 lr=start_lr, old_pi=cur_inner_old_pi)
                if change_rate:
                    start_lr = start_lr * 0.5
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))


    def hessian_vector_product_time(self, episodes, kl, damping=1e-2):
        """
        Hessian-vector product, based on the Perlmutter method.
        """
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector):

            grad_kl_v = torch.dot(flat_grad_kl.float(), vector.float())
            grad2s = torch.autograd.grad(grad_kl_v.float(), self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def hessian_vector_product(self, kl, damping=1e-2):
        """
        Hessian-vector product, based on the Perlmutter method.
        """
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector):

            grad_kl_v = torch.dot(flat_grad_kl.float(), vector.float())
            grad2s = torch.autograd.grad(grad_kl_v.float(), self.policy.parameters(), retain_graph=True)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def hessian_vector_product_task_time(self, args, episodes, damping=1e-2):
        """
        Hessian-vector product, based on the Perlmutter method.
        """

        def _product(vector):
            kl = self.kl_divergence_task_time(args, episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True,
                                        retain_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl.float(), vector.float())
            grad2s = torch.autograd.grad(grad_kl_v.float(), self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product


    def kl_divergence_task_time(self, args, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        count_task = 0
        params = None
        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            if count_task == 0:
                params = None
            params, _ = self.adapt(train_episodes, params=params)

            if count_task == args.task_chain_len - 1:
                pi = self.policy(valid_episodes.observations, params=params)
                if old_pi is None:
                    old_pi = detach_distribution(pi)
                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
                kls.append(kl)
                count_task = 0
            else:
                count_task = count_task + 1

        return torch.mean(torch.stack(kls, dim=0))



























































    def surrogate_loss_time_sequence_offpolicy(self, args, episodes, eps_clip=0.1, critic_weight=0.5, entropy_wt=0.01,
                                               old_pis=None, record_kls=False):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes_lst, valid_episodes), old_pi in zip(episodes, old_pis):
            params = None
            for j in range(len(train_episodes_lst)):
                cur_episodes = train_episodes_lst[j]
                cur_inner_old_pi = self.policy(cur_episodes.observations, params=None)
                cur_inner_old_pi = detach_distribution(cur_inner_old_pi)

                params, _ = self.adapt_offpolicy(args, cur_episodes, first_order=False, params=params,
                                                 lr=self.fast_lr, old_pi=cur_inner_old_pi)

            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))
                if old_pi is None:
                    old_pi = detach_distribution(pi)


                values = self.baseline(valid_episodes)

                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                                                weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                             - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss1 = ratio * advantages
                loss2 = torch.clamp(ratio, 1 - eps_clip, eps_clip) * advantages

                actor_loss = -torch.min(loss1, loss2).mean()




                entropy = torch.mean(
                    - torch.exp(pi.log_prob(valid_episodes.actions)) * pi.log_prob(valid_episodes.actions))


                loss = actor_loss - entropy_wt * entropy

                losses.append(loss)

                if record_kls:
                    mask = valid_episodes.mask
                    if valid_episodes.actions.dim() > 2:
                        mask = mask.unsqueeze(2)
                    kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
                    kls.append(kl)

        return torch.mean(torch.stack(losses, dim=0)), pis, kls


    def surrogate_loss_time_sequence(self, episodes, eps_clip=0.1, critic_weight=0.5, entropy_wt=0.01, old_pis=None,
                                     record_kls=False):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)
        critic_criterion = nn.MSELoss()

        for (train_episodes_lst, valid_episodes), old_pi in zip(episodes, old_pis):
            params = None
            self.policy.reset_context()
            for train_episodes in train_episodes_lst:
                params, _ = self.adapt(train_episodes, first_order=False, params=params)

            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                returns = valid_episodes.returns
                returns = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + 1e-5)
                values = self.baseline(valid_episodes)

                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                                                weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                             - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss1 = ratio * advantages
                loss2 = torch.clamp(ratio, 1 - eps_clip, eps_clip) * advantages

                actor_loss = -torch.min(loss1, loss2).mean()

                values_normalized = values.squeeze(2)
                values_normalized = weighted_normalize(values_normalized, weights=valid_episodes.mask)
                critic_loss = critic_criterion(values_normalized, returns)
                entropy = torch.mean(
                    - torch.exp(pi.log_prob(valid_episodes.actions)) * pi.log_prob(valid_episodes.actions))

                loss = actor_loss + critic_weight * critic_loss - entropy_wt * entropy
                losses.append(loss)

                if record_kls:
                    mask = valid_episodes.mask
                    if valid_episodes.actions.dim() > 2:
                        mask = mask.unsqueeze(2)
                    kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
                    kls.append(kl)

        return torch.mean(torch.stack(losses, dim=0)), pis, kls

    def surrogate_loss(self, episodes, eps_clip=0.1,critic_weight=0.5,entropy_wt=0.01, old_pis=None, record_kls=False):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)
        critic_criterion = nn.MSELoss()

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):

            self.policy.reset_context()
            params, _ = self.adapt(train_episodes)

            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                returns = valid_episodes.returns
                returns = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + 1e-5)

                values = self.baseline(valid_episodes)


                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                             - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss1 = ratio * advantages
                loss2 = torch.clamp(ratio, 1 - eps_clip, eps_clip) * advantages

                actor_loss = -torch.min(loss1, loss2).mean()

                values_normalized = values.squeeze(2)
                values_normalized = weighted_normalize(values_normalized, weights=valid_episodes.mask)
                critic_loss = critic_criterion(values_normalized, returns)
                entropy = torch.mean(- torch.exp(pi.log_prob(valid_episodes.actions))*pi.log_prob(valid_episodes.actions))

                loss = actor_loss + critic_weight*critic_loss - entropy_wt*entropy
                losses.append(loss)

                if record_kls:
                    mask = valid_episodes.mask
                    if valid_episodes.actions.dim() > 2:
                        mask = mask.unsqueeze(2)
                    kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
                    kls.append(kl)

        return torch.mean(torch.stack(losses, dim=0)), pis, kls

    def surrogate_loss_time_task(self, args, episodes, eps_clip=0.1,critic_weight=0.5, entropy_wt=0.01,old_pis=None):
        '''外循环损失最后一个任务的查询集的损失'''
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)
        critic_criterion = nn.MSELoss()

        count_task = 0
        params = None

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            if count_task == 0:
                params = None
            params, _ = self.adapt(train_episodes, params=params)

            if count_task == args.task_chain_len - 1:
                with torch.set_grad_enabled(old_pi is None):
                    pi = self.policy(valid_episodes.observations, params=params)
                    pis.append(detach_distribution(pi))

                    if old_pi is None:
                        old_pi = detach_distribution(pi)

                    returns = valid_episodes.returns
                    returns = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + 1e-5)

                    values = self.baseline(valid_episodes)

                    advantages = valid_episodes.gae(values, tau=self.tau)
                    advantages = weighted_normalize(advantages,
                                                    weights=valid_episodes.mask)

                    log_ratio = (pi.log_prob(valid_episodes.actions)
                                 - old_pi.log_prob(valid_episodes.actions))
                    if log_ratio.dim() > 2:
                        log_ratio = torch.sum(log_ratio, dim=2)
                    ratio = torch.exp(log_ratio)

                    loss1 = ratio * advantages
                    loss2 = torch.clamp(ratio, 1 - eps_clip, eps_clip) * advantages

                    actor_loss = -torch.min(loss1, loss2).mean()

                    values_normalized = values.squeeze(2)
                    values_normalized = weighted_normalize(values_normalized, weights=valid_episodes.mask)
                    critic_loss = critic_criterion(values_normalized, returns)
                    entropy = torch.mean(
                        - torch.exp(pi.log_prob(valid_episodes.actions)) * pi.log_prob(valid_episodes.actions))
                    loss = actor_loss + critic_weight * critic_loss - entropy_wt * entropy
                    losses.append(loss)
                count_task = 0

            else:
                pis.append(None)
                count_task = count_task + 1

        return torch.mean(torch.stack(losses, dim=0)), pis


    def step_time_task(self, args, episodes, meta_lr, eps_clip=0.1, critic_weight=0.5, entropy_wt=0.01, ls_max_steps=15,
                    cg_damping=1e-2, cg_iters=10, max_kl=1e-3,):
        old_loss, old_pis = self.surrogate_loss_time_task(args, episodes, eps_clip=eps_clip,
                                                          critic_weight=critic_weight, entropy_wt=entropy_wt)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        hessian_vector_product = self.hessian_vector_product_task_time(args, episodes, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir).float())
        lagrange_multiplier = torch.sqrt(abs(shs / max_kl))
        step = stepdir / lagrange_multiplier

        old_params = parameters_to_vector(self.policy.parameters())
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - meta_lr * step, self.policy.parameters())
            loss, _ = self.surrogate_loss_time_task(args, episodes, eps_clip=eps_clip, critic_weight=critic_weight,
                                                    entropy_wt=entropy_wt, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0):
                break
            meta_lr = meta_lr * 0.5
        else:
            print('no update?')
            vector_to_parameters(old_params, self.policy.parameters())
        return old_loss

    def step_origin(self, episodes, meta_lr, eps_clip=0.1, critic_weight=0.5, entropy_wt=0.01, ls_max_steps=15,
                    cg_damping=1e-2, cg_iters=10, max_kl=1e-3, ):
        """Meta Optimization Step for PPO"""
        old_loss, old_pis, old_kls = self.surrogate_loss(episodes, eps_clip=eps_clip, critic_weight=critic_weight,
                                                entropy_wt=entropy_wt, record_kls=True)
        grads = torch.autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)
        grads = parameters_to_vector(grads)

        old_kl = torch.mean(torch.stack(old_kls, dim=0))

        hessian_vector_product = self.hessian_vector_product(old_kl, damping=cg_damping)

        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir).float())
        lagrange_multiplier = torch.sqrt(abs(shs / max_kl))
        step = stepdir / lagrange_multiplier

        old_params = parameters_to_vector(self.policy.parameters())
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - meta_lr * step, self.policy.parameters())
            loss, _, _ = self.surrogate_loss(episodes, eps_clip=eps_clip, critic_weight=critic_weight,
                                          entropy_wt=entropy_wt, old_pis=old_pis, record_kls=False)
            improve = loss - old_loss
            if (improve.item() < 0.0):
                break
            meta_lr = meta_lr * 0.5
        else:
            print('no update?')
            vector_to_parameters(old_params, self.policy.parameters())
        return old_loss

    def step_time(self, args, episodes, meta_lr, eps_clip=0.1, critic_weight=0.5, entropy_wt=0.01, ls_max_steps=15,
                    cg_damping=1e-2, cg_iters=10, max_kl=1e-3, ):
        """Meta Optimization Step for PPO"""

        old_loss, old_pis = self.surrogate_loss_time(args, episodes, eps_clip=eps_clip, critic_weight=critic_weight,
                                                entropy_wt=entropy_wt)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        hessian_vector_product = self.hessian_vector_product_time(episodes, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir).float())
        lagrange_multiplier = torch.sqrt(abs(shs / max_kl))
        step = stepdir / lagrange_multiplier

        old_params = parameters_to_vector(self.policy.parameters())
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - meta_lr * step, self.policy.parameters())
            loss, _ = self.surrogate_loss_time(args, episodes, eps_clip=eps_clip, critic_weight=critic_weight,
                                          entropy_wt=entropy_wt, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0):
                break
            meta_lr = meta_lr * 0.5
        else:
            print('no update?')
            vector_to_parameters(old_params, self.policy.parameters())


        return old_loss


    def step_time_sequence(self, episodes, meta_lr, eps_clip=0.1, critic_weight=0.5, entropy_wt=0.01, ls_max_steps=15,
                    cg_damping=1e-2, cg_iters=10, max_kl=1e-3, ):
        """Meta Optimization Step for PPO"""

        old_loss, old_pis, old_kls = self.surrogate_loss_time_sequence(episodes, eps_clip=eps_clip, critic_weight=critic_weight,
                                                entropy_wt=entropy_wt, record_kls=True)
        grads = torch.autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)
        grads = parameters_to_vector(grads)

        old_kl = torch.mean(torch.stack(old_kls, dim=0))

        hessian_vector_product = self.hessian_vector_product_time_sequence(old_kl, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir).float())
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = stepdir / lagrange_multiplier

        old_params = parameters_to_vector(self.policy.parameters())
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - meta_lr * step, self.policy.parameters())
            loss, _, _ = self.surrogate_loss_time_sequence(episodes, eps_clip=eps_clip, critic_weight=critic_weight,
                                          entropy_wt=entropy_wt, old_pis=old_pis, record_kls=False)
            improve = loss - old_loss
            if (improve.item() < 0.0):
                break
            meta_lr = meta_lr * 0.5
        else:
            print('no update?')
            vector_to_parameters(old_params, self.policy.parameters())

        return old_loss

    def step_time_sequence_offpolicy(self, args, episodes, meta_lr, eps_clip=0.1, critic_weight=0.5, entropy_wt=0.01,
                                     ls_max_steps=15, cg_damping=1e-2, cg_iters=10, max_kl=1e-3):
        """Meta Optimization Step for PPO"""
        old_loss, old_pis, old_kls = self.surrogate_loss_time_sequence_offpolicy(args, episodes, eps_clip=eps_clip,
                                                                        critic_weight=critic_weight,
                                                                        entropy_wt=entropy_wt,
                                                                        record_kls=True)
        grads = torch.autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)
        grads = parameters_to_vector(grads)

        old_kl = torch.mean(torch.stack(old_kls, dim=0))

        hessian_vector_product = self.hessian_vector_product_time_sequence_offpolicy(old_kl, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir).float())
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = stepdir / lagrange_multiplier

        old_params = parameters_to_vector(self.policy.parameters())
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - meta_lr * step, self.policy.parameters())
            loss, _, _ = self.surrogate_loss_time_sequence_offpolicy(args, episodes, eps_clip=eps_clip,
                                                                  critic_weight=critic_weight, entropy_wt=entropy_wt,
                                                                  old_pis=old_pis, record_kls=False)
            improve = loss - old_loss
            if (improve.item() < 0.0):
                break
            meta_lr = meta_lr * 0.5
        else:
            print('no update?')
            vector_to_parameters(old_params, self.policy.parameters())


        return old_loss

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device