import datetime
import json
import os
import matplotlib.pyplot as plt
import time

import numpy as np
import scipy.stats as st
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from tensorboardX import SummaryWriter

import utils
from arguments import parse_args
from baseline import LinearFeatureBaseline
from metalearner_ppo import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy

from sampler import BatchSampler
import pandas as pd
from preprocessing.preprocessors import preprocess_data
import random

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

def total_rewards2(episodes_per_task, interval=False):
    all_task_returns = []
    for task_idx in range(len(episodes_per_task)):
        each_update_returns = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            cur_episodes = episodes[update_idx]
            returns_each_traj = cur_episodes.rewards.sum(dim=0)
            cur_mean_returns = torch.mean(returns_each_traj)
            each_update_returns.append(cur_mean_returns)

        all_task_returns.append(torch.stack(each_update_returns))
    all_returns = torch.stack(all_task_returns, dim=0)
    return torch.mean(all_returns, dim=0)





def main_criticbaseline(args):
    from baseline import ValueNetworkBaseLine, ValueNetworkBaseLine_timevary
    print('starting....')
    utils.set_seed(args.seed, cudnn=args.make_deterministic)

    args.maml = True
    if args.name == "DJI":
        args.train_range = [20130815, 20201127]

        args.test_range = [20201130, 20220904]
    else:
        args.train_range = [20130107, 20201126]
        args.test_range = [20201127, 20221130]

    args.deter_sample_flag = True


    method_used = 'maml' if args.maml else 'cavia'
    num_context_params = str(args.num_context_params) + '_' if not args.maml else ''
    output_name = "ppo_SZ50_metabatch30_gamma099_lambda09"
    output_name += '_' + datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_folder = os.path.join(os.path.join(dir_path, 'logs'), args.env_name, method_used, output_name)
    save_folder = os.path.join(os.path.join(dir_path, 'saves'), output_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    writer = SummaryWriter(log_folder)

    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)
    with open(os.path.join(log_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)
    preprocessed_path = "done_data_" + args.name + ".csv"
    if os.path.exists(preprocessed_path):
        print("已存在,加载数据")
        data = pd.read_csv(preprocessed_path)
        data['historySpread'] = data['historySpread'].apply(lambda history_list: eval(history_list))
        data['historySpreadNormalized'] = data['historySpreadNormalized'].apply(lambda history_list: eval(history_list))
    else:
        print("未存在,处理数据")
        data = preprocess_data("continuousdata_" + args.name + "_day.csv")
        data.to_csv(preprocessed_path)

    train_data = data[(data.datadate >= args.train_range[0]) & (data.datadate <= args.train_range[1])]
    train_data.index = train_data.datadate.factorize()[0]
    train_episode_num = int(len(train_data) // args.episode_window)
    train_task_num = train_episode_num - args.task_episode_num + 1
    train_task_start_index_lst = [i * args.episode_window for i in range(train_task_num)]

    support_data_for_first_episode = train_data[-args.episode_window * args.support_episode_num:]
    cur_test_data = data[(data.datadate >= args.test_range[0]) & (data.datadate <= args.test_range[1])]
    test_data = pd.concat([support_data_for_first_episode, cur_test_data], axis=0)
    test_data.index = test_data.datadate.factorize()[0]
    if len(test_data) % args.episode_window >= args.episode_window // 2:
        test_episode_num = len(test_data) // args.episode_window + 1
    else:
        test_episode_num = len(test_data) // args.episode_window
    test_task_num = test_episode_num - args.support_episode_num
    test_task_start_index_lst = [i * args.episode_window for i in range(test_task_num)]

    episode_window = args.episode_window

    sampler = BatchSampler(data, args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers,
                           device=args.device, seed=args.seed, episode_window=args.episode_window)

    policy = CategoricalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        sampler.envs.action_space.n,
        hidden_sizes=(128, 32))


    baseline = ValueNetworkBaseLine_timevary(input_size=int(np.prod(sampler.envs.observation_space.shape)),
                                    hidden_size=(128, 32),
                                    output_size=1)

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              device=args.device)

    meta_batch_size = args.meta_batch_size

    sampler.close()
    for batch in range(args.num_batches):
        if meta_batch_size >= len(train_task_start_index_lst):
            train_sample_lst = train_task_start_index_lst
        else:
            train_sample_lst = random.sample(train_task_start_index_lst, meta_batch_size)
        print(train_sample_lst)
        tasks = sampler.sample_data_tasks(train_data, train_sample_lst, episode_window)

        episodes, inner_losses = metalearner.sample_origin(args, tasks, first_order=args.first_order,
                                                           deter_sample=args.deter_sample_flag)
        outer_loss = metalearner.step_origin(episodes, args.meta_lr, eps_clip=args.eps_clip,
                                             critic_weight=args.critic_weight, entropy_wt=args.entropy_wt,
                                             ls_max_steps=args.ls_max_steps)

        curr_returns = total_rewards(episodes, interval=True)
        print('   return after update: ', curr_returns[0][1])

        writer.add_scalar('policy/actions_train', episodes[0][0].actions.mean(), batch)
        writer.add_scalar('policy/actions_test', episodes[0][1].actions.mean(), batch)

        writer.add_scalar('running_returns/before_update', curr_returns[0][0], batch)
        writer.add_scalar('running_returns/after_update', curr_returns[0][1], batch)

        writer.add_scalar('loss/inner_rl', np.mean(inner_losses), batch)
        writer.add_scalar('loss/outer_rl', outer_loss.item(), batch)

        if batch % 4 == 0:
            val_sample_lst = test_task_start_index_lst
            val_tasks = sampler.sample_data_tasks(test_data, val_sample_lst, episode_window)
            quary_episodes_per_task = metalearner.test_muti_times(args, tasks=val_tasks,
                                                                  num_steps=args.num_test_steps,
                                                                  batch_size=args.fast_batch_size,
                                                                  halve_lr=args.halve_test_lr,
                                                                  deter_sample=args.deter_sample_flag)
            quary_set_returns = total_rewards(quary_episodes_per_task, interval=False)
            for num in range(len(quary_set_returns)):
                writer.add_scalar('evaluation_rew/avg_rew' + str(num), quary_set_returns[num], batch)

        print("=======iter_num_{0}=======".format(batch))

        if batch % 4 == 0 and batch >= 200:
            with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
                torch.save(policy.state_dict(), f)

    sampler.close()


def load_train_config(base_save_dir, args):
    train_config_file = open(os.path.join(base_save_dir, 'config.json'), 'r')
    train_config = json.load(train_config_file)
    args.seed = train_config["seed"]
    args.episode_window = train_config["episode_window"]
    args.task_episode_num = train_config["task_episode_num"]
    args.support_episode_num = train_config["support_episode_num"]
    args.gamma = train_config["gamma"]
    args.tau = train_config["tau"]
    args.meta_batch_size = train_config["meta_batch_size"]
    args.fast_lr = train_config["fast_lr"]
    args.fast_batch_size = train_config["fast_batch_size"]
    args.deter_sample_flag = train_config["deter_sample_flag"]
    args.meta_batch_size = train_config["meta_batch_size"]
    args.meta_lr = train_config["meta_lr"]


def train_continue(args):
    start_iter_num = 7996
    end_iter_num = 20000

    args.maml = True
    if args.name == "DJI":
        args.train_range = [20130815, 20201127]
        args.test_range = [20201130, 20220904]
    else:
        args.train_range = [20130107, 20201126]
        args.test_range = [20201127, 20221118]
    save_name = "mlr001_rand_fast001_entire_CSI500_metabatch30_gamma099_lambda09_04_04_2023_16_44_54"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    method_used = 'maml' if args.maml else 'cavia'
    log_folder = os.path.join(os.path.join(dir_path, 'logs_fixed_rate'), args.env_name, method_used,
                              save_name)
    save_folder = os.path.join(os.path.join(dir_path, 'saves_fixed_rate'), save_name)
    base_model_dir = os.path.join(save_folder, "policy-" + str(start_iter_num) + ".pt")
    writer = SummaryWriter(log_folder)

    load_train_config(log_folder, args)
    utils.set_seed(args.seed, cudnn=args.make_deterministic)

    writer = SummaryWriter(log_folder)

    preprocessed_path = "done_data_" + args.name + ".csv"
    if os.path.exists(preprocessed_path):
        print("已存在,加载数据")
        data = pd.read_csv(preprocessed_path)
        data['historySpread'] = data['historySpread'].apply(lambda history_list: eval(history_list))
        data['historySpreadNormalized'] = data['historySpreadNormalized'].apply(lambda history_list: eval(history_list))
    else:
        print("未存在,处理数据")
        data = preprocess_data("continuousdata_" + args.name + "_day.csv")
        data.to_csv(preprocessed_path)

    train_data = data[(data.datadate >= args.train_range[0]) & (data.datadate <= args.train_range[1])]
    train_data.index = train_data.datadate.factorize()[0]
    train_episode_num = int(len(train_data) // args.episode_window)
    train_task_num = train_episode_num - args.task_episode_num + 1
    train_task_start_index_lst = [i * args.episode_window for i in range(train_task_num)]

    support_data_for_first_episode = train_data[-args.episode_window * args.support_episode_num:]
    cur_test_data = data[(data.datadate >= args.test_range[0]) & (data.datadate <= args.test_range[1])]
    test_data = pd.concat([support_data_for_first_episode, cur_test_data], axis=0)
    test_data.index = test_data.datadate.factorize()[0]
    if len(test_data) % args.episode_window >= args.episode_window // 2:
        test_episode_num = len(test_data) // args.episode_window + 1
    else:
        test_episode_num = len(test_data) // args.episode_window
    test_task_num = test_episode_num - args.support_episode_num
    test_task_start_index_lst = [i * args.episode_window for i in range(test_task_num)]

    episode_window = args.episode_window

    sampler = BatchSampler(data, args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers,
                           device=args.device, seed=args.seed, episode_window=args.episode_window)

    policy = CategoricalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        sampler.envs.action_space.n,
        hidden_sizes=(128, 32))

    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))
    policy.load_state_dict(state_dict=torch.load(base_model_dir, map_location=torch.device('cpu')))

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              device=args.device)

    meta_batch_size = args.meta_batch_size

    sampler.close()
    for batch in range(start_iter_num, end_iter_num+1):
        if meta_batch_size >= len(train_task_start_index_lst):
            train_sample_lst = train_task_start_index_lst
        else:
            train_sample_lst = random.sample(train_task_start_index_lst, meta_batch_size)
        print(train_sample_lst)
        tasks = sampler.sample_data_tasks(train_data, train_sample_lst, episode_window)

        episodes, inner_losses = metalearner.sample_origin(args, tasks, first_order=args.first_order,
                                                           deter_sample=args.deter_sample_flag)
        outer_loss = metalearner.step_origin(episodes, args.meta_lr, eps_clip=args.eps_clip,
                                             critic_weight=args.critic_weight, entropy_wt=args.entropy_wt,
                                             ls_max_steps=args.ls_max_steps)

        curr_returns = total_rewards(episodes, interval=True)

        print('   return after update: ', curr_returns[0][1])

        writer.add_scalar('policy/actions_train', episodes[0][0].actions.mean(), batch)
        writer.add_scalar('policy/actions_test', episodes[0][1].actions.mean(), batch)

        writer.add_scalar('running_returns/before_update', curr_returns[0][0], batch)
        writer.add_scalar('running_returns/after_update', curr_returns[0][1], batch)

        writer.add_scalar('loss/inner_rl', np.mean(inner_losses), batch)
        writer.add_scalar('loss/outer_rl', outer_loss.item(), batch)

        if batch % 8 == 0:
            val_sample_lst = test_task_start_index_lst
            val_tasks = sampler.sample_data_tasks(test_data, val_sample_lst, episode_window)
            quary_episodes_per_task = metalearner.test_muti_times(args, tasks=val_tasks,
                                                                  num_steps=args.num_test_steps,
                                                                  batch_size=args.fast_batch_size,
                                                                  halve_lr=args.halve_test_lr,
                                                                  deter_sample=args.deter_sample_flag)
            quary_set_returns = total_rewards(quary_episodes_per_task, interval=False)
            for num in range(len(quary_set_returns)):
                writer.add_scalar('evaluation_rew/avg_rew' + str(num), quary_set_returns[num], batch)

        print("=======iter_num_{0}=======".format(batch))

        if batch % 8 == 0 and batch >= 3000:
            with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
                torch.save(policy.state_dict(), f)

    sampler.close()


def main_new(args):
    print('starting....')
    utils.set_seed(args.seed, cudnn=args.make_deterministic)

    args.maml = True
    if args.name == "DJI":
        args.train_range = [20130815, 20201127]
        args.test_range = [20201130, 20220904]
    else:
        args.train_range = [20130107, 20201126]
        args.test_range = [20201127, 20221118]

    args.deter_sample_flag = True


    method_used = 'maml' if args.maml else 'cavia'
    num_context_params = str(args.num_context_params) + '_' if not args.maml else ''
    fast_str = "1" if str(args.fast_lr) == "1.0" else str(args.fast_lr).replace(".","")
    if args.deter_sample_flag == True:
        deter_str = "deter_"
    else:
        deter_str = "rand_"
    output_name = deter_str+"mlr001_fast" + fast_str + "_entire_" + args.name + "_metabatch30_gamma099_lambda09"
    output_name += '_' + datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_folder = os.path.join(os.path.join(dir_path, 'logs_fixed_rate2'), args.env_name, method_used, output_name)
    save_folder = os.path.join(os.path.join(dir_path, 'saves_fixed_rate2'), output_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    writer = SummaryWriter(log_folder)

    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)
    with open(os.path.join(log_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)
    preprocessed_path = "done_data_" + args.name + ".csv"
    if os.path.exists(preprocessed_path):
        print("已存在,加载数据")
        data = pd.read_csv(preprocessed_path)
        data['historySpread'] = data['historySpread'].apply(lambda history_list: eval(history_list))
        data['historySpreadNormalized'] = data['historySpreadNormalized'].apply(lambda history_list: eval(history_list))
    else:
        print("未存在,处理数据")
        data = preprocess_data("F:\\aa_maml\CAVIA_Pytorch_RL_trade_day4\data\\continuousdata_" + args.name + "_day.csv")
        data.to_csv(preprocessed_path)

    train_data = data[(data.datadate >= args.train_range[0]) & (data.datadate <= args.train_range[1])]
    train_data.index = train_data.datadate.factorize()[0]
    train_episode_num = int(len(train_data) // args.episode_window)
    train_task_num = train_episode_num - args.task_episode_num + 1
    train_task_start_index_lst = [i * args.episode_window for i in range(train_task_num)]

    support_data_for_first_episode = train_data[-args.episode_window * args.support_episode_num:]
    cur_test_data = data[(data.datadate >= args.test_range[0]) & (data.datadate <= args.test_range[1])]
    test_data = pd.concat([support_data_for_first_episode, cur_test_data], axis=0)
    test_data.index = test_data.datadate.factorize()[0]
    if len(test_data) % args.episode_window >= args.episode_window // 2:
        test_episode_num = len(test_data) // args.episode_window + 1
    else:
        test_episode_num = len(test_data) // args.episode_window
    test_task_num = test_episode_num - args.support_episode_num
    test_task_start_index_lst = [i * args.episode_window for i in range(test_task_num)]

    episode_window = args.episode_window

    sampler = BatchSampler(data, args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers,
                           device=args.device, seed=args.seed, episode_window=args.episode_window)

    policy = CategoricalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        sampler.envs.action_space.n,
        hidden_sizes=(128, 32))

    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              device=args.device)

    meta_batch_size = args.meta_batch_size

    sampler.close()
    for batch in range(args.num_batches):
        start_time = time.time()
        if meta_batch_size >= len(train_task_start_index_lst):
            train_sample_lst = train_task_start_index_lst
        else:
            train_sample_lst = random.sample(train_task_start_index_lst, meta_batch_size)
        print(train_sample_lst)
        tasks = sampler.sample_data_tasks(args, train_data, train_sample_lst, episode_window)

        episodes, inner_losses = metalearner.sample_origin(args, tasks, first_order=args.first_order,
                                                           deter_sample=args.deter_sample_flag)
        outer_loss = metalearner.step_origin(episodes, args.meta_lr, eps_clip=args.eps_clip,
                                             critic_weight=args.critic_weight, entropy_wt=args.entropy_wt,
                                             ls_max_steps=args.ls_max_steps)
        print("cost {} seconds".format(time.time() - start_time))
        print("inner_loss", sum(inner_losses))

        curr_returns = total_rewards(episodes, interval=True)

        print('   return after update: ', curr_returns[0][1])

        writer.add_scalar('policy/actions_train', episodes[0][0].actions.mean(), batch)
        writer.add_scalar('policy/actions_test', episodes[0][1].actions.mean(), batch)

        writer.add_scalar('running_returns/before_update', curr_returns[0][0], batch)
        writer.add_scalar('running_returns/after_update', curr_returns[0][1], batch)

        writer.add_scalar('loss/inner_rl', np.mean(inner_losses), batch)
        writer.add_scalar('loss/outer_rl', outer_loss.item(), batch)

        if batch % 8 == 0:
            val_sample_lst = test_task_start_index_lst
            val_tasks = sampler.sample_data_tasks(args, test_data, val_sample_lst, episode_window)
            quary_episodes_per_task = metalearner.test_muti_times(args, tasks=val_tasks,
                                                                  num_steps=args.num_test_steps,
                                                                  batch_size=args.fast_batch_size,
                                                                  halve_lr=args.halve_test_lr,
                                                                  deter_sample=args.deter_sample_flag)
            quary_set_returns = total_rewards(quary_episodes_per_task, interval=False)
            print("quary_set_returns :{}".format(quary_set_returns))
            for num in range(len(quary_set_returns)):
                writer.add_scalar('evaluation_rew/avg_rew' + str(num), quary_set_returns[num], batch)

        print("=======iter_num_{0}=======".format(batch))

        if batch % 8 == 0 and batch >= 3000:
            with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
                torch.save(policy.state_dict(), f)

    sampler.close()



if __name__ == '__main__':
    args = parse_args()

    main_new(args)
