"""
@Time:2022.12.06
@desc:支撑集将参数更新,在查询集评估表现
输出:查询集仓位曲线、结果存储csv

新增功能2022.12.06 20:22：每个episode最后一个时间点也采取动作
新增功能2022.12.7 17:14: last_episode_action记录一幕最后一个动作,作为在下一幕的第一个点的上一个动作
修改2022.12.21:测试时使用串行环境
----------------------------
修改2023.02.15：添加函数test_trading_time_offpolicy,在考虑样本关系时，不同episode都使用初始策略参数采样,
因此从第2个episode开始存在离轨策略参数更新
"""
import json
from arguments import parse_args
from sampler import BatchSampler
from metalearner_ppo import MetaLearner as MetaLearner
import os
from preprocessing.preprocessors import preprocess_data
from policies.categorical_mlp import CategoricalMLPPolicy
from policies.normal_mlp import Cavia_CategoricalMLPPolicy
from baseline import LinearFeatureBaseline
import torch
import scipy.stats as st
import numpy as np

import matplotlib.pyplot as plt
import datetime
import matplotlib.font_manager as fm
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.ticker import MaxNLocator
import pandas as pd
import utils

plt.rcParams['axes.unicode_minus'] = False

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

def str2date(strdate):
    return datetime.datetime.strptime(strdate[0:4]+"-"+strdate[4:6]+"-"+strdate[6:8],
                                      '%Y-%m-%d')

def load_train_config(base_save_dir, args):
    train_config_file = open(os.path.join(base_save_dir, 'config.json'), 'r')
    train_config = json.load(train_config_file)
    args.episode_window = train_config["episode_window"]


    args.task_episode_num = train_config["task_episode_num"]
    args.support_episode_num = train_config["support_episode_num"]
    args.gamma = train_config["gamma"]
    args.tau = train_config["tau"]

    args.fast_lr = train_config["fast_lr"]
    args.deter_sample_flag = train_config["deter_sample_flag"]
    args.fast_batch_size = train_config["fast_batch_size"]
    args.meta_lr = train_config["meta_lr"]
    args.shuffle = train_config["shuffle"]



def test_for_trading_new():
    mode = "time"
    args = parse_args()

    args.maml = True

    if args.name == "DJI":
        test_range = [20201127, 20221127]
    else:
        test_range = [20201127, 20221127]

    utils.set_seed(args.seed, cudnn=args.make_deterministic)



    base_dir = "XXXXXXXXXX"

    file_name = "XXXXXXXXXXXXXXXXX"

    load_train_config(os.path.join(base_dir, "saves_fixed_rate_time", file_name), args)
    base_model_dir = os.path.join(base_dir, "saves_fixed_rate_time", file_name,
                                  "policy-" + str(args.select_model_num) + ".pt")

    deter_sample_flag = args.deter_sample_flag
    shuffle_flag = args.shuffle




    deter_str = "deter_" if args.deter_sample_flag == True else "rand_"
    shuffle_str = "shuffle" if shuffle_flag else ""
    mlr_str = "1.0" if str(args.meta_lr) == 1.0 else str(args.meta_lr).replace(".", "")
    fast_str = "1.0" if str(args.fast_lr) == 1.0 else str(args.fast_lr).replace(".", "")

    out_name = mode + shuffle_str + "_" + deter_str + "mlr" + mlr_str + "_fast" + fast_str + "_" + args.name + "_" + \
               str(args.select_model_num) + "_step" + str(args.num_test_steps)

    x_distance = 20




    result_path = os.path.join(base_dir, "trade_result", mode)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    preprocessed_path = "done_data_" + args.name + ".csv"
    if os.path.exists(preprocessed_path):
        print("已存在,加载数据")
        data = pd.read_csv(preprocessed_path)

        data['historySpread'] = data['historySpread'].apply(lambda history_list: eval(history_list))
        data['historySpreadNormalized'] = data['historySpreadNormalized'].apply(
            lambda history_list: eval(history_list))
    else:
        print("未存在,处理数据")
        data = preprocess_data("continuousdata_" +args.name+"_day.csv")
        data.to_csv(preprocessed_path)

    cur_test_data = data[(data.datadate >= test_range[0]) & (data.datadate <= test_range[1])]

    support_for_first_episode = data[data.datadate < test_range[0]][-args.support_episode_num * args.episode_window:]
    test_data = pd.concat([support_for_first_episode, cur_test_data])
    test_data.index = test_data.datadate.factorize()[0]

    if len(test_data) % args.episode_window <= 1:

        test_episode_num = int(len(test_data) / args.episode_window) - args.support_episode_num
        last_episode_len = args.episode_window
    else:
        test_episode_num = int(len(test_data) / args.episode_window) + 1 - args.support_episode_num
        last_episode_len = len(test_data) % args.episode_window


    range_lst = [i for i in range(test_episode_num)]



    sampler = BatchSampler(data, args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers,
                           device=args.device, seed=args.seed, episode_window=args.episode_window)

    if args.maml:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(128, 32))
    else:
        policy = Cavia_CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.n)),
            hidden_sizes=(128, 32),
            num_context_params=args.num_context_params,
            device=args.device
        )


    policy.load_state_dict(torch.load(base_model_dir, map_location=torch.device('cpu')))

    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              device=args.device)

    all_date_lst = []
    all_price_lst = []
    all_action_lst = []
    all_reward_lst = [0]


    last_episode_action = 0

    sampler.close()

    for test_num in range_lst:
        print("====task_num{0}=====".format(test_num))
        cur_index = args.episode_window * (args.support_episode_num) + args.episode_window * test_num

        if test_num == range_lst[-1]:

            task_data = test_data.iloc[cur_index - args.episode_window * args.support_episode_num:
                                       cur_index + last_episode_len]
        else:

            task_data = test_data.iloc[cur_index - args.episode_window * args.support_episode_num:
                                       cur_index + args.episode_window * (
                                               args.task_episode_num - args.support_episode_num) + 1]

        task_data.index = task_data.datadate.factorize()[0]

        if mode == "entire":
            actions, rewards = \
                metalearner.test_trading(args, task_data=task_data, num_steps=args.num_test_steps,
                                         batch_size=args.test_batch_size, halve_lr=args.halve_test_lr,
                                         last_episode_action=last_episode_action,
                                         deter_sample=deter_sample_flag)
        elif mode == "time":
            actions, rewards = \
                metalearner.test_trading_time(args, task_data=task_data, num_steps=args.num_test_steps,
                                              batch_size=args.test_batch_size, halve_lr=args.halve_test_lr,
                                              last_episode_action=last_episode_action, deter_sample=deter_sample_flag,
                                              shuffle=shuffle_flag)













        elif mode == "time_off":
            actions, rewards = \
                metalearner.test_trading_time_offpolicy(args, task_data=task_data, num_steps=args.num_test_steps,
                                              batch_size=args.test_batch_size, halve_lr=args.halve_test_lr,
                                              last_episode_action=last_episode_action)
        else:
            print("input illegality")



        if test_num == range_lst[-1]:
            valid_data = task_data.iloc[args.episode_window * args.support_episode_num:
                                        args.episode_window * args.support_episode_num + last_episode_len]
        else:

            valid_data = task_data.iloc[args.episode_window * args.support_episode_num:-1]
        valid_data.index = valid_data.datadate.factorize()[0]


        cur_datadate_lst = list(valid_data.datadate)
        all_date_lst = all_date_lst + cur_datadate_lst

        cur_price_lst = list(valid_data.close)
        all_price_lst = all_price_lst + cur_price_lst

        cur_actions_lst = [int(action) - 1 for action in actions]
        all_action_lst = all_action_lst + cur_actions_lst
        last_episode_action = cur_actions_lst[-1]



        cur_reward_lst = [reward.item()/(args.reward_shorten) for reward in rewards]
        print("return:", np.sum(cur_reward_lst))

        all_reward_lst = all_reward_lst + cur_reward_lst



    all_action_lst.append(0)

    all_sum = 0
    accumulate_profit = []
    for ii in all_reward_lst:
        all_sum += ii
        accumulate_profit.append(all_sum)
    print("acc_sum:", accumulate_profit[-1])

    buy_and_hold_lst = [price - all_price_lst[0] for price in all_price_lst]



    print("accumulate_profit:", accumulate_profit[-1])

    time = [str2date(str(item)) for item in all_date_lst]


    time_plot = []
    n_index = [0]
    for tt in time:
        time_plot.append(str(tt.year) + '-' + str(tt.month) + '-' + str(tt.day))
    for i in range(len(time_plot) - 1):
        if time_plot[i] != time_plot[i + 1]:
            n_index.append(i + 1)
    time_index = [time_plot[n] for n in n_index]

    if x_distance > 0:
        if len(n_index) % x_distance > x_distance / 2 or len(n_index) % x_distance == 0:
            n_index = [n_index[n] for n in range(0, len(n_index), x_distance)] + [n_index[-1]]
            time_index = [time_index[n] for n in range(0, len(time_index), x_distance)] + [time_index[-1]]
        else:
            n_index = [n_index[n] for n in range(0, len(n_index), x_distance)][:-1] + [n_index[-1]]
            time_index = [time_index[n] for n in range(0, len(time_index), x_distance)][:-1] + [time_index[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    plt11 = axes[0].plot(accumulate_profit, '-')
    plt12 = axes[0].plot(buy_and_hold_lst, "-")
    axes[0].legend([plt11, plt12], labels=['acc profit', 'buy and hold'], loc='best')
    axes[0].set_xticks(n_index)
    axes[0].set_xticklabels(time_index, rotation=80)
    axes[0].grid(True, linestyle="-.", color="k", linewidth="0.3", axis='x')
    axes[0].set_ylabel("profits")

    plt2 = axes[1].plot(all_action_lst, '.')
    axes[1].legend([plt2], labels=['action'], loc='best')
    axes[1].set_xticks(n_index)
    axes[1].set_xticklabels(time_index, rotation=80)
    axes[1].grid(True, linestyle="-.", color="k", linewidth="0.3", axis='x')
    axes[1].set_ylabel("action")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gcf().autofmt_xdate()

    plt.savefig(os.path.join(result_path, out_name + ".png"), dpi=300, bbox_inches='tight')


    dct = {"time":time,
           "close":all_price_lst,
           "action":all_action_lst,
           "profit":all_reward_lst,
           "acc_profit":accumulate_profit
           }
    frame = pd.DataFrame(dct)
    frame.to_csv(os.path.join(result_path, out_name+".csv"), index=None, encoding='utf_8_sig')



if __name__ == "__main__":
    test_for_trading_new()
