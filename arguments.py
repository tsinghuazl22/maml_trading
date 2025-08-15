import argparse
import multiprocessing as mp
import os
import warnings

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Fast Context Adaptation via Meta-Learning (CAVIA)')





    parser.add_argument("--set_bI", type=list, default=[-5, -4], help="default value for bI")
    parser.add_argument("--hidden", type=int, default=20, help="default value for bI")
    parser.add_argument("--moving-window", type=int, default=240,
                        help="normalize moving window")
    parser.add_argument("--task_chain_len", type=int, default=4,
                        help="task chain length for considering task relationship")

    parser.add_argument("--reward-shorten", type=float, default=1,
                        help="reward shorten rate")
    parser.add_argument("--name", type=str, default="DJI",
                        help="name")

    parser.add_argument("--fee-time", type=int, default=1,
                        help="times of the normal fee")

    parser.add_argument("--shuffle", type=bool, default=False,
                        help="whether shuffle the support data")

    parser.add_argument('--meta-batch-size', type=int, default=10,
                        help='number of tasks per batch')


    parser.add_argument('--fast-batch-size', type=int, default=10,
                        help='number of rollouts for each individual task (episode)')
    parser.add_argument('--inner_update_nums', type=int, default=10,
                        help='update times for the inner loop')

    parser.add_argument('--episode_window', type=int, default=20, help="an episode length")


    parser.add_argument('--train_episode_num', type=int, default=96, help="num of episodes in training data")

    parser.add_argument('--val_episode_num', type=int, default=24, help="num of episodes in val data")

    parser.add_argument('--test_episode_num', type=int, default=24, help="num of episodes in testing data")



    parser.add_argument('--task_episode_num', type=int, default=4, help="num of episodes in a task")

    parser.add_argument('--support_episode_num', type=int, default=3, help="num of episodes in a support data")

    parser.add_argument('--maml', action='store_true', default=True,
                        help='turn on MAML')

    parser.add_argument('--num-batches', type=int, default=8000,
                        help='number of batches')

    parser.add_argument('--num-test-steps', type=int, default=1,
                        help='Number of inner loops in the test set')

    parser.add_argument('--test-batch-size', type=int, default=10,
                        help='batch size (number of trajectories) for testing')
    parser.add_argument('--halve-test-lr', action='store_true', default=True,
                        help='half LR at test time after one update')
    parser.add_argument('--select_model_num', type=int, default=11496)


    parser.add_argument('--env-name', type=str,
                        default='Trading-v0',
                        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=0.9,
                        help='value of the discount factor for GAE')
    parser.add_argument('--first_order', action='store_true',
                        help='use the first-order approximation of MAML/CAVIA')
    parser.add_argument('--num-context-params', type=int, default=2,
                        help='number of context parameters')





    parser.add_argument('--hidden-size', type=int, default=100,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')














    parser.add_argument('--fast-lr', type=float, default=0.1,
                        help='learning rate for the 1-step gradient update of MAML/CAVIA')


    parser.add_argument('--meta-lr', type=float, default=0.01, help='To be used only for PPO')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='Clipping for the PPO objective')
    parser.add_argument('--critic_weight', type=float, default=0.5,
                        help='Weight for the critic loss term in the multi-loss function')
    parser.add_argument('--entropy_wt', type=float, default=0.01,
                        help='Weight for entropy term in loss function')






    parser.add_argument('--max-kl', type=float, default=1e-2,
                        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
                        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
                        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
                        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
                        help='maximum number of iterations for line search')


    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--make_deterministic', action='store_true',
                        help='make everything deterministic (set cudnn seed; num_workers=1; '
                             'will slow things down but make them reproducible!)')
    parser.add_argument('--restore_model', type=str, help='Path to saved policy')

    args = parser.parse_args()

    if args.make_deterministic:
        args.num_workers = 1



    args.device = torch.device("cpu")

    args.output_folder = 'maml' if args.maml else 'cavia'





    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')


    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    return args
