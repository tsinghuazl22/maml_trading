
import multiprocessing as mp

import gym
import torch

from envs.subproc_vec_env import SubprocVecEnv
from envs.trading_env import TradingEnv
from envs.test_trading_env import TestTradingEnv
from episode import BatchEpisodes, BatchEpisodes2
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv as SubprocVecEnv2



class BatchSampler(object):
    def __init__(self, data, env_name, batch_size, device, seed, episode_window, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.data = data
        self.episode_window = episode_window

        self.queue = mp.Queue()



        self.envs = DummyVecEnv([lambda: TradingEnv(df=self.data.iloc[0:2], episode_window=self.episode_window)])

        self.envs.seed(seed)





    def create_new_envs(self, seed, episode_data, episode_len):
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([lambda: TradingEnv(df=episode_data, episode_window=episode_len)
                                   for _ in range(self.num_workers)], queue=self.queue)
        self.envs.seed(seed)


    def create_test_envs(self, seed, episode_data, episode_len, last_episode_action):
        self.queue = mp.Queue()

        self.envs = SubprocVecEnv([lambda: TestTradingEnv(df=episode_data, episode_window=episode_len,
                                                          last_episode_action=last_episode_action)], queue=self.queue)
        self.envs.seed(seed)


    def create_new_DummyVecEnv_envs(self, seed, episode_data, episode_len):
        self.envs = DummyVecEnv([lambda: TradingEnv(df=episode_data, episode_window=episode_len)])
        self.envs.seed(seed)

    def create_new_DummyVecEnv_envs_lst(self, seed, episode_data_lst, episode_len):
        vec_env_lst = []
        for j in range(len(episode_data_lst)):
            cur_env = lambda: TradingEnv(df=episode_data_lst[j], episode_window=episode_len)
            vec_env_lst.append(cur_env)
        self.envs = DummyVecEnv(vec_env_lst)
        self.envs.seed(seed)

    def create_new_SummyVecEnv_envs_lst(self, seed, episode_data_lst, episode_len):
        self.envs = SubprocVecEnv2([lambda: TradingEnv(df=episode_data, episode_window=episode_len)
                                 for episode_data in episode_data_lst])
        self.envs.seed(seed)


    def create_test_DummyVecEnv_envs(self, seed, episode_data, episode_len, last_episode_action):
        self.envs = DummyVecEnv([lambda: TestTradingEnv(df=episode_data, episode_window=episode_len,
                                                        last_episode_action=last_episode_action)])
        self.envs.seed(seed)



    def close(self):
        if hasattr(self.envs,"workers") == True:
            for worker in self.envs.workers:
                worker.terminate()
        else:
            pass


    def judge_run(self):
        if hasattr(self.envs, "workers") == True:
            for worker in self.envs.workers:
                if worker.is_alive():
                    return True
            return False
        else:
            return False


    def sample_for_test(self, policy, params=None, gamma=0.95, batch_size=None, deterministic=True):
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(1):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):


            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)

                actions_tensor = policy(observations_tensor,
                                        params=params).sample(deterministic=deterministic)
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def sample_rebanlance(self, policy, params=None, gamma=0.95, batch_size=None, deterministic=False,
                          rebanlance_window=240):
        episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        time_steps = 0

        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample(deterministic=deterministic)
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            time_steps = time_steps + 1
            if time_steps == rebanlance_window:
                dones = [True] * self.num_workers
            observations, batch_ids = new_observations, new_batch_ids
        return episodes


    def sample(self, policy, params=None, gamma=0.95, batch_size=None, deterministic=False):
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):


            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample(deterministic=deterministic)
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def sample_one_epsiode(self, traj_num, episodes, params, policy,  deterministic):
        dones = False
        observations = self.envs.reset()
        while not dones:
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample(deterministic=deterministic)
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, traj_num)
            observations, batch_ids = new_observations, traj_num


    def sample_dummy_lst(self, policy, params=None, gamma=0.95, batch_size=None, deterministic=False, episode_data_num=1):
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodes(batch_size=batch_size*episode_data_num, gamma=gamma, device=self.device)
        for traj_num in range(batch_size):
            dones = [False for _ in range(episode_data_num)]
            observations = self.envs.reset()
            batch_ids = [traj_num+tt*batch_size for tt in range(episode_data_num)]
            while not all(dones):
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample(deterministic=deterministic)
                actions = actions_tensor.cpu().numpy()

                new_observations, rewards, dones, info = self.envs.step(actions)
                episodes.append(observations, actions, rewards, batch_ids)
                observations, = new_observations,
        return episodes



    def sample_dummy(self, policy, params=None, gamma=0.95, batch_size=None, deterministic=False):
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodes2(batch_size=batch_size, gamma=gamma, device=self.device)

        for traj_num in range(batch_size):
            dones = False
            observations = self.envs.reset()
            while not dones:
                with torch.no_grad():
                    observations_tensor = torch.from_numpy(observations).to(device=self.device)
                    actions_tensor = policy(observations_tensor, params=params).sample(deterministic=deterministic)
                    actions = actions_tensor.cpu().numpy()

                new_observations, rewards, dones, info = self.envs.step(actions)
                episodes.append(observations, actions, rewards, traj_num)
                observations, batch_ids = new_observations, traj_num
        return episodes






















    def sample_mj(self, policy, params=None, gamma=0.95, batch_size=None):
        self.num_workers = 1
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodes2(batch_size=batch_size, gamma=gamma, device=self.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):



            print("RUNNING")
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def test_sample(self, policy, params=None, gamma=0.95, batch_size=None):
        my_env = gym.make(self.env_name)
        frames = []



        for i in range(4):

            print(i)
            observations = my_env.reset()
            for t in range(80):




                my_env.render('human')
                frame = my_env.render('rgb_array')
                print(frame.shape)

                frames.append(frame)
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
                new_observations, rewards, dones, _ = my_env.step(actions)


                observations = new_observations




















        return frames



    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_data_tasks_chain(self, data, index_lst, episode_window, episode_num):
        tasks = []
        for index in index_lst:
            task_data = data[index:index + episode_window * episode_num]
            tasks.append(task_data)
        return tasks

    def sample_data_tasks(self, args, data, index_lst, episode_window):
        tasks = []
        for index in index_lst:
            task_data = data[index:index + episode_window * args.task_episode_num]
            tasks.append(task_data)
        return tasks


    def sample_tasks(self, args, index_lst, episode_window):
        tasks = []
        for index in index_lst:
            task_data = self.data[index:index + episode_window * args.task_episode_num]
            tasks.append(task_data)
        return tasks

    def sample_tasks_for_test(self, args, index_lst, episode_window):
        tasks = []
        for index in index_lst:
            if index == index_lst[-1]:
                task_data = self.data[index:index + episode_window * args.task_episode_num]
            else:
                task_data = self.data[index:index + episode_window * args.task_episode_num+1]
            tasks.append(task_data)
        return tasks

