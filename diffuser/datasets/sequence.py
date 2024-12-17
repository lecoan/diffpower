# 处理数据
from collections import namedtuple
# import collections
import numpy as np
import torch
# import pdb
import pandas as pd
import os
from functools import reduce

from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


# 从指定目录加载数据集，并且对文件中的数据进行处理。
def load_power_dataset(data_dir):
    """
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """

    columns = ["freq1","freq2","dfreq1","dfreq2","pm1","pm2",
               "pe1","pe2","pgov1","pgov2","ptie","a1","a2"]
    
    # 遍历指定目录（data_dir）下的所有文件，并加载以 .csv 结尾的文件
    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            continue
        
        # 使用 os 模块的 path.join() 函数将目录名和文件名合并成一个完整的路径。
        # abs_file 是当前 CSV 文件的完整路径。
        abs_file = os.path.join(data_dir, file)
        
        # 使用 pandas 库将这些文件加载为数据框
        df = pd.read_csv(
            abs_file,
            header=None,
            names=columns,
        )
        
        # 用于计算每一步的奖励
        def get_step_reward(base, states):
            # conditions（state.size*trac_sample_num:4*1000）：true/false
            conditions = map(lambda x: x<base, states)
            # 只有4个state全部都是true才会返回1，否则为0，stable是一个trac_sample_num纬的向量（boolean
            stable = reduce(lambda x, y: np.logical_and(x, y), conditions)
            # float
            return stable.astype(np.float32)
        
        # 从数据中提取出有关频率以及频率变化的信息，计算出每一步的奖励，并将所有的奖励值求和。
        def rewards_fn():
            freq1, freq2 = df["freq1"].to_numpy(), df["freq2"].to_numpy()
            dfreq1, dfreq2 = df["dfreq1"].to_numpy(), df["dfreq2"].to_numpy()

            rewards = []
            for i in range(2, 8):
                step_rewards = get_step_reward(1e1**(-i), np.array([freq1, freq2, dfreq1, dfreq2])) * i
                rewards.append(step_rewards)

            return np.array(rewards).sum(axis=0)
            # return get_step_reward(5e-5, np.array([freq1, freq2, dfreq1, dfreq2]))
        
        rewards = rewards_fn()
        
        df = df.drop(['dfreq1', 'dfreq2', 'pe1', 'pe2'], axis=1)
        # 返回一个迭代器，会返回若干个结果
        yield {
            "observations": df.iloc[:, :len(df.columns)-2].to_numpy(),
            "actions": df.iloc[:, -2:].to_numpy(),
            "rewards": rewards,
            "terminals": np.array([False] * len(df)).astype(np.bool_),
        }

# 
class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        # 默认的环境是"hopper-medium-replay"
        env="hopper-medium-replay",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=15000,
        termination_penalty=0,
        use_padding=True,
        seed=None,
    ):
        # 在后面的代码中有一个 assert env == "power"，这意味着真实的环境应投入电力系统
        assert env == "power"
        # self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        # self.env = env = load_environment(env)
        # self.env.seed(seed)
        # 这里应用是实际的 horizon 之类的参数是要从外部调用的
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        # itr = sequence_dataset(env, self.preprocess_fn)
        # 调用加载电力数据集的函数 load_power_dataset
        itr = load_power_dataset("assets/rand")
        # 通过对 itr（电力数据集的迭代器）的遍历，来填充 fields实例。
        # max_n_episodes：一条轨迹，最多几条轨迹；max_path_length：轨迹几个点最多；termination_penalty：无
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        # 遍历迭代器，数据给到 fields
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        # 希望传入下标给出对应的 dataset，indices就是一系列下标
        self.indices = self.make_indices(fields.path_lengths, horizon)
        
        # 初始化了一些属性，例如 observation_dim（观察的维度）、 action_dim（行动的维度）、 fields（数据缓冲区）、 n_episodes（总的回合数）、 path_lengths（路径长度）。
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        # 最后，对数据进行正则化，然后打印出fields实例。
        self.normalize()
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=["observations", "actions"]):
        """
        normalize fields that will be predicted by the diffusion model
        """
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, -1
            )

    def make_indices(self, path_lengths, horizon):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    # 初始状态！
    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        """
        condition on both the current observation and the last observation in the plan
        """
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        # discount常被用为折扣因子，用于计算累积奖励。
        self.discount = discount
        # 生成一个从0到 max_path_length-1 的数组，然后将折扣因子按幂运算应用到这个数组，生成一个新的数组
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print(
            "[ datasets/sequence ] Getting value dataset bounds...", end=" ", flush=True
        )
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print("✓")
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        # start:：从start到结尾
        rewards = self.fields["rewards"][path_ind, start:]
        # : len(rewards：从0到len(rewards，数组切片6
        discounts = self.discounts[: len(rewards)]
        # 一个horizon序列的总的奖励
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        # 
        value_batch = ValueBatch(*batch, value)
        return value_batch
