from collections import namedtuple
import collections
import numpy as np
import torch
import pdb
import pandas as pd
import os

from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


def load_power_dataset(data_dir):
    """
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    data = collections.defaultdict(list)
    for file in os.listdir(data_dir):
        abs_file = os.path.join(data_dir, file)
        df = pd.read_csv(
            abs_file,
            header=None,
            names=[
                "freq1",
                "freq2",
                "dfreq1",
                "dfreq2",
                "pm1",
                "pm2",
                "pe1",
                "pe2",
                "pgov1",
                "pgov2",
                "ptie",
                "a1",
                "a2",
            ],
        )

        def rewards_fn():
            r4freq = np.exp(-np.abs(df["freq1"] + df["freq2"]) * 1e2)

            freq1, freq2 = df["freq1"].to_numpy(), df["freq2"].to_numpy()
            next_freq1 = np.concatenate((freq1[1:], [freq1[-1]]))
            next_freq2 = np.concatenate((freq2[1:], [freq2[-1]]))
            delta_freq1 = np.abs(freq1) - np.abs(next_freq1)
            delta_freq2 = np.abs(freq2) - np.abs(next_freq2)
            r4dfreq = (delta_freq1 + delta_freq2) * 1e2

            r4power = (df["a1"] + 1e1 * df["a2"]).to_numpy() * 1e-1

            return r4freq + r4dfreq + r4power

        yield {
            "observations": df.iloc[:, :11].to_numpy(),
            "actions": df.iloc[:, 11:13].to_numpy(),
            "rewards": rewards_fn(),
            "terminals": np.array([False] * len(df)).astype(np.bool_),
        }


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        env="hopper-medium-replay",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        seed=None,
    ):
        assert env == "power"
        # self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        # self.env = env = load_environment(env)
        # self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        # itr = sequence_dataset(env, self.preprocess_fn)
        itr = load_power_dataset("assets/data")
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
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
        self.discount = discount
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
        rewards = self.fields["rewards"][path_ind, start:]
        discounts = self.discounts[: len(rewards)]
        value = (discounts * rewards).sum() / self.horizon
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
