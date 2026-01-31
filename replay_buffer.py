# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import time
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)

    def num_episodes(self):
        return self._num_episodes

    def num_transitions(self):
        return self._num_transitions


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._has_prev_actions = False
        self._has_goal_history = False
        self._has_hidden_state = False
        self._has_warp_params = False
        self._stats = defaultdict(float)
        self._stats_counts = defaultdict(int)

    def _record_stat(self, key, value):
        self._stats[key] += float(value)
        self._stats_counts[key] += 1

    def get_stats(self, reset: bool = False):
        stats = dict(self._stats)
        counts = dict(self._stats_counts)
        if reset:
            self._stats = defaultdict(float)
            self._stats_counts = defaultdict(int)
        return stats, counts

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        if not self._has_prev_actions:
            self._has_prev_actions = 'prev_actions' in episode
        if not self._has_goal_history:
            self._has_goal_history = 'goal_history' in episode
        if not self._has_hidden_state:
            self._has_hidden_state = 'hidden_state' in episode
        if not self._has_warp_params:
            self._has_warp_params = 'warp_params' in episode
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        scan_start = time.perf_counter()
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        self._record_stat("replay_scan_s", time.perf_counter() - scan_start)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            load_start = time.perf_counter()
            if not self._store_episode(eps_fn):
                break
            self._record_stat("replay_load_s", time.perf_counter() - load_start)

    def _sample(self):
        while True:
            try:
                self._try_fetch()
            except:
                traceback.print_exc()
            self._samples_since_last_fetch += 1
            if self._episode_fns:
                break
            time.sleep(0.01)
        sample_start = time.perf_counter()
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        sample = [obs]
        if self._has_prev_actions:
            prev_actions = episode['prev_actions'][idx - 1]
            next_prev_actions = episode['prev_actions'][idx + self._nstep - 1]
            sample.append(prev_actions)
        if self._has_goal_history:
            goal_history = episode['goal_history'][idx - 1]
            next_goal_history = episode['goal_history'][idx + self._nstep - 1]
            sample.append(goal_history)
        if self._has_hidden_state:
            hidden_state = episode['hidden_state'][idx - 1]
            next_hidden_state = episode['hidden_state'][idx + self._nstep - 1]
            sample.append(hidden_state)
        if self._has_warp_params:
            warp_params = episode['warp_params'][idx - 1]
            next_warp_params = episode['warp_params'][idx + self._nstep - 1]
            sample.append(warp_params)
        sample.extend([action, reward, discount, next_obs])
        if self._has_prev_actions:
            sample.append(next_prev_actions)
        if self._has_goal_history:
            sample.append(next_goal_history)
        if self._has_hidden_state:
            sample.append(next_hidden_state)
        if self._has_warp_params:
            sample.append(next_warp_params)
        result = tuple(sample)
        self._record_stat("replay_sample_s", time.perf_counter() - sample_start)
        return result

    def __iter__(self):
        while True:
            yield self._sample()


class InMemoryReplayStorage:
    def __init__(self, data_specs, max_size):
        self._data_specs = data_specs
        self._max_size = max_size
        self._episodes = []
        self._current_episode = defaultdict(list)
        self._num_transitions = 0

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _store_episode(self, episode):
        eps_len = episode_len(episode)
        self._episodes.append(episode)
        self._num_transitions += eps_len
        while self._num_transitions > self._max_size and self._episodes:
            removed = self._episodes.pop(0)
            self._num_transitions -= episode_len(removed)

    def num_episodes(self):
        return len(self._episodes)

    def num_transitions(self):
        return self._num_transitions

    def sample_episode(self):
        return random.choice(self._episodes)


class InMemoryReplayBuffer(IterableDataset):
    def __init__(self, storage: InMemoryReplayStorage, nstep, discount):
        self._storage = storage
        self._nstep = nstep
        self._discount = discount
        self._has_prev_actions = False
        self._has_goal_history = False
        self._has_hidden_state = False
        self._has_warp_params = False
        self._stats = defaultdict(float)
        self._stats_counts = defaultdict(int)

    def _record_stat(self, key, value):
        self._stats[key] += float(value)
        self._stats_counts[key] += 1

    def get_stats(self, reset: bool = False):
        stats = dict(self._stats)
        counts = dict(self._stats_counts)
        if reset:
            self._stats = defaultdict(float)
            self._stats_counts = defaultdict(int)
        return stats, counts

    def _sample(self):
        sample_start = time.perf_counter()
        episode = self._storage.sample_episode()
        if not self._has_prev_actions:
            self._has_prev_actions = 'prev_actions' in episode
        if not self._has_goal_history:
            self._has_goal_history = 'goal_history' in episode
        if not self._has_hidden_state:
            self._has_hidden_state = 'hidden_state' in episode
        if not self._has_warp_params:
            self._has_warp_params = 'warp_params' in episode
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        sample = [obs]
        if self._has_prev_actions:
            prev_actions = episode['prev_actions'][idx - 1]
            next_prev_actions = episode['prev_actions'][idx + self._nstep - 1]
            sample.append(prev_actions)
        if self._has_goal_history:
            goal_history = episode['goal_history'][idx - 1]
            next_goal_history = episode['goal_history'][idx + self._nstep - 1]
            sample.append(goal_history)
        if self._has_hidden_state:
            hidden_state = episode['hidden_state'][idx - 1]
            next_hidden_state = episode['hidden_state'][idx + self._nstep - 1]
            sample.append(hidden_state)
        if self._has_warp_params:
            warp_params = episode['warp_params'][idx - 1]
            next_warp_params = episode['warp_params'][idx + self._nstep - 1]
            sample.append(warp_params)
        sample.extend([action, reward, discount, next_obs])
        if self._has_prev_actions:
            sample.append(next_prev_actions)
        if self._has_goal_history:
            sample.append(next_goal_history)
        if self._has_hidden_state:
            sample.append(next_hidden_state)
        if self._has_warp_params:
            sample.append(next_warp_params)
        result = tuple(sample)
        self._record_stat("replay_sample_s", time.perf_counter() - sample_start)
        return result

    def __iter__(self):
        while True:
            if self._storage.num_episodes() == 0:
                continue
            yield self._sample()


class SequenceReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers,
                 fetch_every, save_snapshot, sequence_length, burn_in):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._has_prev_actions = False
        self._has_goal_history = False
        self._sequence_length = int(sequence_length)
        self._burn_in = int(burn_in)

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except Exception:
            return False
        if not self._has_prev_actions:
            self._has_prev_actions = 'prev_actions' in episode
        if not self._has_goal_history:
            self._has_goal_history = 'goal_history' in episode
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len
        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except Exception:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            try:
                eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            except Exception:
                continue
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _sample(self):
        try:
            self._try_fetch()
        except Exception:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        eps_len = episode_len(episode)  # transitions
        window = self._sequence_length + self._burn_in
        if eps_len <= window:
            return self._sample()  # resample
        start = np.random.randint(0, eps_len - window + 1)
        end = start + window
        obs_seq = episode['observation'][start:end + 1]  # +1 for bootstrap
        action_seq = episode['action'][start + 1:end + 1]
        reward_seq = episode['reward'][start + 1:end + 1]
        discount_seq = episode['discount'][start + 1:end + 1]
        sample = [obs_seq.astype(np.float32)]
        if self._has_prev_actions:
            prev_seq = episode['prev_actions'][start:end]
            sample.append(prev_seq.astype(np.float32))
        if self._has_goal_history:
            goal_seq = episode['goal_history'][start:end]
            sample.append(goal_seq.astype(np.float32))
        sample.extend([
            action_seq.astype(np.float32),
            reward_seq.astype(np.float32),
            discount_seq.astype(np.float32),
        ])
        return tuple(sample)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = int(np.random.get_state()[1][0]) + int(worker_id)
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, fetch_every=1000):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=fetch_every,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader


def make_sequence_replay_loader(replay_dir, max_size, batch_size, num_workers,
                                save_snapshot, sequence_length, burn_in, fetch_every=1000):
    max_size_per_worker = max_size // max(1, num_workers)
    iterable = SequenceReplayBuffer(replay_dir,
                                    max_size_per_worker,
                                    num_workers,
                                    fetch_every=fetch_every,
                                    save_snapshot=save_snapshot,
                                    sequence_length=sequence_length,
                                    burn_in=burn_in)
    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader


def make_replay_loader_in_memory(storage, batch_size, nstep, discount):
    iterable = InMemoryReplayBuffer(storage, nstep, discount)
    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=0,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
