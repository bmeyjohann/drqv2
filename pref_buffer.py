from __future__ import annotations

import datetime
import io
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _save_chunk(chunk: Dict[str, np.ndarray], path: Path) -> None:
    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **chunk)
        buffer.seek(0)
        with path.open("wb") as fp:
            fp.write(buffer.read())


def _load_chunk(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as fp:
        data = np.load(fp)
        return {k: data[k] for k in data.files}


class PreferencePairStorage:
    """Disk-backed preference pair storage mirroring the DrQ replay layout."""

    def __init__(
        self,
        *,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        prev_action_shape: Tuple[int, ...],
        storage_dir: Path,
        chunk_size: int = 64,
        hidden_state_shape: Tuple[int, ...] | None = None,
        warp_param_dim: int = 0,
    ):
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)
        self.prev_action_shape = tuple(prev_action_shape)
        self.hidden_state_shape = tuple(hidden_state_shape) if hidden_state_shape else None
        self.warp_param_dim = int(max(0, warp_param_dim))
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(1, int(chunk_size))
        self._states: list[np.ndarray] = []
        self._prev_actions: list[np.ndarray] = []
        self._hidden_states: list[np.ndarray] = []
        self._warp_params: list[np.ndarray] = []
        self._teacher_actions: list[np.ndarray] = []
        self._student_actions: list[np.ndarray] = []
        self._num_chunks = 0
        self._num_pairs = 0
        self._preload()

    def _preload(self) -> None:
        for fn in sorted(self.storage_dir.glob("*.npz")):
            try:
                _, idx_str, count_str = fn.stem.split("_")
                idx = int(idx_str)
                count = int(count_str)
            except Exception:
                continue
            self._num_chunks = max(self._num_chunks, idx + 1)
            self._num_pairs += count

    def add(self,
            state: np.ndarray,
            prev_actions: np.ndarray,
            teacher_action: np.ndarray,
            student_action: np.ndarray,
            hidden_state: np.ndarray | None = None,
            warp_params: np.ndarray | None = None) -> None:
        state_arr = np.asarray(state)
        prev_arr = np.asarray(prev_actions, dtype=np.float32)
        teacher_arr = np.asarray(teacher_action, dtype=np.float32)
        student_arr = np.asarray(student_action, dtype=np.float32)
        if state_arr.shape != self.obs_shape:
            raise ValueError(f"Preference buffer state shape mismatch: expected {self.obs_shape}, got {state_arr.shape}")
        if prev_arr.shape != self.prev_action_shape:
            raise ValueError(
                f"Preference buffer prev_action shape mismatch: expected {self.prev_action_shape}, got {prev_arr.shape}"
            )
        if teacher_arr.shape != self.action_shape or student_arr.shape != self.action_shape:
            raise ValueError("Preference buffer action shape mismatch")
        if self.hidden_state_shape is not None:
            if hidden_state is None:
                raise ValueError("Preference buffer requires hidden_state but none provided")
            hidden_arr = np.asarray(hidden_state, dtype=np.float32)
            if hidden_arr.shape != self.hidden_state_shape:
                raise ValueError(
                    f"Preference buffer hidden_state shape mismatch: expected {self.hidden_state_shape}, got {hidden_arr.shape}"
                )
        else:
            hidden_arr = None
        if self.warp_param_dim > 0:
            if warp_params is None:
                raise ValueError("Preference buffer requires warp_params but none provided")
            warp_arr = np.asarray(warp_params, dtype=np.float32).reshape(-1)
            if warp_arr.shape[0] != self.warp_param_dim:
                raise ValueError(
                    f"Preference buffer warp_params length mismatch: expected {self.warp_param_dim}, got {warp_arr.shape[0]}"
                )
        else:
            warp_arr = None
        self._states.append(state_arr.astype(np.uint8, copy=False))
        self._prev_actions.append(prev_arr)
        if hidden_arr is not None:
            self._hidden_states.append(hidden_arr)
        if warp_arr is not None:
            self._warp_params.append(warp_arr)
        self._teacher_actions.append(teacher_arr)
        self._student_actions.append(student_arr)
        if len(self._states) >= self.chunk_size:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self._states:
            return
        chunk = {
            "states": np.stack(self._states, axis=0),
            "prev_actions": np.stack(self._prev_actions, axis=0),
            "teacher_actions": np.stack(self._teacher_actions, axis=0),
            "student_actions": np.stack(self._student_actions, axis=0),
        }
        if self._hidden_states:
            chunk["hidden_states"] = np.stack(self._hidden_states, axis=0)
        if self._warp_params:
            chunk["warp_params"] = np.stack(self._warp_params, axis=0)
        chunk_len = chunk["states"].shape[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = self.storage_dir / f"{timestamp}_{self._num_chunks}_{chunk_len}.npz"
        _save_chunk(chunk, filename)
        self._num_chunks += 1
        self._num_pairs += chunk_len
        self._states.clear()
        self._prev_actions.clear()
        self._hidden_states.clear()
        self._warp_params.clear()
        self._teacher_actions.clear()
        self._student_actions.clear()

    def num_pairs(self) -> int:
        return self._num_pairs

    def close(self) -> None:
        self._flush_chunk()


class PreferencePairDataset:
    """Lazy loader that mirrors ReplayBuffer's disk-backed sampling semantics."""

    def __init__(
        self,
        *,
        storage_dir: Path,
        max_size: int,
        fetch_every: int = 512,
        prev_action_shape: Tuple[int, ...] | None = None,
        hidden_state_shape: Tuple[int, ...] | None = None,
        warp_param_dim: int = 0,
    ):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max(1, int(max_size))
        self.fetch_every = max(1, int(fetch_every))
        self._samples_since_fetch = self.fetch_every
        self._chunks: Dict[Path, Dict[str, np.ndarray]] = {}
        self._chunk_order: list[Path] = []
        self._size = 0
        self.prev_action_shape = tuple(prev_action_shape) if prev_action_shape else None
        self.hidden_state_shape = tuple(hidden_state_shape) if hidden_state_shape else None
        self.warp_param_dim = int(max(0, warp_param_dim))

    def _try_fetch(self) -> None:
        if self._samples_since_fetch < self.fetch_every:
            return
        self._samples_since_fetch = 0
        for fn in sorted(self.storage_dir.glob("*.npz")):
            if fn in self._chunks:
                continue
            try:
                _, _, count_str = fn.stem.split("_")
                count = int(count_str)
            except Exception:
                continue
            if count > self.max_size:
                fn.unlink(missing_ok=True)
                continue
            while self._size + count > self.max_size and self._chunk_order:
                old_fn = self._chunk_order.pop(0)
                removed = self._chunks.pop(old_fn)
                self._size -= removed["states"].shape[0]
                old_fn.unlink(missing_ok=True)
            chunk = _load_chunk(fn)
            self._chunks[fn] = chunk
            self._chunk_order.append(fn)
            self._size += chunk["states"].shape[0]

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...] | None:
        if batch_size <= 0:
            return None
        self._samples_since_fetch += 1
        try:
            self._try_fetch()
        except Exception:
            pass
        if not self._chunk_order:
            return None
        states = []
        prev_actions = [] if self.prev_action_shape is not None else None
        hidden_states = [] if self.hidden_state_shape is not None else None
        warp_params = [] if self.warp_param_dim > 0 else None
        teacher_actions = []
        student_actions = []
        for _ in range(batch_size):
            fn = random.choice(self._chunk_order)
            chunk = self._chunks[fn]
            chunk_len = chunk["states"].shape[0]
            idx = np.random.randint(0, chunk_len)
            states.append(chunk["states"][idx])
            if prev_actions is not None:
                if "prev_actions" in chunk:
                    prev_actions.append(chunk["prev_actions"][idx])
                else:
                    prev_actions.append(np.zeros(self.prev_action_shape,
                                                 dtype=np.float32))
            if hidden_states is not None:
                if "hidden_states" in chunk:
                    hidden_states.append(chunk["hidden_states"][idx])
                else:
                    hidden_states.append(np.zeros(self.hidden_state_shape,
                                                  dtype=np.float32))
            if warp_params is not None:
                if "warp_params" in chunk:
                    warp_params.append(chunk["warp_params"][idx])
                else:
                    warp_params.append(np.zeros((self.warp_param_dim,), dtype=np.float32))
            teacher_actions.append(chunk["teacher_actions"][idx])
            student_actions.append(chunk["student_actions"][idx])
        states_arr = np.stack(states, axis=0)
        teacher_arr = np.stack(teacher_actions, axis=0)
        student_arr = np.stack(student_actions, axis=0)
        if prev_actions is not None:
            prev_arr = np.stack(prev_actions, axis=0)
            extras = [states_arr, prev_arr]
        else:
            extras = [states_arr]
        if hidden_states is not None:
            hidden_arr = np.stack(hidden_states, axis=0)
            extras.append(hidden_arr)
        if warp_params is not None:
            warp_arr = np.stack(warp_params, axis=0)
            extras.append(warp_arr)
        extras.extend([teacher_arr, student_arr])
        return tuple(extras)

    def loaded_size(self) -> int:
        return self._size
