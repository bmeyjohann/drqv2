import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .drqv2 import RandomShiftsAug, Actor as FeedforwardActor, Critic as FeedforwardCritic


class ConvEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) == 3
        in_channels = obs_shape[0]
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        self.out_channels = 32
        self.out_hw = 35
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        return self.convnet(obs)


class LinearEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) == 3
        self.convnet = ConvEncoder(obs_shape)
        self.repr_dim = self.convnet.out_channels * (self.convnet.out_hw ** 2)

    def forward(self, obs):
        features = self.convnet(obs)
        return features.view(features.shape[0], -1)


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h):
        if h is None or h.dim() == 0:
            size_h = [x.size(0), self.reset_gate.out_channels, x.size(2), x.size(3)]
            h = torch.zeros(size_h, device=x.device, dtype=x.dtype)
        combined = torch.cat([x, h], dim=1)
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        combined_reset = torch.cat([x, reset * h], dim=1)
        out = torch.tanh(self.out_gate(combined_reset))
        new_h = (1 - update) * h + update * out
        return new_h


def _apply_se2_warp(hidden, warp_params, height, width):
    if warp_params is None:
        return hidden
    if hidden is None:
        return hidden
    batch = hidden.shape[0]
    device = hidden.device
    if warp_params.dim() == 1:
        warp_params = warp_params.view(1, -1)
    if warp_params.shape[1] < 3:
        return hidden
    dx_px = warp_params[:, 0].view(batch, 1)
    dy_px = warp_params[:, 1].view(batch, 1)
    dtheta = warp_params[:, 2].view(batch, 1)
    norm_x = 2.0 * dx_px / max(1.0, float(width))
    norm_y = 2.0 * dy_px / max(1.0, float(height))
    cos_t = torch.cos(dtheta)
    sin_t = torch.sin(dtheta)
    tx = cos_t * norm_x - sin_t * norm_y
    ty = sin_t * norm_x + cos_t * norm_y
    theta = torch.zeros(batch, 2, 3, device=device, dtype=hidden.dtype)
    theta[:, 0, 0] = cos_t.view(-1)
    theta[:, 0, 1] = -sin_t.view(-1)
    theta[:, 0, 2] = tx.view(-1)
    theta[:, 1, 0] = sin_t.view(-1)
    theta[:, 1, 1] = cos_t.view(-1)
    theta[:, 1, 2] = ty.view(-1)
    grid = F.affine_grid(theta, hidden.size(), align_corners=False)
    return F.grid_sample(hidden, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class RecurrentFeatureExtractor(nn.Module):
    def __init__(
        self,
        obs_shape,
        recurrent_type: str,
        hidden_dim: int,
        conv_hidden_channels: int,
        use_se2_warp: bool,
    ):
        super().__init__()
        self.recurrent_type = recurrent_type
        self.use_se2_warp = use_se2_warp and recurrent_type == 'convgru'
        self.aug = RandomShiftsAug(pad=4)
        if recurrent_type == 'gru':
            self.encoder = LinearEncoder(obs_shape)
            self.hidden_dim = int(hidden_dim)
            self.state_shape = (self.hidden_dim,)
            self.core = nn.GRUCell(self.encoder.repr_dim, self.hidden_dim)
            utils.weight_init(self.core)
        else:
            self.encoder = ConvEncoder(obs_shape)
            self.hidden_dim = int(conv_hidden_channels)
            self.state_shape = (self.hidden_dim, self.encoder.out_hw, self.encoder.out_hw)
            self.core = ConvGRUCell(self.encoder.out_channels, self.hidden_dim)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.recurrent_type == 'gru':
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        shape = (batch_size, self.hidden_dim, self.encoder.out_hw, self.encoder.out_hw)
        return torch.zeros(shape, device=device)

    def forward(self, obs, hidden, warp_params=None, augment: bool = True):
        if augment:
            obs = self.aug(obs.float())
        else:
            obs = obs.float()
        if self.recurrent_type == 'gru':
            encoded = self.encoder(obs)
            if hidden is None:
                hidden = torch.zeros(encoded.shape[0], self.hidden_dim, device=encoded.device, dtype=encoded.dtype)
            new_hidden = self.core(encoded, hidden)
            flat = new_hidden
            return new_hidden, flat
        encoded = self.encoder(obs)
        if hidden is None:
            hidden = torch.zeros(
                encoded.shape[0],
                self.hidden_dim,
                self.encoder.out_hw,
                self.encoder.out_hw,
                device=encoded.device,
                dtype=encoded.dtype,
            )
        if self.use_se2_warp and warp_params is not None:
            hidden = _apply_se2_warp(hidden, warp_params, self.encoder.out_hw, self.encoder.out_hw)
        new_hidden = self.core(encoded, hidden)
        flat = new_hidden.view(new_hidden.shape[0], -1)
        return new_hidden, flat


class DrQV2RecurrentAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        action_history_len: int,
        recurrent_type: str,
        recurrent_hidden_dim: int,
        conv_hidden_channels: int,
        use_se2_warp: bool,
        unroll_length: int = 8,
        burn_in: int = 0,
        goal_history_dim: int = 0,
        use_context_mlp: bool = False,
        context_hidden_dim: int = 128,
        context_dim: int = 64,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.prev_action_dim = int(np.prod(action_shape)) * max(0, int(action_history_len or 0))
        self.goal_history_dim = max(0, int(goal_history_dim or 0))
        self.context_input_dim = self.prev_action_dim + self.goal_history_dim
        self.use_context_mlp = bool(use_context_mlp) and self.context_input_dim > 0
        self.context_dim = int(context_dim) if self.use_context_mlp else self.context_input_dim

        self.core = RecurrentFeatureExtractor(
            obs_shape,
            recurrent_type=recurrent_type,
            hidden_dim=recurrent_hidden_dim,
            conv_hidden_channels=conv_hidden_channels,
            use_se2_warp=use_se2_warp,
        ).to(device)
        self.recurrent_type = recurrent_type
        self.use_se2_warp = self.core.use_se2_warp
        self.hidden_state_shape = (self.core.hidden_dim,) if recurrent_type == 'gru' else (
            self.core.hidden_dim,
            self.core.encoder.out_hw,
            self.core.encoder.out_hw,
        )
        self.warp_dim = 3 if self.use_se2_warp else 0
        self.unroll_length = int(unroll_length)
        self.burn_in = int(burn_in)

        repr_dim = int(np.prod(self.hidden_state_shape))
        if self.use_context_mlp:
            self.context_mlp = nn.Sequential(
                nn.Linear(self.context_input_dim, int(context_hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(context_hidden_dim), self.context_dim),
                nn.ReLU(inplace=True),
            ).to(device)
        else:
            self.context_mlp = None
        actor_input_dim = repr_dim + self.context_dim

        self.actor = FeedforwardActor(actor_input_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic = FeedforwardCritic(actor_input_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = FeedforwardCritic(actor_input_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_opt = torch.optim.Adam(self.core.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.context_opt = None
        if self.context_mlp is not None:
            self.context_opt = torch.optim.Adam(self.context_mlp.parameters(), lr=lr)

        self.hidden_state = None
        self.cached_feature = None
        self.cached_prev_actions = None
        self.pending_warp = np.zeros(self.warp_dim, dtype=np.float32)

        self.train()
        self.critic_target.train()

    def train(self, training: bool = True):
        self.training = training
        self.core.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def reset_memory(self):
        self.hidden_state = None
        self.cached_feature = None
        self.cached_prev_actions = None
        self.pending_warp = np.zeros(self.warp_dim, dtype=np.float32)

    def export_state(self) -> np.ndarray:
        if self.hidden_state is None:
            zeros = np.zeros(self.hidden_state_shape, dtype=np.float32)
            return zeros
        snapshot = self.hidden_state.detach().cpu().numpy()
        # Drop batch dimension when exporting to replay/pref buffers.
        if snapshot.ndim > len(self.hidden_state_shape):
            snapshot = snapshot[0]
        return snapshot.copy()

    def export_pending_warp(self) -> np.ndarray:
        return np.array(self.pending_warp, dtype=np.float32)

    def register_pending_warp(self, warp_params: np.ndarray):
        if not self.use_se2_warp or warp_params is None:
            if self.warp_dim > 0:
                self.pending_warp = np.zeros(self.warp_dim, dtype=np.float32)
            return
        warp = np.asarray(warp_params, dtype=np.float32)
        if warp.shape[-1] != self.warp_dim:
            warp = np.pad(warp, (0, max(0, self.warp_dim - warp.shape[-1])), mode='constant')
        self.pending_warp = warp.astype(np.float32, copy=True)

    def _format_prev_actions(self, prev_actions, batch_size):
        if self.prev_action_dim == 0:
            return None
        if prev_actions is None:
            return torch.zeros(batch_size, self.prev_action_dim, device=self.device)
        tensor = torch.as_tensor(prev_actions, device=self.device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _format_goal_history(self, goal_history, batch_size):
        if self.goal_history_dim == 0:
            return None
        if goal_history is None:
            return torch.zeros(batch_size, self.goal_history_dim, device=self.device)
        tensor = torch.as_tensor(goal_history, device=self.device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _build_context(self, prev_actions, goal_history, batch_size):
        if self.context_input_dim == 0:
            return None
        prev_tensor = self._format_prev_actions(prev_actions, batch_size)
        goal_tensor = self._format_goal_history(goal_history, batch_size)
        if prev_tensor is None and goal_tensor is None:
            return None
        if prev_tensor is None:
            context_in = goal_tensor
        elif goal_tensor is None:
            context_in = prev_tensor
        else:
            context_in = torch.cat([prev_tensor, goal_tensor], dim=-1)
        if self.context_mlp is not None:
            return self.context_mlp(context_in)
        return context_in

    def prepare_observation(self, obs, prev_actions, goal_history):
        obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        context_tensor = self._build_context(prev_actions, goal_history, obs_tensor.shape[0])
        if self.hidden_state is None:
            self.hidden_state = self.core.init_hidden(obs_tensor.shape[0], self.device)
        warp = None
        if self.use_se2_warp:
            warp = torch.as_tensor(self.pending_warp, device=self.device, dtype=torch.float32).view(1, -1)
        new_hidden, flat = self.core(obs_tensor, self.hidden_state, warp_params=warp, augment=False)
        self.hidden_state = new_hidden.detach()
        self.cached_feature = flat.detach()
        self.cached_prev_actions = context_tensor
        if self.use_se2_warp:
            self.pending_warp = np.zeros_like(self.pending_warp)

    def act(self, obs, step, eval_mode=False, prev_actions=None, goal_history=None):
        self.prepare_observation(obs, prev_actions, goal_history)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(self.cached_feature, self.cached_prev_actions, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=self.stddev_clip)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def _compute_representation(self, obs, hidden_state, warp_params, augment=True):
        obs_tensor = obs
        hidden_tensor = hidden_state
        warp_tensor = warp_params
        if not torch.is_tensor(obs_tensor):
            obs_tensor = torch.as_tensor(obs_tensor, device=self.device)
        if obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        if not torch.is_tensor(hidden_tensor):
            hidden_tensor = torch.as_tensor(hidden_tensor, device=self.device)
        if hidden_tensor.dim() == len(self.hidden_state_shape):
            hidden_tensor = hidden_tensor.unsqueeze(0)
        if self.use_se2_warp:
            if warp_tensor is None:
                warp_tensor = torch.zeros(obs_tensor.shape[0], self.warp_dim, device=self.device)
            else:
                warp_tensor = torch.as_tensor(warp_tensor, device=self.device)
                if warp_tensor.dim() == 1:
                    warp_tensor = warp_tensor.unsqueeze(0)
        else:
            warp_tensor = None
        new_hidden, flat = self.core(obs_tensor, hidden_tensor, warp_params=warp_tensor, augment=augment)
        return new_hidden, flat

    def _compute_warp_sequence(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.use_se2_warp:
            B, T, _ = actions.shape
            return torch.zeros(B, T + 1, 3, device=actions.device, dtype=actions.dtype)
        B, T, _ = actions.shape
        warp_seq = torch.zeros(B, T + 1, 3, device=actions.device, dtype=actions.dtype)
        dx = actions[..., 0]
        dy = actions[..., 1]
        warp_seq[:, 1:, 0] = dx
        warp_seq[:, 1:, 1] = dy
        ang_prev = torch.atan2(dy.roll(1, dims=1), dx.roll(1, dims=1))
        ang_now = torch.atan2(dy, dx)
        ang_prev[:, 0] = ang_now[:, 0]
        warp_seq[:, 1:, 2] = ang_now - ang_prev
        return warp_seq

    def update(self,
               replay_iter,
               step,
               pref_batch=None,
               pref_weight: float = 0.0,
               pref_margin: float = 0.1,
               pref_loss_type: str = "margin"):
        metrics = {}
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        batch_iter = iter(batch)
        obs_seq = torch.as_tensor(next(batch_iter), device=self.device).float()
        prev_seq = None
        if self.prev_action_dim > 0:
            prev_seq = torch.as_tensor(next(batch_iter), device=self.device).float()
        goal_seq = None
        if self.goal_history_dim > 0:
            goal_seq = torch.as_tensor(next(batch_iter), device=self.device).float()
        action_seq = torch.as_tensor(next(batch_iter), device=self.device).float()
        reward_seq = torch.as_tensor(next(batch_iter), device=self.device).float()
        discount_seq = torch.as_tensor(next(batch_iter), device=self.device).float()

        B, Tp1, _, _, _ = obs_seq.shape
        total_steps = Tp1 - 1
        burn_in = min(self.burn_in, total_steps - 1)

        init_hidden = self.core.init_hidden(B, self.device)
        warp_seq = self._compute_warp_sequence(action_seq) if self.use_se2_warp else None

        feats_list = []
        h = init_hidden
        for t in range(Tp1):
            warp_t = warp_seq[:, t] if warp_seq is not None else None
            h, flat = self.core(obs_seq[:, t], h, warp_params=warp_t, augment=True)
            feats_list.append(flat)
        feats = torch.stack(feats_list, dim=1)
        repr_dim = feats.shape[-1]

        start_idx = burn_in
        end_idx = total_steps
        z_t = feats[:, start_idx:end_idx].reshape(-1, repr_dim)
        z_tp1 = feats[:, start_idx + 1:end_idx + 1].reshape(-1, repr_dim)
        a_t = action_seq[:, start_idx:end_idx].reshape(-1, action_seq.shape[-1])
        r_t = reward_seq[:, start_idx:end_idx].reshape(-1, 1)
        disc_t = discount_seq[:, start_idx:end_idx].reshape(-1, 1)

        if self.prev_action_dim > 0 and prev_seq is not None:
            prev_t = prev_seq[:, start_idx:end_idx].reshape(-1, self.prev_action_dim)
            next_prev = prev_seq[:, start_idx + 1:end_idx + 1].reshape(-1, self.prev_action_dim)
        else:
            prev_t = None
            next_prev = None
        if self.goal_history_dim > 0 and goal_seq is not None:
            goal_t = goal_seq[:, start_idx:end_idx].reshape(-1, self.goal_history_dim)
            next_goal = goal_seq[:, start_idx + 1:end_idx + 1].reshape(-1, self.goal_history_dim)
        else:
            goal_t = None
            next_goal = None

        context_t = self._build_context(prev_t, goal_t, z_t.shape[0])
        next_context = self._build_context(next_prev, next_goal, z_tp1.shape[0])

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist_next = self.actor(z_tp1, next_context, stddev)
            a_tp1 = dist_next.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(z_tp1, next_context, a_tp1)
            target_V = torch.min(target_q1, target_q2)
            target_q = r_t + disc_t * target_V

        current_q1, current_q2 = self.critic(z_t, context_t, a_t)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        if self.context_opt is not None:
            self.context_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        if self.context_opt is not None:
            self.context_opt.step()

        stddev = utils.schedule(self.stddev_schedule, step)
        z_detach = z_t.detach()
        context_detach = context_t.detach() if context_t is not None else None
        dist = self.actor(z_detach, context_detach, stddev)
        new_action = dist.sample(clip=self.stddev_clip)
        actor_q1, actor_q2 = self.critic(z_detach, context_detach, new_action)
        actor_loss = -torch.min(actor_q1, actor_q2).mean()
        log_prob = dist.log_prob(new_action).sum(-1, keepdim=True)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        metrics['critic_target_q'] = target_q.mean().item()
        metrics['critic_q1'] = current_q1.mean().item()
        metrics['critic_q2'] = current_q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()
        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        return metrics
