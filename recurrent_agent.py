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
        if h is None:
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
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.prev_action_dim = int(np.prod(action_shape)) * max(0, int(action_history_len or 0))

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

        repr_dim = int(np.prod(self.hidden_state_shape))
        actor_input_dim = repr_dim + self.prev_action_dim

        self.actor = FeedforwardActor(actor_input_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic = FeedforwardCritic(actor_input_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = FeedforwardCritic(actor_input_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_opt = torch.optim.Adam(self.core.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

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

    def prepare_observation(self, obs, prev_actions):
        obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        prev_act_tensor = self._format_prev_actions(prev_actions, obs_tensor.shape[0])
        if self.hidden_state is None:
            self.hidden_state = self.core.init_hidden(obs_tensor.shape[0], self.device)
        warp = None
        if self.use_se2_warp:
            warp = torch.as_tensor(self.pending_warp, device=self.device, dtype=torch.float32).view(1, -1)
        new_hidden, flat = self.core(obs_tensor, self.hidden_state, warp_params=warp, augment=False)
        self.hidden_state = new_hidden.detach()
        self.cached_feature = flat.detach()
        self.cached_prev_actions = prev_act_tensor
        if self.use_se2_warp:
            self.pending_warp = np.zeros_like(self.pending_warp)

    def act(self, obs, step, eval_mode=False, prev_actions=None):
        self.prepare_observation(obs, prev_actions)
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
        obs = next(batch_iter)
        prev_actions_np = next(batch_iter) if self.prev_action_dim > 0 else None
        hidden_state = next(batch_iter)
        warp_params = next(batch_iter) if self.use_se2_warp else None
        action = next(batch_iter)
        reward = next(batch_iter)
        discount = next(batch_iter)
        next_obs = next(batch_iter)
        next_prev_actions_np = next(batch_iter) if self.prev_action_dim > 0 else None
        next_hidden_state = next(batch_iter)
        next_warp_params = next(batch_iter) if self.use_se2_warp else None

        obs = torch.as_tensor(obs, device=self.device).float()
        next_obs = torch.as_tensor(next_obs, device=self.device).float()
        reward = torch.as_tensor(reward, device=self.device).float()
        discount = torch.as_tensor(discount, device=self.device).float()
        action = torch.as_tensor(action, device=self.device).float()
        prev_actions = torch.as_tensor(prev_actions_np, device=self.device).float() if prev_actions_np is not None else None
        next_prev_actions = torch.as_tensor(next_prev_actions_np, device=self.device).float() if next_prev_actions_np is not None else None
        hidden_tensor = torch.as_tensor(hidden_state, device=self.device).float()
        next_hidden_tensor = torch.as_tensor(next_hidden_state, device=self.device).float()
        warp_tensor = torch.as_tensor(warp_params, device=self.device).float() if self.use_se2_warp else None
        next_warp_tensor = torch.as_tensor(next_warp_params, device=self.device).float() if self.use_se2_warp else None

        _, feats = self._compute_representation(obs, hidden_tensor, warp_tensor, augment=True)
        flat = feats.view(feats.shape[0], -1)

        _, next_feats = self._compute_representation(next_obs, next_hidden_tensor, next_warp_tensor, augment=True)
        next_flat = next_feats.view(next_feats.shape[0], -1)

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_flat, next_prev_actions, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(next_flat, next_prev_actions, next_action)
            target_V = torch.min(target_q1, target_q2)
            target_q = reward + discount * target_V

        current_q1, current_q2 = self.critic(flat, prev_actions, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        pref_loss = None
        if (
            pref_batch is not None
            and pref_weight > 0.0
            and isinstance(pref_batch, tuple)
        ):
            pref_iter = iter(pref_batch)
            pref_obs_np = next(pref_iter)
            pref_prev_np = next(pref_iter) if self.prev_action_dim > 0 else None
            pref_hidden_np = next(pref_iter) if self.hidden_state_shape is not None else None
            pref_warp_np = next(pref_iter) if self.use_se2_warp else None
            pref_teacher_np = next(pref_iter)
            pref_student_np = next(pref_iter)
            pref_obs = torch.as_tensor(pref_obs_np, device=self.device).float()
            pref_prev = torch.as_tensor(pref_prev_np, device=self.device).float() if pref_prev_np is not None else None
            pref_hidden = torch.as_tensor(pref_hidden_np, device=self.device).float() if pref_hidden_np is not None else None
            pref_warp = torch.as_tensor(pref_warp_np, device=self.device).float() if self.use_se2_warp else None
            pref_teacher = torch.as_tensor(pref_teacher_np, device=self.device).float()
            pref_student = torch.as_tensor(pref_student_np, device=self.device).float()
            _, pref_feats = self._compute_representation(pref_obs, pref_hidden, pref_warp, augment=True)
            pref_flat = pref_feats.view(pref_feats.shape[0], -1)
            teacher_q1, teacher_q2 = self.critic(pref_flat, pref_prev, pref_teacher)
            student_q1, student_q2 = self.critic(pref_flat, pref_prev, pref_student)
            teacher_q = torch.min(teacher_q1, teacher_q2)
            student_q = torch.min(student_q1, student_q2)
            delta = teacher_q - student_q
            if pref_loss_type == "bradley_terry":
                pref_loss = F.softplus(-delta).mean()
            else:
                pref_loss = F.softplus(
                    torch.as_tensor(pref_margin, device=delta.device, dtype=delta.dtype) - delta
                ).mean()
            critic_loss = critic_loss + float(pref_weight) * pref_loss

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        actor_obs = flat.detach()
        actor_prev_actions = prev_actions.detach() if prev_actions is not None else None
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(actor_obs, actor_prev_actions, stddev)
        new_action = dist.sample(clip=self.stddev_clip)
        actor_q1, actor_q2 = self.critic(actor_obs, actor_prev_actions, new_action)
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
        if pref_loss is not None:
            metrics['pref_loss'] = pref_loss.item()
        return metrics
