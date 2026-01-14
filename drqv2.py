# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
try:
    import hydra  # type: ignore
except ImportError:  # pragma: no cover - hydra optional for agent use
    hydra = None
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, prev_actions, std):
        if prev_actions is not None:
            trunk_input = torch.cat([obs, prev_actions], dim=-1)
        else:
            trunk_input = obs
        h = self.trunk(trunk_input)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, prev_actions, action):
        if prev_actions is not None:
            trunk_input = torch.cat([obs, prev_actions], dim=-1)
        else:
            trunk_input = obs
        h = self.trunk(trunk_input)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 action_history_len=0,
                 goal_history_dim: int = 0,
                 use_context_mlp: bool = False,
                 context_hidden_dim: int = 128,
                 context_dim: int = 64):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.action_dim = int(np.prod(action_shape))
        self.action_history_len = max(0, int(action_history_len or 0))
        self.prev_action_dim = self.action_dim * self.action_history_len
        self.goal_history_dim = max(0, int(goal_history_dim or 0))
        self.context_input_dim = self.prev_action_dim + self.goal_history_dim
        self.use_context_mlp = bool(use_context_mlp) and self.context_input_dim > 0
        self.context_dim = int(context_dim) if self.use_context_mlp else self.context_input_dim

        # models
        self.encoder = Encoder(obs_shape).to(device)
        if self.use_context_mlp:
            self.context_mlp = nn.Sequential(
                nn.Linear(self.context_input_dim, int(context_hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(context_hidden_dim), self.context_dim),
                nn.ReLU(inplace=True),
            ).to(device)
        else:
            self.context_mlp = None
        actor_input_dim = self.encoder.repr_dim + self.context_dim
        self.actor = Actor(actor_input_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(actor_input_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(actor_input_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.context_opt = None
        if self.context_mlp is not None:
            self.context_opt = torch.optim.Adam(self.context_mlp.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def _format_prev_actions(self, prev_actions, batch_size):
        if self.prev_action_dim == 0:
            return None
        if prev_actions is None:
            return torch.zeros(batch_size,
                               self.prev_action_dim,
                               device=self.device)
        tensor = torch.as_tensor(prev_actions,
                                 device=self.device,
                                 dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _format_goal_history(self, goal_history, batch_size):
        if self.goal_history_dim == 0:
            return None
        if goal_history is None:
            return torch.zeros(batch_size,
                               self.goal_history_dim,
                               device=self.device)
        tensor = torch.as_tensor(goal_history,
                                 device=self.device,
                                 dtype=torch.float32)
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

    def act(self, obs, step, eval_mode=False, prev_actions=None, goal_history=None):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        context_tensor = self._build_context(prev_actions, goal_history, obs.shape[0])
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, context_tensor, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(
        self,
        obs,
        prev_actions,
        action,
        reward,
        discount,
        next_obs,
        next_prev_actions,
        step,
        pref_tensors=None,
        pref_weight: float = 0.0,
        pref_margin: float = 0.1,
        pref_loss_type: str = "margin",
    ):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, next_prev_actions, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs,
                                                     next_prev_actions,
                                                     next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, prev_actions, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        pref_loss = None
        if (
            pref_tensors is not None
            and pref_weight > 0.0
            and isinstance(pref_tensors, tuple)
            and len(pref_tensors) >= 3
        ):
            if len(pref_tensors) == 4:
                (pref_obs,
                 pref_prev_actions,
                 pref_teacher_actions,
                 pref_student_actions) = pref_tensors
            else:
                (pref_obs,
                 pref_teacher_actions,
                 pref_student_actions) = pref_tensors
                pref_prev_actions = None
            teacher_q1, teacher_q2 = self.critic(pref_obs,
                                                 pref_prev_actions,
                                                 pref_teacher_actions)
            student_q1, student_q2 = self.critic(pref_obs,
                                                 pref_prev_actions,
                                                 pref_student_actions)
            q_teacher = torch.min(teacher_q1, teacher_q2)
            q_student = torch.min(student_q1, student_q2)
            delta = q_teacher - q_student
            if pref_loss_type == "pvp":
                target_pos = torch.ones_like(teacher_q1)
                target_neg = -torch.ones_like(student_q1)
                loss_teacher = F.mse_loss(teacher_q1, target_pos) + F.mse_loss(teacher_q2, target_pos)
                loss_student = F.mse_loss(student_q1, target_neg) + F.mse_loss(student_q2, target_neg)
                pref_loss = 0.5 * (loss_teacher + loss_student)
            elif pref_loss_type == "bradley_terry":
                pref_loss = F.softplus(-delta).mean()
            else:
                pref_loss = F.softplus(
                    torch.as_tensor(pref_margin, device=delta.device, dtype=delta.dtype) - delta
                ).mean()
            critic_loss = critic_loss + float(pref_weight) * pref_loss

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()
        if pref_loss is not None:
            metrics['pref_loss'] = pref_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        if self.context_opt is not None:
            self.context_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        if self.context_opt is not None:
            self.context_opt.step()

        return metrics

    def update_actor(self, obs, prev_actions, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, prev_actions, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, prev_actions, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(
        self,
        replay_iter,
        step,
        pref_batch=None,
        pref_weight: float = 0.0,
        pref_margin: float = 0.1,
        pref_loss_type: str = "margin",
    ):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        if len(batch) == 5:
            obs, action, reward, discount, next_obs = utils.to_torch(
                batch, self.device)
            prev_actions = None
            next_prev_actions = None
            goal_history = None
            next_goal_history = None
        elif len(batch) == 7:
            (obs,
             prev_actions,
             action,
             reward,
             discount,
             next_obs,
             next_prev_actions) = utils.to_torch(batch, self.device)
            goal_history = None
            next_goal_history = None
        elif len(batch) == 9:
            (obs,
             prev_actions,
             goal_history,
             action,
             reward,
             discount,
             next_obs,
             next_prev_actions,
             next_goal_history) = utils.to_torch(batch, self.device)
        else:
            raise ValueError(
                f"Unexpected replay batch length {len(batch)}")

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        if prev_actions is not None:
            prev_actions = prev_actions.float()
            next_prev_actions = next_prev_actions.float()
        if goal_history is not None:
            goal_history = goal_history.float()
            next_goal_history = next_goal_history.float()
        prev_context = self._build_context(prev_actions, goal_history, obs.shape[0])
        next_context = self._build_context(next_prev_actions, next_goal_history, next_obs.shape[0])
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        pref_tensors = None
        if (
            pref_batch is not None
            and pref_weight > 0.0
            and isinstance(pref_batch, tuple)
            and len(pref_batch) in (3, 4)
        ):
            if len(pref_batch) == 5:
                pref_obs_np, pref_prev_np, pref_goal_np, pref_teacher_np, pref_student_np = pref_batch
                pref_prev = torch.as_tensor(pref_prev_np, device=self.device).float()
                pref_goal = torch.as_tensor(pref_goal_np, device=self.device).float()
            elif len(pref_batch) == 4:
                pref_obs_np, pref_prev_np, pref_teacher_np, pref_student_np = pref_batch
                pref_prev = torch.as_tensor(pref_prev_np, device=self.device).float()
                pref_goal = None
            else:
                pref_obs_np, pref_teacher_np, pref_student_np = pref_batch
                pref_prev = None
                pref_goal = None
            pref_obs = torch.as_tensor(pref_obs_np, device=self.device).float()
            pref_teacher = torch.as_tensor(pref_teacher_np, device=self.device).float()
            pref_student = torch.as_tensor(pref_student_np, device=self.device).float()
            pref_obs = self.aug(pref_obs)
            pref_obs = self.encoder(pref_obs)
            if pref_prev is not None:
                pref_prev = pref_prev.float()
            if pref_goal is not None:
                pref_goal = pref_goal.float()
            pref_context = self._build_context(pref_prev, pref_goal, pref_obs.shape[0])
            pref_tensors = (pref_obs, pref_context, pref_teacher, pref_student)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(
                obs,
                prev_context,
                action,
                reward,
                discount,
                next_obs,
                next_context,
                step,
                pref_tensors,
                pref_weight,
                pref_margin,
                pref_loss_type,
            ))

        # update actor
        actor_context = prev_context.detach() if prev_context is not None else None
        metrics.update(self.update_actor(obs.detach(), actor_context, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
