import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import skewnorm

from model import MLP, CNN, CNN_with_symbolic, CNN_with_RL
from buffer import ReplayBuffer, IntrinsicReplayBuffer, PrioritisedReplayBuffer, MixReplayBuffer, SILReplayBuffer, SILIntrinsicReplayBuffer
from configs.envSpecific import BASELINE
from utils.utils import get_env_name, to_tensor_float, MovAvg, modified_sigmoid


class DQNAgent:
    def __init__(self, state_space, symbolic_space, action_space, cfg, logger):
        """
        :param state_dim: About Task
        :param action_dim: About Task
        :param cfg: Config, About DQN setting
        """
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.action_dim = action_space.n
        self.frame_idx = 0  # Decay count for epsilon
        self.train_steps = cfg.train_steps
        self.batch_size = cfg.batch_size
        self.buffer_type = cfg.buffer_type
        self.max_grad_norm = cfg.max_grad_norm
        self.intrinsic_type = cfg.intrinsic_type
        self.train_freq = cfg.train_freq
        self.epsilon = lambda frame_idx: \
            cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.logger = logger
        self.rl = cfg.rl  # representation learning
        self.rl_coef = cfg.rl
        self.sil = cfg.sil
        self.sil_coef = cfg.sil_coef

        if self.rl:
            self.q_value_net = CNN_with_RL(state_space, action_space.n, predict_dim=symbolic_space.shape[0], features_dim=cfg.feature_dim,
                                           hidden_dim=cfg.hidden_dim).to(self.device)
            self.target_net = CNN_with_RL(state_space, action_space.n, predict_dim=symbolic_space.shape[0], features_dim=cfg.feature_dim,
                                          hidden_dim=cfg.hidden_dim).to(self.device)
        elif self.buffer_type == "mix":
            self.q_value_net = CNN_with_symbolic(state_space, symbolic_space.shape[0], action_space.n, features_dim=cfg.feature_dim,
                                                 hidden_dim=cfg.hidden_dim).to(self.device)
            self.target_net = CNN_with_symbolic(state_space, symbolic_space.shape[0], action_space.n, features_dim=cfg.feature_dim,
                                                hidden_dim=cfg.hidden_dim).to(self.device)

        elif len(state_space.shape) > 1:
            self.q_value_net = CNN(state_space, action_space.n, features_dim=cfg.feature_dim,
                                   hidden_dim=cfg.hidden_dim).to(self.device)
            self.target_net = CNN(state_space, action_space.n, features_dim=cfg.feature_dim,
                                  hidden_dim=cfg.hidden_dim).to(self.device)
        else:
            self.q_value_net = MLP(state_space.shape[0], action_space.n,
                                   hidden_dim=cfg.hidden_dim).to(self.device)
            self.target_net = MLP(state_space.shape[0], action_space.n,
                                  hidden_dim=cfg.hidden_dim).to(self.device)
        if cfg.load_network:
            self.load(cfg.network_path)

        self.optimizer = optim.Adam(self.q_value_net.parameters(), lr=cfg.lr)
        self.loss = 0
        if self.sil == True and self.buffer_type == "ili":
            self.replay_buffer = SILIntrinsicReplayBuffer(
                cfg.capacity, cfg.gamma)
        elif self.sil == True:
            self.replay_buffer = SILReplayBuffer(cfg.capacity, cfg.gamma)
        elif self.buffer_type == "simple":
            self.replay_buffer = ReplayBuffer(cfg.capacity)
        elif self.buffer_type == "mix":
            self.replay_buffer = MixReplayBuffer(cfg.capacity)
        elif self.buffer_type == "per":
            self.replay_buffer = PrioritisedReplayBuffer(
                cfg.capacity, cfg.alpha, cfg.beta, cfg.incremental_td_error, cfg.device, cfg.game_type)
        elif self.buffer_type == "ili":
            self.replay_buffer = IntrinsicReplayBuffer(cfg.capacity)
        else:
            raise NotImplementedError

        # KB Signal Parameters
        self.best_ripper = None
        self.test_return = MovAvg(size=cfg.window_size)
        self.baseline_score = BASELINE[get_env_name(
            cfg.env)] - (1 if get_env_name(cfg.env) in ["CollectHealth"] else 0)
        # ILI intrinsic reward coefficient
        self.intrinsic_constant = cfg.intrinsic_constant
        self.thresholds = cfg.thresholds
        self.ili_exponential_decay = lambda frame_idx: cfg.intrinsic_constant * \
            math.exp(-1. * frame_idx /
                     cfg.ili_decay_constant)  # ILI reward coefficient in exponential decay
        self.load_ripper = cfg.load_ripper  # knowledge base for transfer
        self.dexp = cfg.dexp  # Directed Exploration

        skew_alpha = cfg.skew_alpha
        dense = cfg.skew_dense
        self.lb, self.ub = skewnorm.ppf(
            dense, skew_alpha), skewnorm.ppf(1-dense, skew_alpha)
        self.rv = skewnorm(skew_alpha)

    def action(self, state, symbolic_state=None):
        """ For test policy

        Args:
            state (_type_): _description_

        Returns:
            _type_: Execution action == Q-value action
        """
        return self.choose_action(state, symbolic_state, test=True)[0]

    def choose_action(self, state, symbolic_state=None, test=False):
        """ For train policy

        Args:
            state (_type_): _description_
            symbolic_state (_type_): if it is a pixel-symbolic RL env, we will action according to joint pixel and symbolic state
            test (bool, optional): Cancel exploration(epsilon-greedy) during testing . Defaults to False.

        Returns:
            _type_: Execution action, Q-value action
        """
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float)
            if self.rl:
                q_value = self.q_value_net(state)[0]
            elif self.buffer_type == "mix":
                q_value = self.q_value_net(state, symbolic_state)
            else:
                q_value = self.q_value_net(state)
            action = torch.argmax(q_value, dim=1).cpu().numpy()
            q_action = action
        if test == False:
            # Select actions using e—greedy principle
            self.frame_idx += state.shape[0]
            if random.random() <= self.epsilon(self.frame_idx):
                if self.load_ripper or (self.dexp and self.best_ripper["ripper"].jrip_available):
                    if random.random() >= self.epsilon(self.frame_idx):
                        action = self.best_ripper["ripper"].action(
                            symbolic_state)
                    else:
                        action = np.array([random.randrange(self.action_dim)
                                           for _ in range(state.shape[0])])
                else:
                    action = np.array([random.randrange(self.action_dim)
                                       for _ in range(state.shape[0])])
        return action, q_action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        losses, q_values, prediction_losses, sil_losses = [], [], [], []
        if self.buffer_type == "ili":
            intrinsic_coeff = self.get_intrinsic_coeff()

        # start update Q network
        for _ in range(self.train_freq):
            # Randomly sample transitions from the replay buffer
            if self.buffer_type == "ili":
                sample_data = self.replay_buffer.sample(
                    self.batch_size, self.best_ripper["ripper"], intrinsic_coeff)
            elif self.buffer_type == "simple" or self.buffer_type == "per" or self.buffer_type == "mix" or self.sil == True:
                sample_data = self.replay_buffer.sample(self.batch_size)
            else:
                raise NotImplementedError
            if self.buffer_type == "mix":
                frame_state_batch, value_state_batch, action_batch, reward_batch, next_frame_state_batch, next_value_state_batch, done_batch = sample_data
                frame_state_batch = to_tensor_float(
                    frame_state_batch, device=self.device)
                value_state_batch = to_tensor_float(
                    value_state_batch, device=self.device)
                next_frame_state_batch = to_tensor_float(
                    next_frame_state_batch, device=self.device)
                next_value_state_batch = to_tensor_float(
                    next_value_state_batch, device=self.device)
            else:
                if self.sil == True:
                    frame_state_batch, value_state_batch, action_batch, reward_batch, next_state_batch, done_batch, return_batch = sample_data
                    return_batch = to_tensor_float(
                        return_batch, device=self.device)
                else:
                    frame_state_batch, value_state_batch, action_batch, reward_batch, next_state_batch, done_batch = sample_data
                if frame_state_batch[0] is not None:
                    state_batch = frame_state_batch
                else:
                    state_batch = value_state_batch
                state_batch = to_tensor_float(state_batch, device=self.device)
                next_state_batch = to_tensor_float(
                    next_state_batch, device=self.device)
            action_batch = torch.tensor(
                action_batch, device=self.device).unsqueeze(1)
            reward_batch = to_tensor_float(reward_batch, device=self.device)
            done_batch = to_tensor_float(done_batch, device=self.device)

            if self.rl:
                q_value, frame_state_pred = self.q_value_net(state_batch)
                q_value = q_value.gather(
                    dim=1, index=action_batch)  # shape: [32,1]
                next_q_value = self.target_net(next_state_batch)[
                    0].max(1)[0].detach()
                # Calculate prediction loss
                value_state_batch = to_tensor_float(
                    value_state_batch, device=self.device)
                prediction_loss = nn.MSELoss()(frame_state_pred, value_state_batch)
            elif self.buffer_type == "mix":
                q_value = self.q_value_net(frame_state_batch, value_state_batch).gather(
                    dim=1, index=action_batch)
                next_q_value = self.target_net(
                    next_frame_state_batch, next_value_state_batch).max(1)[0].detach()
            else:
                # index action_batch is obtained from the replay buffer
                q_value = self.q_value_net(state_batch).gather(
                    dim=1, index=action_batch)  # shape: [32,1]
                # Calculate Q(s,a) at time t+1
                # q_{t+1}=max_a Q(s_t+1,a)
                next_q_value = self.target_net(
                    next_state_batch).max(1)[0].detach()
                if self.sil == True:
                    advantages = return_batch - q_value.squeeze()
                    positive_advantages = torch.clamp(advantages, min=0)
                    sil_loss = 0.5 * torch.mean(positive_advantages ** 2)
            # For the termination state, the corresponding expected_q_value is equal to reward
            expected_q_value = reward_batch + \
                self.gamma * next_q_value * (1 - done_batch)

            loss = nn.MSELoss()(q_value, expected_q_value.unsqueeze(1))
            if self.rl:
                loss += prediction_loss * self.rl_coef
            if self.sil == True:
                loss += sil_loss * self.sil_coef
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.q_value_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.buffer_type == "per":
                td_errors = expected_q_value.detach().cpu().numpy() - \
                    q_value.squeeze().detach().cpu().numpy()
                self.replay_buffer.update_td_errors(td_errors)

            losses.append(loss.item())
            q_values.append(q_value.mean().item())
            if self.rl:
                prediction_losses.append(prediction_loss.item())
            if self.sil == True:
                sil_losses.append(sil_loss.item())

        self.logger.log("loss/td_loss", np.mean(losses), self.frame_idx)
        self.logger.log("q_value/dqn", np.mean(q_values), self.frame_idx)
        if self.rl:
            self.logger.log("loss/prediction_loss",
                            np.mean(prediction_losses), self.frame_idx)
        if self.sil == True:
            self.logger.log("loss/sil_loss",
                            np.mean(sil_losses), self.frame_idx)

    def save(self, path):
        torch.save(self.q_value_net.state_dict(),
                   os.path.join(path, "q_checkpoint.pth"))
        torch.save(self.target_net.state_dict(),
                   os.path.join(path, "target_checkpoint.pth"))

    def load(self, path):
        self.q_value_net.load_state_dict(
            torch.load(os.path.join(path, "q_checkpoint.pth")))
        self.target_net.load_state_dict(torch.load(
            os.path.join(path, "target_checkpoint.pth")))

    def get_intrinsic_coeff(self):
        best_ripper_return = self.best_ripper["return"].mean - \
            self.baseline_score
        test_return = self.test_return.mean() - \
            self.baseline_score
        if self.intrinsic_type == "adaptation":
            intrinsic_coeff = modified_sigmoid(
                max(1-(test_return/(best_ripper_return+1e-6)), 0)) * self.intrinsic_constant
        elif self.intrinsic_type == "constant":
            intrinsic_coeff = self.intrinsic_constant if best_ripper_return * \
                self.thresholds > test_return else 0
        elif self.intrinsic_type == "epsilon":
            intrinsic_coeff = self.ili_exponential_decay(
                self.frame_idx) if best_ripper_return * self.thresholds > test_return else 0
        elif self.intrinsic_type == "skew":
            x = (self.frame_idx/self.train_steps)*(self.ub-self.lb)+self.lb
            intrinsic_coeff = self.rv.pdf(x)
        else:
            raise NotImplementedError
        self.logger.log("hyperP/intrinsic_coeff",
                        intrinsic_coeff, self.frame_idx)
        return intrinsic_coeff
