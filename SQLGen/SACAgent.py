import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
from copy import deepcopy
from collections import deque
import pickle
import logging

from SQLGen.config import Config
from SQLGen.SQLNet import SQLStateEncoder
config = Config()
logger = logging.getLogger(__name__)
# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_annealing=0.001, epsilon=1e-6):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # determines how much prioritization to use (0 = no prioritization, 1 = full prioritization)
        self.beta = beta    # importance-sampling correction (annealed from beta to 1 during training)
        self.beta_annealing = beta_annealing  # controls how quickly beta increases
        self.epsilon = epsilon  # small constant to ensure all priorities are non-zero
        self.max_priority = 1.0  # initial priority for new experiences
        self.episode_id = 0
        self.history_reward = {}

    def push(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = [state, action, reward, next_state, done, action_mask, next_action_mask, self.episode_id]
        if done:
            self.episode_id += 1
        # New experiences get maximum priority
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity

    def get_position(self):
        return (self.position - 1) % self.capacity
    
    def set_step_reward(self, step_reward, position): 
        if position == -1:
            return
        episode_id = self.buffer[position][7]
        old_reward = self.buffer[position][2]
        self.buffer[position][2] = self.rollout_reward(step_reward, episode_id)
        if old_reward != 0.0 and abs(old_reward - self.buffer[position][2]) / old_reward > 0.1: 
            len_buffer = len(self.buffer)
            while episode_id == self.buffer[position][7]:
                self.priorities[position] = self.max_priority
                if position == 0 and len_buffer != self.capacity:
                    break
                else:
                    position = (position - 1) % self.capacity

    def rollout_reward(self, new_reward, episode_id, gamma = 0.95): 
        if episode_id not in self.history_reward:
            self.history_reward[episode_id] = []
        self.history_reward[episode_id].append(new_reward)
        self.history_reward[episode_id] = self.history_reward[episode_id][-5:]
        n_reward = len(self.history_reward[episode_id])
        weight_mean = sum([r * (gamma ** (n_reward - i - 1)) for i, r in enumerate(self.history_reward[episode_id])]) / n_reward
        mean_mean = sum(self.history_reward[episode_id]) / n_reward
        std = min((sum([(r - mean_mean) ** 2 for r in self.history_reward[episode_id]]) / n_reward) ** 0.5, 5.0) 
        return weight_mean + std

    def get_reward_stats(self):
        reward_all = []
        for i in range(len(self.buffer)):
            if self.buffer[i][2] > 1e-4:
                reward_all.append(self.buffer[i][2])
        max_reward = max(reward_all)
        min_reward = min(reward_all)
        mean_reward = sum(reward_all) / len(reward_all)
        return max_reward, min_reward, mean_reward

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
            
        # Calculate the sampling probabilities based on the priorities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Sample batch of indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance-sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Anneal beta towards 1 over time (for more accurate updates)
        self.beta = min(1.0, self.beta + self.beta_annealing)
        
        # Retrieve samples
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done, action_mask, next_action_mask, episode_id = map(list, zip(*batch))
        
        return state, action, reward, next_state, done, action_mask, next_action_mask, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
        self.max_priority = max(self.max_priority, np.max(priorities))
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path):
        """Save buffer state to disk using pickle"""
        os.makedirs(path, exist_ok=True)
        buffer_state = {
            'buffer': self.buffer,
            'position': self.position,
            'priorities': self.priorities,
            'max_priority': self.max_priority,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_annealing': self.beta_annealing,
            'epsilon': self.epsilon,
            'episode_id': self.episode_id,
            'history_reward': self.history_reward
        }
        with open(os.path.join(path, "prioritized_replay_buffer.pkl"), 'wb') as f:
            pickle.dump(buffer_state, f)
        
    @classmethod
    def load(cls, path, capacity):
        """Load buffer state from disk using pickle"""
        buffer_path = os.path.join(path, "prioritized_replay_buffer.pkl")
        if not os.path.exists(buffer_path):
            return cls(capacity)
            
        with open(buffer_path, 'rb') as f:
            buffer_state = pickle.load(f)
            
        buffer = cls(capacity)
        buffer.buffer = buffer_state['buffer']
        buffer.position = buffer_state['position']
        buffer.priorities = buffer_state['priorities']
        buffer.max_priority = buffer_state['max_priority']
        buffer.alpha = buffer_state['alpha']
        buffer.beta = buffer_state['beta']
        buffer.beta_annealing = buffer_state['beta_annealing']
        buffer.epsilon = buffer_state['epsilon']
        buffer.episode_id = buffer_state['episode_id']
        buffer.history_reward = buffer_state['history_reward']
        return buffer

# Preserve the original ReplayBuffer as fallback
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, action_mask, next_action_mask)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, action_mask, next_action_mask = map(list, zip(*batch))
        return state, action, reward, next_state, done, action_mask, next_action_mask
    
    def __len__(self):
        return len(self.buffer)
        
    def save(self, path):
        """Save buffer state to disk using pickle"""
        os.makedirs(path, exist_ok=True)
        buffer_state = {
            'buffer': self.buffer,
            'position': self.position
        }
        with open(os.path.join(path, "replay_buffer.pkl"), 'wb') as f:
            pickle.dump(buffer_state, f)
        
    @classmethod
    def load(cls, path, capacity):
        """Load buffer state from disk using pickle"""
        buffer_path = os.path.join(path, "replay_buffer.pkl")
        if not os.path.exists(buffer_path):
            return cls(capacity)
            
        with open(buffer_path, 'rb') as f:
            buffer_state = pickle.load(f)
            
        buffer = cls(capacity)
        buffer.buffer = buffer_state['buffer']
        buffer.position = buffer_state['position']
        return buffer

# N-step return helper
class NStepBuffer:
    def __init__(self, n_step=1, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
        
    def push(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        self.buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))
        
    def get(self):
        if len(self.buffer) < self.n_step:
            return None
            
        # Get the first (oldest) transition
        state, action, reward, _, _, action_mask, _ = self.buffer[0]
        
        # Compute n-step reward and get the latest next_state
        n_reward = 0
        for i in range(self.n_step):
            r = self.buffer[i][2]  # reward
            n_reward += r * (self.gamma ** i)
            
            if self.buffer[i][4]:  # if done
                break
        
        next_state = self.buffer[-1][3]
        done = self.buffer[-1][4]
        next_action_mask = self.buffer[-1][6]
        
        return state, action, n_reward, next_state, done, action_mask, next_action_mask
        
    def clear(self):
        self.buffer.clear()
        
    def save(self, path):
        """Save n-step buffer state to disk using pickle"""
        os.makedirs(path, exist_ok=True)
        n_step_state = {
            'buffer': list(self.buffer),
            'n_step': self.n_step,
            'gamma': self.gamma
        }
        with open(os.path.join(path, "n_step_buffer.pkl"), 'wb') as f:
            pickle.dump(n_step_state, f)
        
    @classmethod
    def load(cls, path):
        """Load n-step buffer state from disk using pickle"""
        n_step_path = os.path.join(path, "n_step_buffer.pkl")
        if not os.path.exists(n_step_path):
            return None
            
        with open(n_step_path, 'rb') as f:
            n_step_state = pickle.load(f)
            
        buffer = cls(n_step=n_step_state['n_step'], gamma=n_step_state['gamma'])
        buffer.buffer = deque(n_step_state['buffer'], maxlen=n_step_state['n_step'])
        return buffer

# QNetwork for SAC with twin Q functions
class QNetwork(nn.Module):
    def __init__(self, state_encoder, action_state_dim, action_dim, hidden_dim=256, layer_norm=False):
        super(QNetwork, self).__init__()
        self.state_encoder: SQLStateEncoder = state_encoder
        self.action_state_dim = action_state_dim
        self.action_type_embed = nn.Embedding(30, action_state_dim)
        
        q_layers = []
        q_layers.append(nn.Linear(state_encoder.hidden_dim + action_state_dim, hidden_dim))
        if layer_norm:
            q_layers.append(nn.LayerNorm(hidden_dim))
        q_layers.append(nn.LeakyReLU())
        q_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if layer_norm:
            q_layers.append(nn.LayerNorm(hidden_dim))
        q_layers.append(nn.LeakyReLU())
        q_layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.qnetwork = nn.Sequential(*q_layers)
        
    def forward(self, obs, action_type=None):
        # Process the main observation through embedmodel
        representation = self.state_encoder(obs)
        
        # Process the action code if provided
        if action_type is not None:
            action_type_embed = self.action_type_embed(action_type.long().squeeze(-1))
            
            # Concatenate the embeddings
            combined_obs = torch.cat((representation, action_type_embed), dim=1)
        else:
            # During training, the action type might be embedded in the obs
            action_type = obs["action_type"]
            action_type_embed = self.action_type_embed(action_type.long().squeeze(-1))
            combined_obs = torch.cat((representation, action_type_embed), dim=1)
        
        return self.qnetwork(combined_obs)

# Policy Network for SAC
class PolicyNetwork(nn.Module):
    def __init__(self, state_encoder, action_state_dim, action_dim, hidden_dim=256, layer_norm=False):
        super(PolicyNetwork, self).__init__()
        self.state_encoder: SQLStateEncoder = state_encoder
        self.action_state_dim = action_state_dim
        self.action_type_embed = nn.Embedding(30, action_state_dim)
        
        self.fc_mean = nn.Linear(state_encoder.hidden_dim, state_encoder.hidden_dim)
        self.fc_log_std = nn.Linear(state_encoder.hidden_dim, state_encoder.hidden_dim)
        # Use larger networks as suggested by RLlib
        policy_layers = []
        policy_layers.append(nn.Linear(state_encoder.hidden_dim + action_state_dim, hidden_dim))
        if layer_norm:
            policy_layers.append(nn.LayerNorm(hidden_dim))
        policy_layers.append(nn.Tanh())
        policy_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if layer_norm:
            policy_layers.append(nn.LayerNorm(hidden_dim))
        policy_layers.append(nn.Tanh())
        policy_layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.policy = nn.Sequential(*policy_layers)
        
    def forward(self, obs, action_mask=None, get_log_prob=False, deterministic=False, get_all_probs_log_probs=False, add_noise=True):
        representation = self.state_encoder(obs)
        
        mean = self.fc_mean(representation)
        if not add_noise:
            z = mean
        else:
            log_std = self.fc_log_std(representation).clamp(min=-5.0, max=2.0) 
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            z = mean + eps * std  # reparameterization trick
        
        # Get action type from observation
        action_type = obs["action_type"]
        action_type_embed = self.action_type_embed(action_type.long().squeeze(-1))
        
        # Concatenate the embeddings
        combined_obs = torch.cat((z, action_type_embed), dim=1)
        
        # Get raw logits
        logits = self.policy(combined_obs)
        
        # Apply mask: set logits for invalid actions to -inf
        if action_mask is not None:
            mask = action_mask.bool()
            logits = torch.where(mask, logits, torch.tensor(-1e38, device=logits.device))
        
        # Get action probabilities
        probs = F.softmax(logits, dim=-1)
        
        if get_all_probs_log_probs:
            # log_softmax is numerically stabler than log(softmax(x))
            # It handles the masking implicitly as logits for masked actions are very low.
            log_probs_all = F.log_softmax(logits, dim=-1)
            return probs, log_probs_all
        
        if deterministic:
            # Choose most likely action
            action = torch.argmax(probs, dim=-1)
            log_prob = None  # Not needed for deterministic actions
        else:
            # Sample action from distribution
            distribution = torch.distributions.Categorical(probs=probs)
            action = distribution.sample()
            
            # Calculate log probability if needed
            if get_log_prob:
                log_prob = distribution.log_prob(action)
                return action, log_prob
        
        return action, probs

# Huber loss for more stable gradients
def huber_loss(x, delta=1.0):
    return torch.where(
        torch.abs(x) < delta,
        0.5 * x.pow(2),
        delta * (torch.abs(x) - 0.5 * delta)
    )

# SAC Agent
class SACAgent:
    def __init__(
        self,
        action_dim = 466,
        action_state_dim=16,
        hidden_dim=256,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        n_step=1,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta=0.4,
        actor_lr=3e-4,
        critic_lr=3e-4,
        entropy_lr=3e-4, 
        grad_clip=None,
        layer_norm=False,
        target_network_update_freq=0,  # 0 means use soft updates
        alpha=0.2,
        automatic_entropy_tuning=True,
        # deterministic_eval=True,
        device=config.device
    ):
        self.action_dim = action_dim
        self.action_state_dim = action_state_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = layer_norm
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_lr = entropy_lr

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = device
        self.prioritized_replay = prioritized_replay
        self.grad_clip = grad_clip
        self.n_step = n_step
        # self.deterministic_eval = deterministic_eval
        self.target_network_update_freq = target_network_update_freq
        self.update_counter = 0
        

        self.buffer_capacity = buffer_size
        self.buffer_alpha = prioritized_replay_alpha
        self.buffer_beta = prioritized_replay_beta
        self.buffer_beta_annealing = 0.001
        
        self.state_encoder = SQLStateEncoder().to(device)
        
        # Get the action space size
        
        # Initialize policy network
        self.policy_net = PolicyNetwork(self.state_encoder, action_state_dim, action_dim, hidden_dim, layer_norm).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=actor_lr)
        
        # Initialize two Q-networks and their target networks
        self.q_net1 = QNetwork(self.state_encoder, action_state_dim, action_dim, hidden_dim, layer_norm).to(device)
        self.q_net2 = QNetwork(self.state_encoder, action_state_dim, action_dim, hidden_dim, layer_norm).to(device)
        
        self.target_q_net1 = deepcopy(self.q_net1).to(device)
        self.target_q_net2 = deepcopy(self.q_net2).to(device)
        
        # Freeze target networks (no gradient updates)
        for param in self.target_q_net1.parameters():
            param.requires_grad = False
        for param in self.target_q_net2.parameters():
            param.requires_grad = False
            
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=critic_lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=critic_lr)
        
        # Initialize replay buffer
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.buffer_capacity,
                alpha=self.buffer_alpha,
                beta=self.buffer_beta
            )
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_capacity)
            
        # Initialize n-step buffer if n_step > 1
        self.n_step_buffer = NStepBuffer(n_step=n_step, gamma=gamma) if n_step > 1 else None
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -0.98 * np.log(1.0 / action_dim) 
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=entropy_lr)

    def select_action(self, state, action_mask, evaluate=False):
        self.state_encoder.train() # enable droupout 
        with torch.no_grad():
            # Convert numpy arrays to tensors and move to the correct device
            state_tensor = {k: torch.from_numpy(v).float().unsqueeze(0).to(self.device) for k, v in state.items()}
            action_mask_tensor = torch.from_numpy(action_mask).float().unsqueeze(0).to(self.device)
            
            # Use deterministic actions for evaluation if configured
            # deterministic = evaluate and self.deterministic_eval
            
            action, _ = self.policy_net(state_tensor, action_mask_tensor, get_log_prob=False, deterministic = False, add_noise=True)
            action = action.squeeze().cpu().item()
                
            return action
    
    def select_dbs(self, state_list):
        self.state_encoder.eval()
        with torch.no_grad():
            keys = state_list[0].keys()
            
            # Process each key in the state dictionaries
            state_tensor = {}
            for key in keys:
                state_tensor[key] = torch.cat([torch.from_numpy(s[key]).float().unsqueeze(0) for s in state_list], dim=0).to(self.device)
     
            action_masks = torch.stack([torch.from_numpy(s['action_mask']).float() for s in state_list]).to(self.device)
            
    
            q1 = self.q_net1(state_tensor)
            q2 = self.q_net2(state_tensor)
            q = torch.min(q1, q2)  
            
    
            # V(s) = α * log ∑_{a∈valid} exp(Q(s,a)/α)
            alpha = self.alpha if not self.automatic_entropy_tuning else self.log_alpha.exp().item()
            
            actor_probs, actor_log_probs = self.policy_net(state_tensor, action_mask=action_masks, deterministic=False, add_noise=False, get_all_probs_log_probs=True)
            v_values = torch.sum(actor_probs * (q - alpha * actor_log_probs), dim=1)
            temperature = 5.0  # exploration or exploitation
            v_probs = F.softmax(v_values / temperature, dim=0)
            
            distribution = torch.distributions.Categorical(probs=v_probs)
            db_idx = distribution.sample().item()
            
            return db_idx
            
    def store_transition(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        if self.n_step_buffer is not None:
            # Add to n-step buffer
            self.n_step_buffer.push(state, action, reward, next_state, done, action_mask, next_action_mask)
            
            # Get n-step transition if available
            n_step_transition = self.n_step_buffer.get()
            
            # Add n-step transition to replay buffer
            if n_step_transition is not None:
                self.replay_buffer.push(*n_step_transition)
                
            # If episode ended, clear n-step buffer
            if done:
                self.n_step_buffer.clear()
        else:
            # Directly add to replay buffer (standard 1-step)
            self.replay_buffer.push(state, action, reward, next_state, done, action_mask, next_action_mask)
            
    def set_step_reward(self, step_reward, position):
        self.replay_buffer.set_step_reward(step_reward, position)
        
    def get_buffer_position(self):
        return self.replay_buffer.get_position()
    
    def update_parameters(self, updates=1):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0, 0, 0
        
        q1_losses, q2_losses, policy_losses, alpha_losses = [], [], [], []
        self.state_encoder.train()
        for _ in range(updates):
            # Sample from replay buffer
            if self.prioritized_replay:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, action_mask_batch, next_action_mask_batch, weights_batch, indices = self.replay_buffer.sample(self.batch_size)
                # Convert to torch tensor
                weights_tensor = torch.tensor(weights_batch, dtype=torch.float).to(self.device)
            else:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, action_mask_batch, next_action_mask_batch = self.replay_buffer.sample(self.batch_size)
                weights_tensor = torch.ones(self.batch_size, device=self.device)  # Uniform weights
                indices = None  # Not needed for standard replay buffer
            
            # Convert to torch tensors
            state_tensors = {}
            next_state_tensors = {}
            
            # Collect keys from first state to determine structure
            keys = state_batch[0].keys()
            
            # Process each key in the state dictionaries
            for key in keys:
                state_tensors[key] = torch.cat([torch.from_numpy(s[key]).float().unsqueeze(0) for s in state_batch], dim=0).to(self.device)
                next_state_tensors[key] = torch.cat([torch.from_numpy(ns[key]).float().unsqueeze(0) for ns in next_state_batch], dim=0).to(self.device)
            
            action_tensor = torch.tensor(action_batch, dtype=torch.long).to(self.device)
            reward_tensor = torch.tensor(reward_batch, dtype=torch.float).unsqueeze(1).to(self.device)
            done_tensor = torch.tensor(done_batch, dtype=torch.float).unsqueeze(1).to(self.device)
            action_mask_tensor = torch.stack([torch.from_numpy(m).float() for m in action_mask_batch]).to(self.device)
            next_action_mask_tensor = torch.stack([torch.from_numpy(m).float() for m in next_action_mask_batch]).to(self.device)
            
            # Update Q-networks
            with torch.no_grad():
                # Get next actions and log_probs using current policy
                next_actions, next_log_probs = self.policy_net(next_state_tensors, next_action_mask_tensor, get_log_prob=True)
                
                # Get target Q-values
                next_q1 = self.target_q_net1(next_state_tensors)
                next_q2 = self.target_q_net2(next_state_tensors)
                
                # Use the minimum Q-value for the next action
                next_q_values = torch.min(
                    next_q1.gather(1, next_actions.unsqueeze(1)),
                    next_q2.gather(1, next_actions.unsqueeze(1))
                )
                
                # Calculate target value with entropy regularization
                alpha = self.alpha if not self.automatic_entropy_tuning else self.log_alpha.exp().item()
                next_q_values = next_q_values - alpha * next_log_probs.unsqueeze(1)
                target_q = reward_tensor + (1 - done_tensor) * (self.gamma ** self.n_step) * next_q_values
            
            # Current Q-values
            current_q1 = self.q_net1(state_tensors)
            # Gather Q-values for the actions that were taken
            q1_action_values = current_q1.gather(1, action_tensor.unsqueeze(1))
            # Calculate Q-network losses with importance sampling weights and Huber loss
            q1_loss = (weights_tensor * huber_loss(q1_action_values - target_q)).mean()
            # Update first Q-network
            self.q_optimizer1.zero_grad()
            q1_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), self.grad_clip)
            self.q_optimizer1.step()

            current_q2 = self.q_net2(state_tensors)
            q2_action_values = current_q2.gather(1, action_tensor.unsqueeze(1))
            q2_loss = (weights_tensor * huber_loss(q2_action_values - target_q)).mean()
            # Update second Q-network
            self.q_optimizer2.zero_grad()
            q2_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), self.grad_clip)
            self.q_optimizer2.step()

            # Calculate TD errors for prioritized replay buffer
            td_error1 = torch.abs(q1_action_values - target_q)
            td_error2 = torch.abs(q2_action_values - target_q)
            td_error = torch.min(td_error1, td_error2).detach().cpu().numpy()
            # Update priority in the replay buffer
            if self.prioritized_replay:
                self.replay_buffer.update_priorities(indices, td_error.flatten())
            # Update policy network
            actions, log_probs = self.policy_net(state_tensors, action_mask_tensor, get_log_prob=True)
            
            # Get current Q-values for the sampled actions
            q1 = self.q_net1(state_tensors)
            q2 = self.q_net2(state_tensors)
            
            # Use minimum of the two Q-values (more conservative estimate)
            q_values = torch.min(
                q1.gather(1, actions.unsqueeze(1)),
                q2.gather(1, actions.unsqueeze(1))
            )
            
            # Calculate policy loss with entropy regularization
            alpha = self.alpha if not self.automatic_entropy_tuning else self.log_alpha.exp().item()
            policy_loss = (weights_tensor * (alpha * log_probs.unsqueeze(1) - q_values)).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
            self.policy_optimizer.step()
            
            # Update temperature parameter alpha
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_([self.log_alpha], self.grad_clip)
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp().item()
                alpha_losses.append(alpha_loss.item())
            
            # Update target networks
            self.update_counter += 1
            if self.target_network_update_freq > 0:
                # Hard update based on frequency
                if self.update_counter % self.target_network_update_freq == 0:
                    # Full copy of parameters
                    self.target_q_net1.load_state_dict(self.q_net1.state_dict())
                    self.target_q_net2.load_state_dict(self.q_net2.state_dict())
            else:
                # Soft update with tau
                for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            q1_losses.append(q1_loss.item())
            q2_losses.append(q2_loss.item())
            policy_losses.append(policy_loss.item())
        
        return np.mean(q1_losses), np.mean(q2_losses), np.mean(policy_losses), np.mean(alpha_losses) if alpha_losses else 0
    
    def retrain_from_buffer(self, max_epochs=1000, updates_per_epoch=1, convergence_patience=20, convergence_threshold=1e-3, min_delta_improvement=1e-3):
        logger.info("[SACAgent] Starting retraining from buffer...")
        self.reset_networks_and_optimizers()

        if len(self.replay_buffer) < self.batch_size:
            logger.warning("[SACAgent] Not enough samples in replay buffer to start retraining. Skipping.")
            return 0, 0, 0, 0

        recent_policy_losses = deque(maxlen=convergence_patience * 2)
        
        q1_loss_epoch, q2_loss_epoch, policy_loss_epoch, alpha_loss_epoch = 0,0,0,0
        max_reward, min_reward, mean_reward = self.replay_buffer.get_reward_stats()
        logger.info(f"[SACAgent] max_reward: {max_reward:.2f}, min_reward: {min_reward:.2f}, mean_reward: {mean_reward:.2f}")
        for epoch in range(max_epochs):
            q1_loss, q2_loss, policy_loss, alpha_loss = self.update_parameters(updates=updates_per_epoch)
            q1_loss_epoch = q1_loss
            q2_loss_epoch = q2_loss
            policy_loss_epoch = policy_loss
            alpha_loss_epoch = alpha_loss

            if policy_loss is None or np.isnan(policy_loss): # np.isnan needs policy_loss to be a float, not list
                logger.warning(f"[SACAgent] Retraining stopped at epoch {epoch+1} due to NaN or None policy loss.")
                break
            
            recent_policy_losses.append(policy_loss)

            if len(recent_policy_losses) == convergence_patience * 2:
                current_window_avg_loss = np.mean(list(recent_policy_losses)[-convergence_patience:])
                prev_window_avg_loss = np.mean(list(recent_policy_losses)[:convergence_patience])
                
                improvement = prev_window_avg_loss - current_window_avg_loss
                
                if abs(improvement) < convergence_threshold or improvement < min_delta_improvement : # also consider if improvement is too small even if not stagnant
                    logger.info(f"[SACAgent] Retraining converged at epoch {epoch+1}. Improvement: {improvement:.6f}")
                    break
            
            if (epoch + 1) % 10 == 0: # Log every 10 epochs
                logger.info(f"[SACAgent] Retraining Epoch {epoch+1}/{max_epochs}: Policy Loss: {policy_loss:.4f}, Q1 Loss: {q1_loss:.4f}, Q2 Loss: {q2_loss:.4f}, Alpha: {self.alpha:.4f}")
        else: # Loop finished without break (i.e. max_epochs reached)
            logger.info(f"[SACAgent] Retraining reached max_epochs ({max_epochs}).")
        
        logger.info(f"[SACAgent] Retraining finished. Final losses - Q1: {q1_loss_epoch:.4f}, Q2: {q2_loss_epoch:.4f}, Policy: {policy_loss_epoch:.4f}, Alpha: {alpha_loss_epoch:.4f}")
        return q1_loss_epoch, q2_loss_epoch, policy_loss_epoch, alpha_loss_epoch


    def reset_networks_and_optimizers(self):
        logger.info("[SACAgent] Resetting networks and optimizers...")
        # Re-initialize state encoder (if it has learnable parameters not shared via reference)
        # Assuming SQLStateEncoder re-initialization is simple or handled if it's re-assigned.
        # If state_encoder is complex, ensure its weights are also reset.
        # For now, we assume state_encoder is either fresh or its existing instance is fine to reuse.
        # If it needs to be reset, it should be: self.state_encoder = SQLStateEncoder().to(self.device)

        # Re-initialize policy network
        self.policy_net = PolicyNetwork(self.state_encoder, self.action_state_dim, self.action_dim, self.hidden_dim, self.layer_norm).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.actor_lr)

        # Re-initialize Q-networks and their target networks
        self.q_net1 = QNetwork(self.state_encoder, self.action_state_dim, self.action_dim, self.hidden_dim, self.layer_norm).to(self.device)
        self.q_net2 = QNetwork(self.state_encoder, self.action_state_dim, self.action_dim, self.hidden_dim, self.layer_norm).to(self.device)
        
        self.target_q_net1 = deepcopy(self.q_net1).to(self.device)
        self.target_q_net2 = deepcopy(self.q_net2).to(self.device)
        
        # Freeze target networks
        for param in self.target_q_net1.parameters():
            param.requires_grad = False
        for param in self.target_q_net2.parameters():
            param.requires_grad = False
            
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.critic_lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.critic_lr)
        
        # Reset automatic entropy tuning if enabled
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.entropy_lr)
            self.alpha = self.log_alpha.exp().item() # Ensure self.alpha is updated

        # Reset update counter
        self.update_counter = 0
        logger.info("[SACAgent] Networks and optimizers reset.")


    def save_models(self, path, only_models = False):
        """Save all model states and training states to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save neural networks with torch.save
        with open(os.path.join(path, "state_encoder.pth"), 'wb') as f:
            torch.save(self.state_encoder.state_dict(), f)
        with open(os.path.join(path, "policy_net.pth"), 'wb') as f:
            torch.save(self.policy_net.state_dict(), f)
        with open(os.path.join(path, "q_net1.pth"), 'wb') as f:
            torch.save(self.q_net1.state_dict(), f)
        with open(os.path.join(path, "q_net2.pth"), 'wb') as f:
            torch.save(self.q_net2.state_dict(), f)
        if only_models:
        # Save training state using pickle
            training_state = {
                'update_counter': self.update_counter,
                'alpha': self.alpha,
                'automatic_entropy_tuning': self.automatic_entropy_tuning,
                'tau': self.tau,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'prioritized_replay': self.prioritized_replay,
                'grad_clip': self.grad_clip,
                'n_step': self.n_step,
                # 'deterministic_eval': self.deterministic_eval,
                'target_network_update_freq': self.target_network_update_freq,
                'buffer_capacity': self.buffer_capacity,
                'buffer_alpha': self.buffer_alpha,
                'buffer_beta': self.buffer_beta,
                'buffer_beta_annealing': self.buffer_beta_annealing,
                # Save network configuration parameters
                'action_dim': self.action_dim,
                'action_state_dim': self.action_state_dim,
                'hidden_dim': self.hidden_dim,
                'layer_norm': self.layer_norm,
                'actor_lr': self.actor_lr,
                'critic_lr': self.critic_lr,
                'entropy_lr': self.entropy_lr,
            }
            
            if self.automatic_entropy_tuning:
                training_state['log_alpha'] = self.log_alpha.detach().cpu()
                training_state['target_entropy'] = self.target_entropy
            
            with open(os.path.join(path, "training_state.pkl"), 'wb') as f:
                pickle.dump(training_state, f)
        
        # Save replay buffer
        # self.replay_buffer.save(path)
        
        # # Save n-step buffer if it exists
        # if self.n_step_buffer is not None:
        #     self.n_step_buffer.save(path)
        
        # gc.collect()
    
    def load_models(self, path, is_load_buffer=True, only_models = False):
        """Load all model states and training states from disk"""
        # Load network states using torch.load

        self.state_encoder.load_state_dict(torch.load(os.path.join(path, "state_encoder.pth")))
        self.policy_net.load_state_dict(torch.load(os.path.join(path, "policy_net.pth")))
        self.q_net1.load_state_dict(torch.load(os.path.join(path, "q_net1.pth")))
        self.q_net2.load_state_dict(torch.load(os.path.join(path, "q_net2.pth")))
        
        # Update target networks
        self.target_q_net1 = deepcopy(self.q_net1)
        self.target_q_net2 = deepcopy(self.q_net2)
        if not only_models:
        # Load training state using pickle
            training_state_path = os.path.join(path, "training_state.pkl")
            if os.path.exists(training_state_path):
                with open(training_state_path, 'rb') as f:
                    training_state = pickle.load(f)
                
                # Restore training state attributes
                self.update_counter = training_state['update_counter']
                self.alpha = training_state['alpha']
                self.automatic_entropy_tuning = training_state['automatic_entropy_tuning']
                self.tau = training_state['tau']
                self.gamma = training_state['gamma']
                self.batch_size = training_state['batch_size']
                self.prioritized_replay = training_state['prioritized_replay']
                self.grad_clip = training_state['grad_clip']
                self.n_step = training_state['n_step']
                # self.deterministic_eval = training_state['deterministic_eval']
                self.target_network_update_freq = training_state['target_network_update_freq']
                self.buffer_capacity = training_state['buffer_capacity']
                self.buffer_alpha = training_state['buffer_alpha']
                self.buffer_beta = training_state['buffer_beta']
                self.buffer_beta_annealing = training_state['buffer_beta_annealing']
                # Also restore network config parameters
                self.action_dim = training_state.get('action_dim', self.action_dim) # Use current if not in old save
                self.action_state_dim = training_state.get('action_state_dim', self.action_state_dim)
                self.hidden_dim = training_state.get('hidden_dim', self.hidden_dim)
                self.layer_norm = training_state.get('layer_norm', self.layer_norm)
                self.actor_lr = training_state.get('actor_lr', self.actor_lr)
                self.critic_lr = training_state.get('critic_lr', self.critic_lr)
                self.entropy_lr = training_state.get('entropy_lr', self.entropy_lr)
                
                if self.automatic_entropy_tuning:
                    # Load and prepare log_alpha for training
                    loaded_log_alpha = training_state['log_alpha'].to(self.device)
                    self.log_alpha = loaded_log_alpha.clone().detach().requires_grad_(True)
                    self.target_entropy = training_state['target_entropy']
                    # Reinitialize alpha optimizer to use new log_alpha
                    lr = self.alpha_optimizer.param_groups[0]['lr']
                    self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            
            # Load replay buffer if requested
            if is_load_buffer:
                if self.prioritized_replay:
                    self.replay_buffer = PrioritizedReplayBuffer.load(path, self.buffer_capacity, self.buffer_alpha, self.buffer_beta)
                else:
                    self.replay_buffer = ReplayBuffer.load(path, self.buffer_capacity)
            else:
                if self.prioritized_replay:
                    self.replay_buffer = PrioritizedReplayBuffer(
                        capacity=self.buffer_capacity,
                        alpha=self.buffer_alpha,
                        beta=self.buffer_beta,
                        beta_annealing=self.buffer_beta_annealing
                    )
                else:
                    self.replay_buffer = ReplayBuffer(self.buffer_capacity)
            
            # Load n-step buffer if it exists
            if self.n_step > 1:
                loaded_n_step = NStepBuffer.load(path)
                if loaded_n_step is not None:
                    self.n_step_buffer = loaded_n_step
                else:
                    self.n_step_buffer = NStepBuffer(n_step=self.n_step, gamma=self.gamma)