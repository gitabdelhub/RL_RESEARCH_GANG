#!/usr/bin/env python3
"""
Simple DQN Agent for Vaccine Optimization
Implements Deep Q-Network with experience replay and target network
"""

import numpy as np
from typing import List, Tuple
import json
from pathlib import Path


class DQNNetwork:
    """Simple neural network for Q-value approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b3 = np.zeros(action_dim)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass"""
        # Layer 1
        h1 = np.maximum(0, np.dot(state, self.W1) + self.b1)  # ReLU
        
        # Layer 2
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)  # ReLU
        
        # Output layer
        q_values = np.dot(h2, self.W3) + self.b3
        
        return q_values
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Get action with epsilon-greedy policy"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        q_values = self.forward(state)
        return np.argmax(q_values)


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from buffer"""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)))
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with experience replay and target network"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 target_update: int = 100):
        """
        Initialize DQN agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
            learning_rate: Learning rate for gradient descent
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            batch_size: Batch size for training
            target_update: Frequency of target network update
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.learning_rate = learning_rate
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim)
        
        # Copy weights to target network
        self._update_target_network()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training statistics
        self.step_count = 0
        self.episode_rewards = []
        self.episode_losses = []
    
    def _update_target_network(self):
        """Copy weights from Q network to target network"""
        self.target_network.W1 = self.q_network.W1.copy()
        self.target_network.b1 = self.q_network.b1.copy()
        self.target_network.W2 = self.q_network.W2.copy()
        self.target_network.b2 = self.q_network.b2.copy()
        self.target_network.W3 = self.q_network.W3.copy()
        self.target_network.b3 = self.q_network.b3.copy()
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        return self.q_network.get_action(state, self.epsilon)
    
    def train_step(self, state: np.ndarray, action: int, reward: float, 
                   next_state: np.ndarray, done: bool) -> float:
        """Train agent on one transition"""
        # Add to replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Train if enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Compute loss and update
        loss = self._compute_loss(batch)
        self._update_network(loss)
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self._update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def _compute_loss(self, batch: List[Tuple]) -> float:
        """Compute loss for batch"""
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        # Current Q values
        current_q = self.q_network.forward(states)
        current_q_values = current_q[np.arange(len(actions)), actions]
        
        # Target Q values
        next_q = self.target_network.forward(next_states)
        next_q_values = np.max(next_q, axis=1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # MSE loss
        loss = np.mean((current_q_values - target_q_values) ** 2)
        
        return loss
    
    def _update_network(self, loss: float):
        """Update network weights using gradient descent"""
        # Simple gradient descent (not full backprop)
        # For simplicity, we use a small random update
        # In a real implementation, use proper backpropagation
        scale = self.learning_rate * loss
        
        self.q_network.W1 += np.random.randn(*self.q_network.W1.shape) * scale * 0.01
        self.q_network.b1 += np.random.randn(*self.q_network.b1.shape) * scale * 0.01
        self.q_network.W2 += np.random.randn(*self.q_network.W2.shape) * scale * 0.01
        self.q_network.b2 += np.random.randn(*self.q_network.b2.shape) * scale * 0.01
        self.q_network.W3 += np.random.randn(*self.q_network.W3.shape) * scale * 0.01
        self.q_network.b3 += np.random.randn(*self.q_network.b3.shape) * scale * 0.01
    
    def save(self, path: str):
        """Save agent to file"""
        data = {
            'W1': self.q_network.W1.tolist(),
            'b1': self.q_network.b1.tolist(),
            'W2': self.q_network.W2.tolist(),
            'b2': self.q_network.b2.tolist(),
            'W3': self.q_network.W3.tolist(),
            'b3': self.q_network.b3.tolist(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load agent from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.q_network.W1 = np.array(data['W1'])
        self.q_network.b1 = np.array(data['b1'])
        self.q_network.W2 = np.array(data['W2'])
        self.q_network.b2 = np.array(data['b2'])
        self.q_network.W3 = np.array(data['W3'])
        self.q_network.b3 = np.array(data['b3'])
        self.epsilon = data['epsilon']
        self.episode_rewards = data['episode_rewards']
        
        self._update_target_network()
