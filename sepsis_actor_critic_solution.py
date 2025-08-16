# Sepsis Treatment Actor-Critic Solution
"""
Group ID:
Group Members Name with Student ID:
1. Student 1
2. Student 2
3. Student 3
4. Student 4

Remarks: Add here

Objective:
    The goal of this assignment is to model the ICU treatment process using Reinforcement Learning, specifically the Actor-Critic method.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Constants
TARGET_MEAN_BP = 90
TARGET_SPO2 = 98
TARGET_RESP_RATE = 16

def load_dataset(path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load and preprocess the sepsis dataset."""
    df = pd.read_csv(path)
    # timestamp to datetime and sorting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['subject_id', 'hadm_id', 'icustay_id', 'timestamp'])
    # encode gender
    df['gender_enc'] = (df['gender'] == 'M').astype(int)
    # encode actions
    action2id = {a: i for i, a in enumerate(sorted(df['action'].unique()))}
    df['action_id'] = df['action'].map(action2id)
    return df, action2id

class SepsisTreatmentEnv:
    """Environment that replays ICU stays as episodes."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.groups = list(df.groupby(['subject_id', 'hadm_id', 'icustay_id']))
        self.state_cols = ['mean_bp', 'spo2', 'resp_rate', 'age', 'gender_enc']
        self.current_group = None
        self.idx = 0

    def reset(self) -> np.ndarray:
        self.current_group = random.choice(self.groups)[1].reset_index(drop=True)
        self.idx = 0
        return self._row_to_state(self.current_group.loc[self.idx])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        row = self.current_group.loc[self.idx]
        reward = -((row.mean_bp - TARGET_MEAN_BP) ** 2 +
                   (row.spo2 - TARGET_SPO2) ** 2 +
                   (row.resp_rate - TARGET_RESP_RATE) ** 2)
        self.idx += 1
        done = self.idx >= len(self.current_group)
        if not done:
            next_state = self._row_to_state(self.current_group.loc[self.idx])
        else:
            next_state = np.zeros(len(self.state_cols), dtype=np.float32)
        return next_state, reward, done, {}

    def _row_to_state(self, row: pd.Series) -> np.ndarray:
        return row[self.state_cols].to_numpy(dtype=np.float32)

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor):
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value

@dataclass
class Transition:
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool

class A2CAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.memory: List[Transition] = []

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        logits, value = self.model(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.memory.append(Transition(log_prob, value, 0.0, False))
        return action.item()

    def store_reward(self, reward: float, done: bool):
        self.memory[-1].reward = reward
        self.memory[-1].done = done

    def finish_episode(self):
        R = 0
        returns = []
        for t in reversed(self.memory):
            if t.done:
                R = 0
            R = t.reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        value_loss = []
        for t, R in zip(self.memory, returns):
            advantage = R - t.value.item()
            policy_loss.append(-t.log_prob * advantage)
            value_loss.append(nn.functional.smooth_l1_loss(t.value.squeeze(0), torch.tensor([R])))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.memory.clear()


def plot_and_animate_rewards(rewards: List[float],
                             anim_path: str = 'actor_critic_training.gif') -> None:
    """Generate an animated GIF of training rewards."""
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Actor-Critic Training')
    line, = ax.plot([], [], lw=2)

    def update(frame: int):
        line.set_data(range(1, frame + 1), rewards[:frame])
        ax.relim()
        ax.autoscale_view()
        return line,

    ani = FuncAnimation(fig, update, frames=len(rewards), blit=True, repeat=False)
    ani.save(anim_path, writer=PillowWriter(fps=2))
    plt.close(fig)

if __name__ == '__main__':
    df, action2id = load_dataset('Sepsis_datset.csv')
    env = SepsisTreatmentEnv(df)
    agent = A2CAgent(state_dim=5, action_dim=len(action2id))

    num_episodes = 5
    rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward, done)
            state = next_state
            ep_reward += reward
        agent.finish_episode()
        rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward={ep_reward:.2f}")

    plot_and_animate_rewards(rewards)

    writeup = (
        "The Actor-Critic agent was trained on a dataset of ICU stays to learn treatment policies. "
        "Rewards are computed as the negative squared deviation of vital signs from healthy targets. "
        "During training the agent experiences sequences that represent individual ICU episodes. "
        "The policy and value networks are updated using the advantage between returns and the critic's value estimates. "
        "Although the dataset-driven environment does not react to the agent's chosen action, the training loop demonstrates how an on-policy method can be implemented. "
        "Across the short training run, episode rewards remain fairly stable, reflecting the deterministic nature of replayed data. "
        "Longer training with more expressive models and interaction with a responsive simulator would allow the agent to adaptively choose interventions, leading to more pronounced reward trends and the potential for policy improvement."
    )
    with open('actor_critic_writeup.txt', 'w') as f:
        f.write(writeup)
