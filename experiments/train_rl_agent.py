#!/usr/bin/env python3
"""
Train DQN agent for vaccine schedule optimization
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.patient import VirtualPatient
from rl.vaccine_env import VaccineEnv
from rl.dqn_agent import DQNAgent


def create_realistic_patients(n_patients: int = 1000, seed: int = 123) -> list[VirtualPatient]:
    """Create realistic virtual patients with heterogeneous responses"""
    rng = np.random.default_rng(seed)
    patients = []
    
    for i in range(n_patients):
        half_life = rng.lognormal(mean=np.log(108), sigma=0.35)
        boost_mean = rng.lognormal(mean=np.log(0.45), sigma=0.5)
        boost_std = 0.15 * boost_mean
        patient_rng = np.random.default_rng(seed + i + 1000)
        
        patient = VirtualPatient(
            titer=0.02,
            half_life_days=half_life,
            boost_mean=boost_mean,
            boost_std=boost_std,
            rng=patient_rng
        )
        patients.append(patient)
    
    return patients


def train_agent(n_episodes: int = 100, n_patients: int = 1000, seed: int = 123):
    """Train DQN agent"""
    print("=== Training DQN Agent for Vaccine Optimization ===")
    
    # Create patients
    patients = create_realistic_patients(n_patients=n_patients, seed=seed)
    print(f"Created {len(patients)} patients")
    
    # Create environment
    env = VaccineEnv(patients, days=365, cost_per_dose=0.5, p10_threshold=0.5, p10_penalty=0.1)
    
    # Create agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=32,
        target_update=100
    )
    
    # Training loop
    episode_rewards = []
    episode_doses = []
    
    print(f"\nTraining for {n_episodes} episodes...")
    print(f"{'Episode':<8} {'Reward':<10} {'Doses':<6} {'Epsilon':<8}")
    print("-" * 40)
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Train agent
            loss = agent.train_step(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_doses.append(info['doses'])
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_doses = np.mean(episode_doses[-10:])
            print(f"{episode + 1:<8} {avg_reward:<10.4f} {avg_doses:<6.1f} {agent.epsilon:<8.4f}")
    
    print("\nTraining completed!")
    
    # Save agent
    save_path = Path(__file__).parent.parent / "results" / "rl" / "dqn_agent.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_path))
    print(f"Agent saved to: {save_path}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_doses, save_path.parent)
    
    # Evaluate final policy
    print("\n=== Evaluating Final Policy ===")
    final_metrics = evaluate_agent(env, agent, n_eval=5)
    print(f"Final AUC_mean: {final_metrics['AUC_mean']:.4f}")
    print(f"Final AUC_p10: {final_metrics['AUC_p10']:.4f}")
    print(f"Final doses: {final_metrics['doses']}")
    print(f"Vaccination days: {final_metrics['vacc_days']}")
    
    return agent, episode_rewards, episode_doses, final_metrics


def evaluate_agent(env: VaccineEnv, agent: DQNAgent, n_eval: int = 5) -> dict:
    """Evaluate agent over multiple episodes"""
    metrics_list = []
    
    for _ in range(n_eval):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)  # No exploration
            state, reward, done, info = env.step(action)
        
        metrics = env.get_final_metrics()
        metrics_list.append(metrics)
    
    # Average metrics
    avg_metrics = {
        'AUC_mean': np.mean([m['AUC_mean'] for m in metrics_list]),
        'AUC_p10': np.mean([m['AUC_p10'] for m in metrics_list]),
        'doses': np.mean([m['doses'] for m in metrics_list]),
        'vacc_days': metrics_list[0]['vacc_days']  # Use first episode's schedule
    }
    
    return avg_metrics


def plot_training_curves(rewards: list, doses: list, save_dir: Path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Rewards
    ax1.plot(rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True)
    
    # Moving average
    window = 10
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 'r-', label=f'Moving Avg ({window})')
        ax1.legend()
    
    # Doses
    ax2.plot(doses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Doses')
    ax2.set_title('Vaccination Doses per Episode')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_dir / 'training_curves.png'}")
    plt.close()


def main():
    """Main training function"""
    agent, rewards, doses, metrics = train_agent(n_episodes=50, n_patients=1000, seed=123)
    
    print("\n=== Training Summary ===")
    print(f"Final average reward: {np.mean(rewards[-10:]):.4f}")
    print(f"Final average doses: {np.mean(doses[-10:]):.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")


if __name__ == "__main__":
    main()
