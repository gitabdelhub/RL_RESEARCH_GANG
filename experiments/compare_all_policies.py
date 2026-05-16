#!/usr/bin/env python3
"""
Compare all policies: RL agent vs baselines vs random search
"""

import numpy as np
import json
import csv
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.patient import VirtualPatient
from sim.swarm import simulate_swarm
from policies.baselines import (fixed_two_dose_schedule, 
                               periodic_booster_schedule)
from policies.robust_search import clone_patient_cohort
from report.report_agent import compute_summary_metrics
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


def evaluate_policy(patients: list[VirtualPatient], 
                   vacc_days_set: set,
                   days: int = 365,
                   lambda_robust: float = 0.7,
                   alpha_dose: float = 0.05) -> dict:
    """Evaluate a vaccination policy"""
    eval_patients = clone_patient_cohort(patients, seed_offset=9999)
    
    for patient in eval_patients:
        patient.titer = 0.02
    
    titer_ts, protprob_ts, prot_ts = simulate_swarm(eval_patients, days, vacc_days_set)
    metrics = compute_summary_metrics(titer_ts, protprob_ts, prot_ts, vacc_days_set)
    
    score_robust = (metrics['AUC_fraction_protected_mean'] + 
                   lambda_robust * metrics['AUC_protprob_p10'] - 
                   alpha_dose * metrics['doses_per_patient'])
    
    metrics['score_robust'] = score_robust
    return metrics


def main():
    """Compare all policies"""
    print("=== COMPARING ALL POLICIES ===")
    print("RL Agent vs Baselines vs Random Search")
    
    # Parameters
    n_patients = 1000
    days = 365
    lambda_robust = 0.7
    alpha_dose = 0.05
    seed = 123
    
    # Create patients
    patients = create_realistic_patients(n_patients=n_patients, seed=seed)
    print(f"Created {len(patients)} patients")
    
    # Load RL agent
    rl_agent_path = Path(__file__).parent.parent / "results" / "rl" / "dqn_agent.json"
    if rl_agent_path.exists():
        print(f"Loading RL agent from: {rl_agent_path}")
        agent = DQNAgent(state_dim=7, action_dim=2)
        agent.load(str(rl_agent_path))
        
        # Evaluate RL agent
        env = VaccineEnv(patients, days=days, cost_per_dose=alpha_dose, 
                        p10_threshold=0.5, p10_penalty=0.1)
        
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
        
        rl_metrics = env.get_final_metrics()
        rl_metrics['score_robust'] = (rl_metrics['AUC_mean'] + 
                                     lambda_robust * rl_metrics['AUC_p10'] - 
                                     alpha_dose * rl_metrics['doses'])
        print(f"RL Agent: AUC_mean={rl_metrics['AUC_mean']:.4f}, AUC_p10={rl_metrics['AUC_p10']:.4f}, "
              f"doses={rl_metrics['doses']:.1f}, score={rl_metrics['score_robust']:.4f}")
    else:
        print("RL agent not found, skipping RL evaluation")
        rl_metrics = None
    
    # Evaluate baselines
    print("\nEvaluating baselines...")
    
    # Fixed two-dose
    fixed_days = fixed_two_dose_schedule()
    fixed_metrics = evaluate_policy(patients, fixed_days, days, lambda_robust, alpha_dose)
    print(f"Fixed two-dose: AUC_mean={fixed_metrics['AUC_fraction_protected_mean']:.4f}, "
          f"AUC_p10={fixed_metrics['AUC_protprob_p10']:.4f}, doses={fixed_metrics['doses_per_patient']:.1f}, "
          f"score={fixed_metrics['score_robust']:.4f}")
    
    # Periodic booster
    periodic_days = periodic_booster_schedule(period_days=180, start_days=(0, 28), horizon_days=days)
    periodic_metrics = evaluate_policy(patients, periodic_days, days, lambda_robust, alpha_dose)
    print(f"Periodic booster: AUC_mean={periodic_metrics['AUC_fraction_protected_mean']:.4f}, "
          f"AUC_p10={periodic_metrics['AUC_protprob_p10']:.4f}, doses={periodic_metrics['doses_per_patient']:.1f}, "
          f"score={periodic_metrics['score_robust']:.4f}")
    
    # Load robust search best
    robust_config_path = Path(__file__).parent.parent / "results" / "robust" / "best_config.json"
    if robust_config_path.exists():
        with open(robust_config_path, 'r') as f:
            robust_config = json.load(f)
        
        robust_days = set(robust_config['vacc_days'])
        robust_metrics = evaluate_policy(patients, robust_days, days, lambda_robust, alpha_dose)
        print(f"Robust search: AUC_mean={robust_metrics['AUC_fraction_protected_mean']:.4f}, "
              f"AUC_p10={robust_metrics['AUC_protprob_p10']:.4f}, doses={robust_metrics['doses_per_patient']:.1f}, "
              f"score={robust_metrics['score_robust']:.4f}")
    else:
        print("Robust search config not found")
        robust_metrics = None
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("FINAL COMPARISON TABLE")
    print("="*80)
    print(f"{'Policy':<20} {'Doses':<8} {'AUC_mean':<10} {'AUC_p10':<10} {'Score':<10}")
    print("-" * 80)
    
    print(f"{'Fixed two-dose':<20} {fixed_metrics['doses_per_patient']:<8.1f} "
          f"{fixed_metrics['AUC_fraction_protected_mean']:<10.4f} "
          f"{fixed_metrics['AUC_protprob_p10']:<10.4f} "
          f"{fixed_metrics['score_robust']:<10.4f}")
    
    print(f"{'Periodic booster':<20} {periodic_metrics['doses_per_patient']:<8.1f} "
          f"{periodic_metrics['AUC_fraction_protected_mean']:<10.4f} "
          f"{periodic_metrics['AUC_protprob_p10']:<10.4f} "
          f"{periodic_metrics['score_robust']:<10.4f}")
    
    if robust_metrics:
        print(f"{'Robust search':<20} {robust_metrics['doses_per_patient']:<8.1f} "
              f"{robust_metrics['AUC_fraction_protected_mean']:<10.4f} "
              f"{robust_metrics['AUC_protprob_p10']:<10.4f} "
              f"{robust_metrics['score_robust']:<10.4f}")
    
    if rl_metrics:
        print(f"{'RL Agent (DQN)':<20} {rl_metrics['doses']:<8.1f} "
              f"{rl_metrics['AUC_mean']:<10.4f} "
              f"{rl_metrics['AUC_p10']:<10.4f} "
              f"{rl_metrics['score_robust']:<10.4f}")
    
    print("="*80)
    
    # Save comparison to CSV
    results_dir = Path(__file__).parent.parent / "results" / "rl"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = results_dir / "comparison_table.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['policy', 'doses', 'AUC_mean', 'AUC_p10', 'score_robust']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        writer.writerow({
            'policy': 'fixed_two_dose',
            'doses': fixed_metrics['doses_per_patient'],
            'AUC_mean': fixed_metrics['AUC_fraction_protected_mean'],
            'AUC_p10': fixed_metrics['AUC_protprob_p10'],
            'score_robust': fixed_metrics['score_robust']
        })
        
        writer.writerow({
            'policy': 'periodic_booster',
            'doses': periodic_metrics['doses_per_patient'],
            'AUC_mean': periodic_metrics['AUC_fraction_protected_mean'],
            'AUC_p10': periodic_metrics['AUC_protprob_p10'],
            'score_robust': periodic_metrics['score_robust']
        })
        
        if robust_metrics:
            writer.writerow({
                'policy': 'robust_search',
                'doses': robust_metrics['doses_per_patient'],
                'AUC_mean': robust_metrics['AUC_fraction_protected_mean'],
                'AUC_p10': robust_metrics['AUC_protprob_p10'],
                'score_robust': robust_metrics['score_robust']
            })
        
        if rl_metrics:
            writer.writerow({
                'policy': 'rl_agent_dqn',
                'doses': rl_metrics['doses'],
                'AUC_mean': rl_metrics['AUC_mean'],
                'AUC_p10': rl_metrics['AUC_p10'],
                'score_robust': rl_metrics['score_robust']
            })
    
    print(f"\nComparison table saved to: {csv_path}")
    print("✅ Comparison completed!")


if __name__ == "__main__":
    main()
