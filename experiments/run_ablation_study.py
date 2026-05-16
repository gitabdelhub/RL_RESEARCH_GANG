#!/usr/bin/env python3
"""
Ablation study: Effect of lambda_robust on optimal policy
Tests lambda values: 0, 0.5, 0.7, 1.0
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
from policies.robust_search import (threshold_adaptive_policy_days,
                                    clone_patient_cohort)
from report.report_agent import compute_summary_metrics


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


def evaluate_config(patients: list[VirtualPatient], 
                   config: dict,
                   days: int = 365,
                   lambda_robust: float = 0.7,
                   alpha_dose: float = 0.05) -> dict:
    """Evaluate a single configuration"""
    eval_patients = clone_patient_cohort(patients, seed_offset=config.get('seed_offset', 0))
    
    vacc_days_set = threshold_adaptive_policy_days(
        eval_patients, days,
        p10_threshold=config['p10_threshold'],
        cooldown_days=config['cooldown_days'],
        max_doses=config['max_doses'],
        prime_days=config.get('prime_days', (0, 28))
    )
    
    for patient in eval_patients:
        patient.titer = 0.02
    
    titer_ts, protprob_ts, prot_ts = simulate_swarm(eval_patients, days, vacc_days_set)
    metrics = compute_summary_metrics(titer_ts, protprob_ts, prot_ts, vacc_days_set)
    
    score_robust = (metrics['AUC_fraction_protected_mean'] + 
                   lambda_robust * metrics['AUC_protprob_p10'] - 
                   alpha_dose * metrics['doses_per_patient'])
    
    results = config.copy()
    results.update(metrics)
    results['score_robust'] = score_robust
    results['lambda_robust'] = lambda_robust
    results['vacc_days'] = sorted(vacc_days_set)
    
    return results


def random_search_lambda(patients: list[VirtualPatient], 
                        lambda_robust: float,
                        n_trials: int = 30,
                        days: int = 365,
                        alpha_dose: float = 0.05,
                        seed: int = 123) -> dict:
    """Random search for specific lambda value"""
    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best_config = None
    
    print(f"  Lambda={lambda_robust}: Running {n_trials} trials...")
    
    for trial in range(n_trials):
        config = {
            'p10_threshold': rng.uniform(0.35, 0.75),
            'cooldown_days': rng.choice([30, 60, 90, 120, 150, 180]),
            'max_doses': rng.choice([2, 3, 4, 5, 6]),
            'seed_offset': trial * 1000
        }
        
        results = evaluate_config(patients, config, days, lambda_robust, alpha_dose)
        
        if results['score_robust'] > best_score:
            best_score = results['score_robust']
            best_config = results.copy()
    
    print(f"    Best score: {best_score:.4f}")
    return best_config


def main():
    """Main ablation study"""
    print("=== ABLATION STUDY: Lambda Robustness ===")
    print("Testing lambda values: 0, 0.5, 0.7, 1.0")
    
    # Parameters
    n_patients = 1000
    days = 365
    n_trials = 30  # Fast for ablation
    alpha_dose = 0.05
    seed = 123
    lambda_values = [0.0, 0.5, 0.7, 1.0]
    
    # Create patient cohort (once!)
    patients = create_realistic_patients(n_patients=n_patients, seed=seed)
    print(f"Created {len(patients)} patients with heterogeneous responses")
    
    # Create output directory
    results_dir = Path(__file__).parent.parent / "results" / "robust"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run ablation study
    all_results = []
    
    for lambda_val in lambda_values:
        best_config = random_search_lambda(patients, lambda_val, n_trials, days, alpha_dose, seed)
        all_results.append(best_config)
    
    # Save ablation results
    csv_path = results_dir / "ablation.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['lambda_robust', 'score_robust', 'AUC_fraction_protected_mean', 
                     'AUC_protprob_p10', 'doses_per_patient', 'p10_threshold', 
                     'cooldown_days', 'max_doses']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            row = {}
            for field in fieldnames:
                value = result.get(field, 0)
                if hasattr(value, 'item'):
                    row[field] = value.item()
                else:
                    row[field] = value
            writer.writerow(row)
    
    print(f"\nAblation results saved to: {csv_path}")
    
    # Print ablation table
    print(f"\n{'='*80}")
    print("ABLATION ROBUSTNESS")
    print("="*80)
    print(f"{'Lambda':<8} {'Score':<8} {'AUC_mean':<9} {'AUC_p10':<8} {'Doses':<6} {'P10_thr':<8} {'Cooldown':<9}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['lambda_robust']:<8.1f} {result['score_robust']:<8.4f} "
              f"{result['AUC_fraction_protected_mean']:<9.4f} {result['AUC_protprob_p10']:<8.4f} "
              f"{result['doses_per_patient']:<6} {result['p10_threshold']:<8.3f} "
              f"{result['cooldown_days']:<9}")
    
    print("="*80)
    print("✅ Ablation study completed!")
    
    return all_results


if __name__ == "__main__":
    main()
