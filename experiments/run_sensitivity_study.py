#!/usr/bin/env python3
"""
Sensitivity study: Fast waning scenario (half_life_days = 60 instead of 108)
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
from policies.robust_search import (threshold_adaptive_policy_days,
                                    clone_patient_cohort)
from report.report_agent import (compute_summary_metrics, 
                                plot_titer, 
                                plot_protection)


def create_fast_waning_patients(n_patients: int = 1000, seed: int = 456) -> list[VirtualPatient]:
    """Create virtual patients with fast waning (half_life_days = 60)"""
    rng = np.random.default_rng(seed)
    patients = []
    
    for i in range(n_patients):
        # Fast waning: half-life around 60 days instead of 108
        half_life = rng.lognormal(mean=np.log(60), sigma=0.35)
        
        # Same boost response distribution
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


def run_policy_comparison(patients: list[VirtualPatient], 
                        scenario_name: str,
                        days: int = 365,
                        lambda_robust: float = 0.7,
                        alpha_dose: float = 0.05) -> dict:
    """Run baseline + robust best comparison"""
    
    # Load robust best config from previous run
    config_path = Path(__file__).parent.parent / "results" / "robust" / "best_config.json"
    with open(config_path, 'r') as f:
        robust_config = json.load(f)
    
    results = {}
    figures_dir = Path(__file__).parent.parent / "results" / "figures"
    
    # Fixed two-dose
    print(f"  Running fixed_two_dose ({scenario_name})...")
    vacc_days = fixed_two_dose_schedule()
    eval_patients = clone_patient_cohort(patients, seed_offset=9999)
    
    for patient in eval_patients:
        patient.titer = 0.02
    
    titer_ts, protprob_ts, prot_ts = simulate_swarm(eval_patients, days, vacc_days)
    metrics = compute_summary_metrics(titer_ts, protprob_ts, prot_ts, vacc_days)
    metrics['score_robust'] = (metrics['AUC_fraction_protected_mean'] + 
                              lambda_robust * metrics['AUC_protprob_p10'] - 
                              alpha_dose * metrics['doses_per_patient'])
    results['fixed_two_dose'] = metrics
    
    # Generate figures
    time_array = np.arange(days)
    title_suffix = f" ({scenario_name})"
    plot_titer(time_array, titer_ts, vacc_days, 
              figures_dir / f"fixed_two_dose_{scenario_name}_titer.png", title_suffix)
    plot_protection(time_array, prot_ts, protprob_ts, vacc_days,
                    figures_dir / f"fixed_two_dose_{scenario_name}_protection.png", title_suffix)
    
    # Periodic booster
    print(f"  Running periodic_booster ({scenario_name})...")
    vacc_days = periodic_booster_schedule(period_days=180, start_days=(0, 28), horizon_days=days)
    eval_patients = clone_patient_cohort(patients, seed_offset=8888)
    
    for patient in eval_patients:
        patient.titer = 0.02
    
    titer_ts, protprob_ts, prot_ts = simulate_swarm(eval_patients, days, vacc_days)
    metrics = compute_summary_metrics(titer_ts, protprob_ts, prot_ts, vacc_days)
    metrics['score_robust'] = (metrics['AUC_fraction_protected_mean'] + 
                              lambda_robust * metrics['AUC_protprob_p10'] - 
                              alpha_dose * metrics['doses_per_patient'])
    results['periodic_booster'] = metrics
    
    plot_titer(time_array, titer_ts, vacc_days, 
              figures_dir / f"periodic_booster_{scenario_name}_titer.png", title_suffix)
    plot_protection(time_array, prot_ts, protprob_ts, vacc_days,
                    figures_dir / f"periodic_booster_{scenario_name}_protection.png", title_suffix)
    
    # Robust best
    print(f"  Running robust_best ({scenario_name})...")
    vacc_days_set = set(robust_config['vacc_days'])
    eval_patients = clone_patient_cohort(patients, seed_offset=7777)
    
    for patient in eval_patients:
        patient.titer = 0.02
    
    titer_ts, protprob_ts, prot_ts = simulate_swarm(eval_patients, days, vacc_days_set)
    metrics = compute_summary_metrics(titer_ts, protprob_ts, prot_ts, vacc_days_set)
    metrics['score_robust'] = (metrics['AUC_fraction_protected_mean'] + 
                              lambda_robust * metrics['AUC_protprob_p10'] - 
                              alpha_dose * metrics['doses_per_patient'])
    results['robust_best'] = metrics
    
    plot_titer(time_array, titer_ts, vacc_days_set, 
              figures_dir / f"robust_best_{scenario_name}_titer.png", title_suffix)
    plot_protection(time_array, prot_ts, protprob_ts, vacc_days_set,
                    figures_dir / f"robust_best_{scenario_name}_protection.png", title_suffix)
    
    return results


def main():
    """Main sensitivity study"""
    print("=== SENSITIVITY STUDY: Fast Waning ===")
    print("Testing scenario: half_life_days = 60 (fast waning)")
    
    # Parameters
    n_patients = 1000
    days = 365
    lambda_robust = 0.7
    alpha_dose = 0.05
    seed = 456  # Different seed for sensitivity
    
    # Create fast waning patient cohort
    patients = create_fast_waning_patients(n_patients=n_patients, seed=seed)
    print(f"Created {len(patients)} patients with fast waning (median half-life = 60 days)")
    
    # Create output directory
    results_dir = Path(__file__).parent.parent / "results" / "robust"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    print(f"\n{'='*60}")
    results = run_policy_comparison(patients, "fastwaning", days, lambda_robust, alpha_dose)
    
    # Save sensitivity results
    csv_path = results_dir / "sensitivity.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['scenario', 'policy', 'AUC_fraction_protected_mean', 
                     'AUC_protprob_p10', 'doses_per_patient', 'score_robust']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for policy_name, metrics in results.items():
            row = {'scenario': 'fastwaning', 'policy': policy_name}
            for key, value in metrics.items():
                if key in fieldnames:
                    if hasattr(value, 'item'):
                        row[key] = value.item()
                    else:
                        row[key] = value
            writer.writerow(row)
    
    print(f"\nSensitivity results saved to: {csv_path}")
    
    # Print sensitivity table
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS (Fast Waning)")
    print("="*80)
    print(f"{'Policy':<20} {'AUC_mean':<9} {'AUC_p10':<8} {'Doses':<6} {'Score':<8}")
    print("-" * 80)
    
    for policy_name, metrics in results.items():
        print(f"{policy_name:<20} {metrics['AUC_fraction_protected_mean']:<9.4f} "
              f"{metrics['AUC_protprob_p10']:<8.4f} {metrics['doses_per_patient']:<6} "
              f"{metrics['score_robust']:<8.4f}")
    
    print("="*80)
    print("✅ Sensitivity study completed!")
    
    return results


if __name__ == "__main__":
    main()
