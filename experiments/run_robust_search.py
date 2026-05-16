#!/usr/bin/env python3
"""
Robust policy search with mean + p10 optimization
Finds optimal vaccination policy through random search
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
                                plot_protection,
                                print_metrics_table)


def create_realistic_patients(n_patients: int = 1000, seed: int = 123) -> list[VirtualPatient]:
    """
    Create realistic virtual patients with heterogeneous responses
    Same parameters as run_baselines.py for fair comparison
    
    Args:
        n_patients: Number of patients to create
        seed: Random seed for reproducibility
        
    Returns:
        List of VirtualPatient objects
    """
    rng = np.random.default_rng(seed)
    patients = []
    
    for i in range(n_patients):
        # Half-life around 108 days with lognormal distribution
        half_life = rng.lognormal(mean=np.log(108), sigma=0.35)
        
        # Boost response with lognormal distribution (some weak responders)
        boost_mean = rng.lognormal(mean=np.log(0.45), sigma=0.5)
        boost_std = 0.15 * boost_mean  # Proportional standard deviation
        
        # Individual patient with their own RNG
        patient_rng = np.random.default_rng(seed + i + 1000)
        
        patient = VirtualPatient(
            titer=0.02,  # Small initial titer
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
    """
    Evaluate a single configuration
    
    Args:
        patients: Patient cohort
        config: Configuration dictionary
        days: Simulation horizon
        lambda_robust: Weight for p10 in robust score
        alpha_dose: Cost per dose
        
    Returns:
        Results dictionary with metrics
    """
    # Clone patients for this evaluation
    eval_patients = clone_patient_cohort(patients, seed_offset=config.get('seed_offset', 0))
    
    # Generate vaccination days
    vacc_days_set = threshold_adaptive_policy_days(
        eval_patients, days,
        p10_threshold=config['p10_threshold'],
        cooldown_days=config['cooldown_days'],
        max_doses=config['max_doses'],
        prime_days=config.get('prime_days', (0, 28))
    )
    
    # Reset patients and run simulation
    for patient in eval_patients:
        patient.titer = 0.02
    
    titer_ts, protprob_ts, prot_ts = simulate_swarm(eval_patients, days, vacc_days_set)
    
    # Compute metrics
    metrics = compute_summary_metrics(titer_ts, protprob_ts, prot_ts, vacc_days_set)
    
    # Compute robust score
    score_robust = (metrics['AUC_fraction_protected_mean'] + 
                   lambda_robust * metrics['AUC_protprob_p10'] - 
                   alpha_dose * metrics['doses_per_patient'])
    
    # Add to results
    results = config.copy()
    results.update(metrics)
    results['score_robust'] = score_robust
    results['vacc_days'] = sorted(vacc_days_set)
    
    return results


def random_search(patients: list[VirtualPatient], 
                 n_trials: int = 100,
                 days: int = 365,
                 lambda_robust: float = 0.7,
                 alpha_dose: float = 0.05,
                 seed: int = 123) -> tuple[list, dict]:
    """
    Perform random search for optimal policy
    
    Args:
        patients: Patient cohort
        n_trials: Number of random trials
        days: Simulation horizon
        lambda_robust: Weight for p10 in robust score
        alpha_dose: Cost per dose
        seed: Random seed
        
    Returns:
        Tuple of (all_results, best_config)
    """
    rng = np.random.default_rng(seed)
    all_results = []
    best_score = -np.inf
    best_config = None
    
    print(f"Starting random search with {n_trials} trials...")
    print(f"{'Trial':<6} {'Score':<8} {'P10_thr':<8} {'Cooldown':<9} {'Max_doses':<9} {'Doses':<6} {'Best':<8}")
    print("-" * 80)
    
    for trial in range(n_trials):
        # Generate random configuration
        config = {
            'p10_threshold': rng.uniform(0.35, 0.75),
            'cooldown_days': rng.choice([30, 60, 90, 120, 150, 180]),
            'max_doses': rng.choice([2, 3, 4, 5, 6]),
            'seed_offset': trial * 1000  # Ensure different seeds for cloning
        }
        
        # Evaluate configuration
        results = evaluate_config(patients, config, days, lambda_robust, alpha_dose)
        all_results.append(results)
        
        # Update best if better
        is_new_best = results['score_robust'] > best_score
        if is_new_best:
            best_score = results['score_robust']
            best_config = results.copy()
        
        # Print trial results
        best_marker = "★" if is_new_best else " "
        print(f"{trial + 1:<6} {results['score_robust']:<8.4f} {config['p10_threshold']:<8.3f} "
              f"{config['cooldown_days']:<9} {config['max_doses']:<9} "
              f"{results['doses_per_patient']:<6} {best_marker:<8}")
        
        # Show best params if new best found
        if is_new_best:
            print(f"  → NEW BEST! Score: {best_score:.4f}, Params: p10={config['p10_threshold']:.3f}, "
                  f"cooldown={config['cooldown_days']}, max_doses={config['max_doses']}")
    
    print(f"\nSearch completed. Best score: {best_score:.4f}")
    return all_results, best_config


def run_baseline_comparison(patients: list[VirtualPatient], 
                           days: int = 365,
                           lambda_robust: float = 0.7,
                           alpha_dose: float = 0.05) -> dict:
    """
    Run baseline policies for comparison
    
    Args:
        patients: Patient cohort
        days: Simulation horizon
        lambda_robust: Weight for p10 in robust score
        alpha_dose: Cost per dose
        
    Returns:
        Dictionary of baseline results
    """
    baseline_results = {}
    
    # Fixed two-dose
    print("Running fixed_two_dose baseline...")
    vacc_days = fixed_two_dose_schedule()
    eval_patients = clone_patient_cohort(patients, seed_offset=9999)
    
    for patient in eval_patients:
        patient.titer = 0.02
    
    titer_ts, protprob_ts, prot_ts = simulate_swarm(eval_patients, days, vacc_days)
    metrics = compute_summary_metrics(titer_ts, protprob_ts, prot_ts, vacc_days)
    metrics['score_robust'] = (metrics['AUC_fraction_protected_mean'] + 
                              lambda_robust * metrics['AUC_protprob_p10'] - 
                              alpha_dose * metrics['doses_per_patient'])
    baseline_results['fixed_two_dose'] = metrics
    
    # Periodic booster
    print("Running periodic_booster baseline...")
    vacc_days = periodic_booster_schedule(period_days=180, start_days=(0, 28), horizon_days=days)
    eval_patients = clone_patient_cohort(patients, seed_offset=8888)
    
    for patient in eval_patients:
        patient.titer = 0.02
    
    titer_ts, protprob_ts, prot_ts = simulate_swarm(eval_patients, days, vacc_days)
    metrics = compute_summary_metrics(titer_ts, protprob_ts, prot_ts, vacc_days)
    metrics['score_robust'] = (metrics['AUC_fraction_protected_mean'] + 
                              lambda_robust * metrics['AUC_protprob_p10'] - 
                              alpha_dose * metrics['doses_per_patient'])
    baseline_results['periodic_booster'] = metrics
    
    return baseline_results


def save_results(best_config: dict, 
                all_results: list, 
                baseline_results: dict,
                results_dir: Path):
    """
    Save results to files
    
    Args:
        best_config: Best configuration found
        all_results: All trial results
        baseline_results: Baseline results
        results_dir: Results directory
    """
    # Save best config as JSON (convert numpy types to Python types)
    best_config_path = results_dir / "best_config.json"
    # Convert numpy types to Python types for JSON serialization
    json_config = {}
    for key, value in best_config.items():
        if hasattr(value, 'item'):  # numpy scalar
            json_config[key] = value.item()
        elif isinstance(value, np.ndarray):
            json_config[key] = value.tolist()
        else:
            json_config[key] = value
    
    with open(best_config_path, 'w') as f:
        json.dump(json_config, f, indent=2)
    print(f"Best config saved to: {best_config_path}")
    
    # Save summary table as CSV
    summary_data = []
    
    # Define common fields for CSV
    common_fields = ['policy', 'AUC_fraction_protected_mean', 'AUC_protprob_p10', 
                    'time_above_threshold_mean', 'doses_per_patient', 'score_robust']
    
    # Add baselines
    for name, metrics in baseline_results.items():
        row = {'policy': name}
        # Convert numpy types to Python types and only include common fields
        for key, value in metrics.items():
            if key in common_fields:
                if hasattr(value, 'item'):  # numpy scalar
                    row[key] = value.item()
                else:
                    row[key] = value
        summary_data.append(row)
    
    # Add robust best
    row = {'policy': 'robust_best'}
    for key, value in best_config.items():
        if key in common_fields:
            if hasattr(value, 'item'):  # numpy scalar
                row[key] = value.item()
            else:
                row[key] = value
    summary_data.append(row)
    
    # Write CSV manually
    csv_path = results_dir / "summary_table.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        if summary_data:
            fieldnames = common_fields
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
    print(f"Summary table saved to: {csv_path}")


def main():
    """Main robust search experiment"""
    print("=== Robust Policy Search ===")
    print("1000 patients, 365 days, random search optimization")
    
    # Parameters
    n_patients = 1000
    days = 365
    n_trials = 10  # DEBUG: Start with 10 trials for fast testing
    lambda_robust = 0.7
    alpha_dose = 0.05
    seed = 123
    
    print(f"DEBUG: n_trials = {n_trials} (fast testing)")
    
    # Create patient cohort (once!)
    patients = create_realistic_patients(n_patients=n_patients, seed=seed)
    print(f"Created {len(patients)} patients with heterogeneous responses")
    
    # Create output directories
    results_dir = Path(__file__).parent.parent / "results" / "robust"
    figures_dir = Path(__file__).parent.parent / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Run random search
    print(f"\n{'='*60}")
    all_results, best_config = random_search(
        patients, n_trials, days, lambda_robust, alpha_dose, seed)
    
    # Run baseline comparison
    print(f"\n{'='*60}")
    baseline_results = run_baseline_comparison(patients, days, lambda_robust, alpha_dose)
    
    # Save results
    save_results(best_config, all_results, baseline_results, results_dir)
    
    # Generate plots for best config
    print(f"\n{'='*60}")
    print("Generating plots for best configuration...")
    
    eval_patients = clone_patient_cohort(patients, seed_offset=7777)
    for patient in eval_patients:
        patient.titer = 0.02
    
    vacc_days_set = set(best_config['vacc_days'])
    titer_ts, protprob_ts, prot_ts = simulate_swarm(eval_patients, days, vacc_days_set)
    
    time_array = np.arange(days)
    title_suffix = f" (Robust Best - Score: {best_config['score_robust']:.4f})"
    
    plot_titer(time_array, titer_ts, vacc_days_set, 
              figures_dir / "robust_best_titer.png", title_suffix)
    plot_protection(time_array, prot_ts, protprob_ts, vacc_days_set,
                    figures_dir / "robust_best_protection.png", title_suffix)
    
    # Print top 10 configurations
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS")
    print("="*80)
    
    # Sort by score and get top 10
    sorted_results = sorted(all_results, key=lambda x: x['score_robust'], reverse=True)[:10]
    
    print(f"{'Rank':<5} {'Score':<8} {'P10_thr':<8} {'Cooldown':<9} {'Max_doses':<9} {'Doses':<6} {'AUC_mean':<9} {'AUC_p10':<8}")
    print("-" * 80)
    
    for i, config in enumerate(sorted_results):
        print(f"{i+1:<5} {config['score_robust']:<8.4f} {config['p10_threshold']:<8.3f} "
              f"{config['cooldown_days']:<9} {config['max_doses']:<9} "
              f"{config['doses_per_patient']:<6} {config['AUC_fraction_protected_mean']:<9.4f} "
              f"{config['AUC_protprob_p10']:<8.4f}")
    
    # Print final comparison table
    print(f"\n{'='*80}")
    print("FINAL COMPARISON (Baselines + Robust Best)")
    print("="*80)
    
    comparison_data = {}
    for name, metrics in baseline_results.items():
        comparison_data[name] = metrics
    comparison_data['robust_best'] = best_config
    
    print_metrics_table(comparison_data)
    
    print(f"\n✅ Robust search completed!")
    print(f"Best robust score: {best_config['score_robust']:.4f}")
    print(f"Best parameters: p10_threshold={best_config['p10_threshold']:.3f}, "
          f"cooldown={best_config['cooldown_days']}, max_doses={best_config['max_doses']}")
    
    return best_config, all_results


if __name__ == "__main__":
    main()
