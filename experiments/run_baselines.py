#!/usr/bin/env python3
"""
Run baseline vaccination policies comparison
Simulates 1000 patients over 365 days with 3 different policies
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.patient import VirtualPatient
from sim.swarm import simulate_swarm
from policies.baselines import (fixed_two_dose_schedule, 
                               periodic_booster_schedule,
                               threshold_policy_days)
from report.report_agent import (compute_summary_metrics, 
                                plot_titer, 
                                plot_protection,
                                print_metrics_table)


def create_realistic_patients(n_patients: int = 1000, seed: int = 42) -> list[VirtualPatient]:
    """
    Create realistic virtual patients with heterogeneous responses
    
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


def run_policy_simulation(patients: list[VirtualPatient], 
                          policy_name: str,
                          days: int = 365) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
    """
    Run simulation for a specific policy
    
    Args:
        patients: List of virtual patients
        policy_name: Name of the policy to run
        days: Simulation horizon
        
    Returns:
        Tuple of (titer_ts, protprob_ts, prot_ts, vacc_days_set)
    """
    print(f"Running {policy_name} policy...")
    
    # Get vaccination days based on policy
    if policy_name == "fixed_two_dose":
        vacc_days_set = fixed_two_dose_schedule()
    elif policy_name == "periodic_booster":
        vacc_days_set = periodic_booster_schedule(period_days=180, 
                                                 start_days=(0, 28), 
                                                 horizon_days=days)
    elif policy_name == "threshold_adaptive":
        vacc_days_set = threshold_policy_days(patients, days, 
                                             threshold=0.5,
                                             cooldown_days=90,
                                             max_doses=4,
                                             target_fraction=0.8)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    print(f"  Vaccination days: {sorted(vacc_days_set)}")
    
    # Reset patients to initial state
    for patient in patients:
        patient.titer = 0.02
    
    # Run simulation
    titer_ts, protprob_ts, prot_ts = simulate_swarm(patients, days, vacc_days_set)
    
    return titer_ts, protprob_ts, prot_ts, vacc_days_set


def main():
    """Main baseline comparison experiment"""
    print("=== Baseline Vaccination Policies Comparison ===")
    print("1000 patients, 365 days, 3 policies")
    
    # Create realistic patient cohort
    patients = create_realistic_patients(n_patients=1000, seed=42)
    print(f"Created {len(patients)} patients with heterogeneous responses")
    
    # Simulation parameters
    days = 365
    policies = ["fixed_two_dose", "periodic_booster", "threshold_adaptive"]
    
    # Storage for results
    all_metrics = {}
    time_array = np.arange(days)
    
    # Create output directory
    results_dir = Path(__file__).parent.parent / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run each policy
    for policy_name in policies:
        print(f"\n{'='*50}")
        
        # Run simulation
        titer_ts, protprob_ts, prot_ts, vacc_days_set = run_policy_simulation(
            patients, policy_name, days)
        
        # Compute metrics
        metrics = compute_summary_metrics(titer_ts, protprob_ts, prot_ts, vacc_days_set)
        all_metrics[policy_name] = metrics
        
        # Generate plots
        title_suffix = f" ({policy_name.replace('_', ' ').title()})"
        
        titer_path = results_dir / f"baseline_{policy_name}_titer.png"
        protection_path = results_dir / f"baseline_{policy_name}_protection.png"
        
        plot_titer(time_array, titer_ts, vacc_days_set, titer_path, title_suffix)
        plot_protection(time_array, prot_ts, protprob_ts, vacc_days_set, 
                       protection_path, title_suffix)
        
        print(f"  Figures saved to: {titer_path.name}, {protection_path.name}")
    
    # Print comparison table
    print_metrics_table(all_metrics)
    
    print(f"\n✅ Baseline comparison completed!")
    print(f"Results saved to: {results_dir}")
    return all_metrics


if __name__ == "__main__":
    main()
