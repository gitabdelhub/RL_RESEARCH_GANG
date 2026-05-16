#!/usr/bin/env python3
"""
Vaccine test with swarm of virtual patients
Simulates 10 patients over 180 days with fixed vaccination schedule
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.patient import VirtualPatient
from sim.swarm import simulate_swarm


def create_patients(n_patients: int = 10, seed: int = 42) -> list[VirtualPatient]:
    """
    Create virtual patients with realistic parameters
    
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
        half_life = rng.lognormal(mean=np.log(108), sigma=0.3)
        
        # Individual patient with their own RNG
        patient_rng = np.random.default_rng(seed + i + 1)
        
        patient = VirtualPatient(
            titer=0.02,  # Small initial titer
            half_life_days=half_life,
            boost_mean=1.0,
            boost_std=0.2,
            rng=patient_rng
        )
        patients.append(patient)
    
    return patients


def create_titer_figure(titer_ts: np.ndarray, vacc_days_set: set[int], output_path: Path):
    """
    Create and save titer evolution figure
    
    Args:
        titer_ts: Titer time series [days, patients]
        vacc_days_set: Set of vaccination days
        output_path: Path to save figure
    """
    n_days, n_patients = titer_ts.shape
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual patients (light lines)
    for i in range(n_patients):
        plt.plot(range(n_days), titer_ts[:, i], 
                alpha=0.3, linewidth=0.8, color='blue')
    
    # Plot statistics
    mean_titer = np.mean(titer_ts, axis=1)
    p10_titer = np.percentile(titer_ts, 10, axis=1)
    p90_titer = np.percentile(titer_ts, 90, axis=1)
    
    plt.plot(range(n_days), mean_titer, 'b-', linewidth=2.5, label='Mean')
    plt.plot(range(n_days), p10_titer, 'b--', linewidth=2, label='P10')
    plt.plot(range(n_days), p90_titer, 'b:', linewidth=2, label='P90')
    
    # Mark vaccination days
    for vacc_day in vacc_days_set:
        plt.axvline(x=vacc_day, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.xlabel('Days')
    plt.ylabel('Antibody Titer')
    plt.title('Antibody Titer Evolution (10 patients, 180 days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Titer figure saved to: {output_path}")


def create_protection_figure(protprob_ts: np.ndarray, prot_ts: np.ndarray, 
                           vacc_days_set: set[int], output_path: Path):
    """
    Create and save protection evolution figure
    
    Args:
        protprob_ts: Protection probability time series [days, patients]
        prot_ts: Binary protection time series [days, patients]
        vacc_days_set: Set of vaccination days
        output_path: Path to save figure
    """
    n_days, n_patients = protprob_ts.shape
    
    plt.figure(figsize=(12, 8))
    
    # Calculate metrics
    fraction_protected = np.mean(prot_ts, axis=1)
    mean_protprob = np.mean(protprob_ts, axis=1)
    
    # Plot protection metrics
    plt.plot(range(n_days), fraction_protected, 'g-', linewidth=2.5, 
             label='Fraction Protected')
    plt.plot(range(n_days), mean_protprob, 'b--', linewidth=2, 
             label='Mean Protection Probability')
    
    # Mark vaccination days
    for vacc_day in vacc_days_set:
        plt.axvline(x=vacc_day, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.xlabel('Days')
    plt.ylabel('Protection')
    plt.title('Protection Evolution (10 patients, 180 days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Protection figure saved to: {output_path}")


def calculate_metrics(protprob_ts: np.ndarray, prot_ts: np.ndarray, 
                     n_doses: int, days: int) -> dict:
    """
    Calculate simulation metrics
    
    Args:
        protprob_ts: Protection probability time series
        prot_ts: Binary protection time series
        n_doses: Number of vaccine doses per patient
        days: Number of simulation days
        
    Returns:
        Dictionary of metrics
    """
    # AUC calculations (using trapezoidal rule)
    fraction_protected = np.mean(prot_ts, axis=1)
    p10_protprob = np.percentile(protprob_ts, 10, axis=1)
    
    auc_fraction_protected = np.trapezoid(fraction_protected, dx=1.0) / days
    auc_protprob_p10 = np.trapezoid(p10_protprob, dx=1.0) / days
    
    return {
        'AUC_fraction_protected_mean': auc_fraction_protected,
        'AUC_protprob_p10': auc_protprob_p10,
        'total_doses_per_patient': n_doses,
        'simulation_days': days
    }


def main():
    """Main vaccine test simulation"""
    print("=== Vaccine Test Simulation ===")
    print("10 patients, 180 days, fixed schedule {0, 28}")
    
    # Create patients
    patients = create_patients(n_patients=10, seed=42)
    
    # Simulation parameters
    days = 180
    vacc_days_set = {0, 28}  # Fixed vaccination schedule
    
    # Run simulation
    titer_ts, protprob_ts, prot_ts = simulate_swarm(patients, days, vacc_days_set)
    
    # Create output directory
    results_dir = Path(__file__).parent.parent / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    create_titer_figure(titer_ts, vacc_days_set, results_dir / "vaccine_test_titer.png")
    create_protection_figure(protprob_ts, prot_ts, vacc_days_set, 
                           results_dir / "vaccine_test_protection.png")
    
    # Calculate and print metrics
    metrics = calculate_metrics(protprob_ts, prot_ts, len(vacc_days_set), days)
    
    print("\n=== Simulation Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n✅ Vaccine test completed successfully!")
    return metrics


if __name__ == "__main__":
    main()
