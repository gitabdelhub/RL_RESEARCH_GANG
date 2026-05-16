"""
Report agent for visualization and metrics computation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Set, Dict, Tuple


def compute_summary_metrics(titer_ts: np.ndarray, 
                           protprob_ts: np.ndarray, 
                           prot_ts: np.ndarray, 
                           vacc_days_set: Set[int]) -> Dict[str, float]:
    """
    Compute summary metrics for vaccination policy evaluation
    
    Args:
        titer_ts: Titer time series [days, patients]
        protprob_ts: Protection probability time series [days, patients]
        prot_ts: Binary protection time series [days, patients]
        vacc_days_set: Set of vaccination days
        
    Returns:
        Dictionary of computed metrics
    """
    days, n_patients = titer_ts.shape
    
    # AUC calculations (normalized by time)
    fraction_protected = np.mean(prot_ts, axis=1)
    p10_protprob = np.percentile(protprob_ts, 10, axis=1)
    
    auc_fraction_protected_mean = np.trapz(fraction_protected, dx=1.0) / days
    auc_protprob_p10 = np.trapz(p10_protprob, dx=1.0) / days
    
    # Time above threshold (optional metric)
    time_above_threshold_mean = np.mean(fraction_protected >= 0.5)
    
    # Doses per patient
    doses_per_patient = len(vacc_days_set)
    
    # Robustness score (higher is better)
    score_robust = (auc_fraction_protected_mean + 
                   0.5 * auc_protprob_p10 - 
                   0.05 * doses_per_patient)
    
    return {
        'AUC_fraction_protected_mean': auc_fraction_protected_mean,
        'AUC_protprob_p10': auc_protprob_p10,
        'time_above_threshold_mean': time_above_threshold_mean,
        'doses_per_patient': doses_per_patient,
        'score_robust': score_robust
    }


def plot_titer(times: np.ndarray, 
               titer_ts: np.ndarray, 
               vacc_days: Set[int], 
               out_path: Path,
               title_suffix: str = ""):
    """
    Create titer evolution plot
    
    Args:
        times: Time array
        titer_ts: Titer time series [days, patients]
        vacc_days: Set of vaccination days
        out_path: Output file path
        title_suffix: Additional suffix for plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot individual patients (sample if too many)
    n_patients = titer_ts.shape[1]
    max_show = min(50, n_patients)  # Show at most 50 individual trajectories
    
    if n_patients <= max_show:
        # Show all patients
        for i in range(n_patients):
            plt.plot(times, titer_ts[:, i], alpha=0.2, linewidth=0.5, color='blue')
    else:
        # Sample patients to show
        sample_indices = np.random.choice(n_patients, max_show, replace=False)
        for i in sample_indices:
            plt.plot(times, titer_ts[:, i], alpha=0.2, linewidth=0.5, color='blue')
    
    # Plot statistics
    mean_titer = np.mean(titer_ts, axis=1)
    p10_titer = np.percentile(titer_ts, 10, axis=1)
    p90_titer = np.percentile(titer_ts, 90, axis=1)
    
    plt.plot(times, mean_titer, 'b-', linewidth=2.5, label='Mean')
    plt.plot(times, p10_titer, 'b--', linewidth=2, label='P10')
    plt.plot(times, p90_titer, 'b:', linewidth=2, label='P90')
    
    # Mark vaccination days
    for vacc_day in vacc_days:
        if vacc_day < len(times):
            plt.axvline(x=vacc_day, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.xlabel('Days')
    plt.ylabel('Antibody Titer')
    plt.title(f'Antibody Titer Evolution{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory


def plot_protection(times: np.ndarray, 
                   prot_ts: np.ndarray, 
                   protprob_ts: np.ndarray, 
                   vacc_days: Set[int], 
                   out_path: Path,
                   title_suffix: str = ""):
    """
    Create protection evolution plot
    
    Args:
        times: Time array
        prot_ts: Binary protection time series [days, patients]
        protprob_ts: Protection probability time series [days, patients]
        vacc_days: Set of vaccination days
        out_path: Output file path
        title_suffix: Additional suffix for plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate metrics
    fraction_protected = np.mean(prot_ts, axis=1)
    mean_protprob = np.mean(protprob_ts, axis=1)
    p10_protprob = np.percentile(protprob_ts, 10, axis=1)
    p90_protprob = np.percentile(protprob_ts, 90, axis=1)
    
    # Plot protection metrics
    plt.plot(times, fraction_protected, 'g-', linewidth=2.5, 
             label='Fraction Protected')
    plt.plot(times, mean_protprob, 'b--', linewidth=2, 
             label='Mean Protection Probability')
    plt.plot(times, p10_protprob, 'r:', linewidth=2, 
             label='P10 Protection Probability')
    plt.plot(times, p90_protprob, 'orange', linestyle='-.', linewidth=1.5, 
             label='P90 Protection Probability')
    
    # Mark vaccination days
    for vacc_day in vacc_days:
        if vacc_day < len(times):
            plt.axvline(x=vacc_day, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.xlabel('Days')
    plt.ylabel('Protection')
    plt.title(f'Protection Evolution{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory


def print_metrics_table(metrics_dict: Dict[str, Dict[str, float]]):
    """
    Print formatted metrics table
    
    Args:
        metrics_dict: Dictionary of metrics for each policy
    """
    print("\n" + "="*80)
    print("VACCINATION POLICY COMPARISON")
    print("="*80)
    
    # Header
    print(f"{'Policy':<25} | {'Doses':<6} | {'AUC_mean':<9} | {'AUC_p10':<9} | {'Score_robust':<12}")
    print("-" * 80)
    
    # Rows
    for policy_name, metrics in metrics_dict.items():
        print(f"{policy_name:<25} | {metrics['doses_per_patient']:<6.0f} | "
              f"{metrics['AUC_fraction_protected_mean']:<9.4f} | "
              f"{metrics['AUC_protprob_p10']:<9.4f} | "
              f"{metrics['score_robust']:<12.4f}")
    
    print("="*80)
