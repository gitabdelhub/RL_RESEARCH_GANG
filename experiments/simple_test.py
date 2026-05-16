#!/usr/bin/env python3
"""
Simple test script: 10 patients, 30 days simulation
Validates basic setup and shows antibody evolution over time
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


class SimplePatient:
    """Minimal patient model with antibody dynamics"""
    
    def __init__(self, patient_id: int):
        self.id = patient_id
        # Random initial antibody level (0-0.5)
        self.antibodies = np.random.uniform(0, 0.5)
        # Individual decay rate (0.01-0.05 per day)
        self.decay_rate = np.random.uniform(0.01, 0.05)
        # Individual response strength (0.5-2.0)
        self.response_strength = np.random.uniform(0.5, 2.0)
        
    def update(self, day: int):
        """Update antibody level for one day"""
        # Simple exponential decay
        self.antibodies *= (1 - self.decay_rate)
        
    def vaccinate(self, dose: float = 1.0):
        """Apply vaccine dose"""
        # Antibody boost proportional to response strength
        boost = dose * self.response_strength
        self.antibodies += boost
        # Cap at reasonable maximum
        self.antibodies = min(self.antibodies, 5.0)


def run_simple_simulation():
    """Run simulation with 10 patients for 30 days"""
    print("=== Simple Simulation Test ===")
    print("10 patients, 30 days, no vaccination")
    
    # Create patients
    patients = [SimplePatient(i) for i in range(10)]
    
    # Simulation parameters
    n_days = 30
    
    # Storage for results
    antibody_history = np.zeros((len(patients), n_days))
    
    # Run simulation
    for day in range(n_days):
        for i, patient in enumerate(patients):
            patient.update(day)
            antibody_history[i, day] = patient.antibodies
    
    # Print summary statistics
    print(f"\nDay {n_days}:")
    print(f"  Mean antibodies: {np.mean(antibody_history[:, -1]):.3f}")
    print(f"  Std antibodies: {np.std(antibody_history[:, -1]):.3f}")
    print(f"  Min antibodies: {np.min(antibody_history[:, -1]):.3f}")
    print(f"  Max antibodies: {np.max(antibody_history[:, -1]):.3f}")
    
    # Create simple visualization
    plt.figure(figsize=(10, 6))
    
    # Plot individual patients
    for i in range(len(patients)):
        plt.plot(range(n_days), antibody_history[i, :], 
                alpha=0.6, linewidth=1, label=f'Patient {i}')
    
    # Plot mean and percentiles
    mean_antibodies = np.mean(antibody_history, axis=0)
    p10_antibodies = np.percentile(antibody_history, 10, axis=0)
    p90_antibodies = np.percentile(antibody_history, 90, axis=0)
    
    plt.plot(range(n_days), mean_antibodies, 'k-', linewidth=2, label='Mean')
    plt.plot(range(n_days), p10_antibodies, 'k--', linewidth=1.5, label='P10')
    plt.plot(range(n_days), p90_antibodies, 'k:', linewidth=1.5, label='P90')
    
    plt.xlabel('Days')
    plt.ylabel('Antibody Level')
    plt.title('Simple Antibody Evolution (10 patients, 30 days)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    results_dir = Path(__file__).parent.parent / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "simple_test.png", dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {results_dir / 'simple_test.png'}")
    
    plt.show()
    
    print("\n✅ Simple test completed successfully!")
    return antibody_history


if __name__ == "__main__":
    run_simple_simulation()
