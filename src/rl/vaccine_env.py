#!/usr/bin/env python3
"""
Vaccination Environment for Reinforcement Learning
OpenAI Gym-style environment for vaccine schedule optimization
"""

import numpy as np
from typing import Tuple, Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.patient import VirtualPatient
from sim.swarm import simulate_swarm
from policies.robust_search import clone_patient_cohort


class VaccineEnv:
    """
    Vaccination Environment for RL
    
    State: [day, mean_titer, std_titer, p10_titer, p90_titer, days_since_vacc, doses_given]
    Action: 0 (no vaccinate), 1 (vaccinate all)
    Reward: protection_mean - cost_per_dose * doses - penalty if p10 < threshold
    """
    
    def __init__(self, 
                 patients: list[VirtualPatient],
                 days: int = 365,
                 cost_per_dose: float = 0.5,  # Increased penalty
                 p10_threshold: float = 0.5,
                 p10_penalty: float = 0.1):
        """
        Initialize environment
        
        Args:
            patients: List of virtual patients
            days: Simulation horizon
            cost_per_dose: Cost penalty per vaccination
            p10_threshold: Minimum acceptable p10 protection
            p10_penalty: Penalty if p10 falls below threshold
        """
        self.original_patients = patients
        self.patients = clone_patient_cohort(patients, seed_offset=0)
        self.days = days
        self.cost_per_dose = cost_per_dose
        self.p10_threshold = p10_threshold
        self.p10_penalty = p10_penalty
        
        # Reset state
        self.current_day = 0
        self.doses_given = 0
        self.last_vacc_day = -100
        self.vacc_days_set = set()
        
        # State space dimension
        self.state_dim = 7
        self.action_dim = 2  # 0: no vaccinate, 1: vaccinate
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_day = 0
        self.doses_given = 0
        self.last_vacc_day = -100
        self.vacc_days_set = set()
        
        # Reset patients
        self.patients = clone_patient_cohort(self.original_patients, seed_offset=0)
        for patient in self.patients:
            patient.titer = 0.02
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if len(self.patients) == 0:
            return np.zeros(self.state_dim)
        
        # Get current titers
        titers = np.array([p.titer for p in self.patients])
        
        # Compute statistics
        mean_titer = np.mean(titers)
        std_titer = np.std(titers)
        p10_titer = np.percentile(titers, 10)
        p90_titer = np.percentile(titers, 90)
        
        # Normalize features
        state = np.array([
            self.current_day / self.days,  # Normalized day
            np.log1p(mean_titer),  # Log-transformed mean
            np.log1p(std_titer),  # Log-transformed std
            np.log1p(p10_titer),  # Log-transformed p10
            np.log1p(p90_titer),  # Log-transformed p90
            (self.current_day - self.last_vacc_day) / 100,  # Days since vacc
            self.doses_given / 10  # Normalized doses
        ])
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step
        
        Args:
            action: 0 (no vaccinate), 1 (vaccinate all)
            
        Returns:
            state, reward, done, info
        """
        # Apply action
        if action == 1:  # Vaccinate
            self.vacc_days_set.add(self.current_day)
            self.doses_given += 1
            self.last_vacc_day = self.current_day
            
            # Vaccinate all patients
            for patient in self.patients:
                patient.vaccinate()
        
        # Decay antibodies for all patients
        for patient in self.patients:
            patient.decay(dt_days=1)
        
        # Move to next day
        self.current_day += 1
        
        # Check if done
        done = self.current_day >= self.days
        
        # Compute reward
        reward = self._compute_reward()
        
        # Get new state
        state = self._get_state()
        
        # Info dict
        info = {
            'day': self.current_day,
            'doses': self.doses_given,
            'vacc_days': sorted(self.vacc_days_set)
        }
        
        return state, reward, done, info
    
    def _compute_reward(self) -> float:
        """Compute reward based on current protection"""
        if len(self.patients) == 0:
            return 0.0
        
        # Compute protection probabilities
        prot_probs = np.array([p.protection_prob() for p in self.patients])
        
        # Mean protection
        mean_protection = np.mean(prot_probs)
        
        # p10 protection
        p10_protection = np.percentile(prot_probs, 10)
        
        # Reward components
        reward = mean_protection
        reward -= self.cost_per_dose * self.doses_given  # Cost penalty
        
        # Penalty if p10 below threshold
        if p10_protection < self.p10_threshold:
            reward -= self.p10_penalty
        
        return reward
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final metrics after episode"""
        # Simulate full episode with current vaccination schedule
        eval_patients = clone_patient_cohort(self.original_patients, seed_offset=999)
        
        for patient in eval_patients:
            patient.titer = 0.02
        
        titer_ts, protprob_ts, prot_ts = simulate_swarm(
            eval_patients, self.days, self.vacc_days_set
        )
        
        # Compute metrics
        mean_protected = np.mean(prot_ts, axis=1)
        auc_mean = np.mean(mean_protected)
        
        p10_protprob = np.percentile(protprob_ts, 10, axis=1)
        auc_p10 = np.mean(p10_protprob)
        
        return {
            'AUC_mean': auc_mean,
            'AUC_p10': auc_p10,
            'doses': len(self.vacc_days_set),
            'vacc_days': sorted(self.vacc_days_set)
        }
