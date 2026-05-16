"""
Swarm simulation for multiple virtual patients
"""

import numpy as np
from typing import List, Tuple, Set
from src.models.patient import VirtualPatient


def simulate_swarm(patients: List[VirtualPatient], 
                   days: int, 
                   vacc_days_set: Set[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate swarm of virtual patients over time
    
    Args:
        patients: List of virtual patients
        days: Number of days to simulate
        vacc_days_set: Set of days when vaccination occurs
        
    Returns:
        Tuple of (titer_ts, protprob_ts, prot_ts):
        - titer_ts: Titer time series [days, patients]
        - protprob_ts: Protection probability time series [days, patients]
        - prot_ts: Binary protection time series [days, patients]
    """
    n_patients = len(patients)
    
    # Initialize storage arrays
    titer_ts = np.zeros((days, n_patients))
    protprob_ts = np.zeros((days, n_patients))
    prot_ts = np.zeros((days, n_patients), dtype=bool)
    
    # Run simulation day by day
    for day in range(days):
        # Vaccinate all patients if scheduled
        if day in vacc_days_set:
            for patient in patients:
                patient.vaccinate(dose=1.0)
        
        # Apply decay and record state
        for i, patient in enumerate(patients):
            patient.decay(dt_days=1.0)
            
            # Record current state
            titer_ts[day, i] = patient.titer
            protprob_ts[day, i] = patient.protection_prob()
            prot_ts[day, i] = patient.is_protected()
    
    return titer_ts, protprob_ts, prot_ts
