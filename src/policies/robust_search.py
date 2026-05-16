"""
Robust policy search with parameterized adaptive threshold policy
"""

import numpy as np
from typing import Set, List, Tuple
from src.models.patient import VirtualPatient


def threshold_adaptive_policy_days(patients: List[VirtualPatient], 
                                  days: int, 
                                  p10_threshold: float = 0.5,
                                  cooldown_days: int = 90,
                                  max_doses: int = 4,
                                  prime_days: Tuple[int, int] = (0, 28)) -> Set[int]:
    """
    Parameterized adaptive threshold policy for robust optimization
    
    Args:
        patients: List of virtual patients
        days: Simulation horizon
        p10_threshold: P10 protection probability threshold for vaccination
        cooldown_days: Minimum days between vaccinations
        max_doses: Maximum number of doses per patient
        prime_days: Initial vaccination days (default (0, 28))
        
    Returns:
        Set of vaccination days
    """
    vacc_days = set()
    last_vacc_day = -cooldown_days - 1  # Initialize to allow first vaccination
    doses_given = 0
    
    # Add prime days if within horizon
    for prime_day in prime_days:
        if prime_day < days:
            vacc_days.add(prime_day)
            last_vacc_day = max(last_vacc_day, prime_day)
            doses_given += 1
    
    # Simulate day by day to make adaptive decisions
    for day in range(days):
        # Skip if already a vaccination day (prime days)
        if day in vacc_days:
            continue
            
        # Check constraints
        if (day - last_vacc_day < cooldown_days or 
            doses_given >= max_doses):
            continue
        
        # Quick simulation to assess current protection status
        # Create temporary patients for assessment
        temp_patients = []
        for p in patients:
            # Copy patient state
            temp_p = VirtualPatient(
                titer=p.titer,
                half_life_days=p.half_life_days,
                boost_mean=p.boost_mean,
                boost_std=p.boost_std,
                rng=p.rng
            )
            temp_patients.append(temp_p)
        
        # Apply previous vaccinations up to current day
        for vacc_day in sorted(vacc_days):
            if vacc_day <= day:
                for temp_p in temp_patients:
                    temp_p.vaccinate(dose=1.0)
                # Apply decay for remaining days
                for d in range(vacc_day + 1, day + 1):
                    for temp_p in temp_patients:
                        temp_p.decay(dt_days=1.0)
        
        # Calculate P10 of protection probability
        protprobs = [p.protection_prob() for p in temp_patients]
        p10_protprob = np.percentile(protprobs, 10)
        
        # Decision rule: vaccinate if P10 protection is below threshold
        if p10_protprob < p10_threshold:
            vacc_days.add(day)
            last_vacc_day = day
            doses_given += 1
    
    return vacc_days


def clone_patient(patient: VirtualPatient, seed_offset: int = 0) -> VirtualPatient:
    """
    Create a clone of a patient with identical parameters
    
    Args:
        patient: Patient to clone
        seed_offset: Offset for random seed
        
    Returns:
        Cloned patient with same parameters
    """
    # Create new patient with same parameters (use deterministic seed)
    cloned = VirtualPatient(
        titer=patient.titer,
        half_life_days=patient.half_life_days,
        boost_mean=patient.boost_mean,
        boost_std=patient.boost_std,
        rng=np.random.default_rng(seed_offset)  # Use seed_offset directly
    )
    
    return cloned


def clone_patient_cohort(patients: List[VirtualPatient], seed_offset: int = 1000) -> List[VirtualPatient]:
    """
    Clone an entire patient cohort for multiple simulations
    
    Args:
        patients: Original patient cohort
        seed_offset: Base offset for random seeds
        
    Returns:
        Cloned patient cohort
    """
    cloned_patients = []
    for i, patient in enumerate(patients):
        cloned = clone_patient(patient, seed_offset + i)
        cloned_patients.append(cloned)
    
    return cloned_patients
