"""
Baseline vaccination policies
"""

import numpy as np
from typing import Set, List
from src.models.patient import VirtualPatient
from src.sim.swarm import simulate_swarm


def fixed_two_dose_schedule() -> Set[int]:
    """
    Fixed two-dose schedule: days 0 and 28
    
    Returns:
        Set of vaccination days: {0, 28}
    """
    return {0, 28}


def periodic_booster_schedule(period_days: int = 180, 
                             start_days: tuple = (0, 28), 
                             horizon_days: int = 365) -> Set[int]:
    """
    Periodic booster schedule with fixed period
    
    Args:
        period_days: Period between boosters (default 180)
        start_days: Initial vaccination days (default (0, 28))
        horizon_days: Simulation horizon (default 365)
        
    Returns:
        Set of vaccination days
    """
    vacc_days = set(start_days)
    
    # Add periodic boosters from the last start day
    last_start = max(start_days)
    next_booster = last_start + period_days
    
    while next_booster < horizon_days:
        vacc_days.add(next_booster)
        next_booster += period_days
    
    return vacc_days


def threshold_policy_days(patients: List[VirtualPatient], 
                         days: int, 
                         threshold: float = 0.5,
                         cooldown_days: int = 90,
                         max_doses: int = 4,
                         target_fraction: float = 0.8) -> Set[int]:
    """
    Adaptive threshold-based policy
    
    Args:
        patients: List of virtual patients
        days: Simulation horizon
        threshold: Protection threshold (default 0.5)
        cooldown_days: Minimum days between vaccinations (default 90)
        max_doses: Maximum number of doses per patient (default 4)
        target_fraction: Target fraction of protected patients (default 0.8)
        
    Returns:
        Set of vaccination days
    """
    vacc_days = set()
    last_vacc_day = -cooldown_days - 1  # Initialize to allow first vaccination
    doses_given = 0
    
    # Simulate day by day to make decisions
    for day in range(days):
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
            # Apply decay up to current day
            for d in range(day):
                temp_p.decay(dt_days=1.0)
            temp_patients.append(temp_p)
        
        # Apply previous vaccinations up to current day
        for vacc_day in vacc_days:
            if vacc_day <= day:
                for temp_p in temp_patients:
                    temp_p.vaccinate(dose=1.0)
                # Apply decay for remaining days
                for d in range(vacc_day + 1, day + 1):
                    for temp_p in temp_patients:
                        temp_p.decay(dt_days=1.0)
        
        # Calculate current protection metrics
        protected_count = sum(1 for p in temp_patients if p.is_protected())
        fraction_protected = protected_count / len(temp_patients)
        
        # Get p10 of protection probability
        protprobs = [p.protection_prob() for p in temp_patients]
        p10_protprob = np.percentile(protprobs, 10)
        
        # Decision rule: vaccinate if protection is low
        if (fraction_protected < target_fraction or 
            p10_protprob < threshold):
            vacc_days.add(day)
            last_vacc_day = day
            doses_given += 1
    
    return vacc_days
