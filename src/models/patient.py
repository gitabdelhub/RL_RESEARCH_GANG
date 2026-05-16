"""
Virtual Patient model with antibody dynamics and vaccine response
"""

import numpy as np
from typing import Optional


class VirtualPatient:
    """
    Virtual patient with individual immune response characteristics
    
    Attributes:
        titer: Current antibody titer level
        half_life_days: Antibody decay half-life in days
        boost_mean: Mean vaccine boost response
        boost_std: Standard deviation of vaccine boost response
        rng: Random number generator for individual variation
    """
    
    def __init__(self, 
                 titer: float = 0.02,
                 half_life_days: float = 108.0,
                 boost_mean: float = 1.0,
                 boost_std: float = 0.2,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize virtual patient
        
        Args:
            titer: Initial antibody titer level
            half_life_days: Antibody decay half-life in days
            boost_mean: Mean vaccine boost response
            boost_std: Standard deviation of vaccine boost response
            rng: Random number generator
        """
        self.titer = titer
        self.half_life_days = half_life_days
        self.boost_mean = boost_mean
        self.boost_std = boost_std
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def decay(self, dt_days: float = 1.0):
        """
        Apply exponential decay to antibody titer
        
        Args:
            dt_days: Time step in days
        """
        decay_factor = np.exp(-np.log(2) * dt_days / self.half_life_days)
        self.titer *= decay_factor
    
    def vaccinate(self, dose: float = 1.0):
        """
        Apply vaccine dose to patient
        
        Args:
            dose: Vaccine dose strength (default 1.0)
        """
        # Individual boost response with normal distribution
        boost = self.rng.normal(self.boost_mean, self.boost_std)
        boost = max(0, boost) * dose  # Ensure non-negative
        self.titer += boost
    
    def protection_prob(self, titer50: float = 0.202, k: float = 3.0, eps: float = 1e-9) -> float:
        """
        Calculate protection probability from antibody titer
        
        Args:
            titer50: Titer giving 50% protection (from Khoury 2021)
            k: Steepness parameter
            eps: Small value to avoid log(0)
            
        Returns:
            Protection probability between 0 and 1
        """
        # Logistic function based on log(titer/titer50)
        log_ratio = np.log(self.titer + eps) - np.log(titer50)
        prob = 1.0 / (1.0 + np.exp(-k * log_ratio))
        return prob
    
    def is_protected(self, threshold: float = 0.5) -> bool:
        """
        Check if patient is protected
        
        Args:
            threshold: Protection probability threshold (default 0.5)
            
        Returns:
            True if protected, False otherwise
        """
        return self.protection_prob() >= threshold
