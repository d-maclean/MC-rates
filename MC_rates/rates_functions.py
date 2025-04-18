import numpy as np
from astropy import units as u, constants as const
import pandas as pd
from scipy.stats import norm

from numpy.typing import NDArray
from astropy.units import Quantity

# Madau & Fragos (2017)

def calc_mean_metallicity_madau_fragos(redshift: float | NDArray) -> float | NDArray:
    '''
    Get cosmic gas-phase metallicity means and functions using the best-fit function per Madau & Fragos 2017.
    ### Parameters:
    - z: redshift
    ### Returns:
    - mean_Z: average absolute metallicity [0<Z<1]
    '''
    return np.power(10, (0.153 - 0.074 * redshift ** 1.34 )) * 0.02

def calc_SFR_madau_fragos(redshift: float | NDArray) -> Quantity | NDArray:
    '''
    Cosmic redshift-dependent SFR as a function of redshift.
    Madau & Fragos 17.
    ### Parameters:
    - z: redshift value(s)
    ### Returns:
    - SFR: star formation rate (Msun/yr/Mpc^3)
    '''
    numerator = np.float_power((1 + redshift), 2.6)
    denominator = 1 + np.float_power((1 + redshift)/3.2, 6.2)
    units = u.Msun * u.year ** -1 * u.Mpc ** -3
    psi_z = 1e-2 * numerator/denominator * units

    return psi_z

def trivial_Pdet(data: dict) -> NDArray:
    '''Returns ones; for calculating intrinsic rates.'''
    n: int = data["bin_num"].shape[0]
    return np.ones(shape=n, dtype=float)
