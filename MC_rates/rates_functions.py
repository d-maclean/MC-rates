import numpy as np
from astropy import units as u, constants as const
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad

from numpy.typing import NDArray
from astropy.units import Quantity

# Madau & Fragos (2017)

def calc_mean_metallicity_madau_fragos(redshift: float | NDArray, Zsun: float = 0.02) -> float | NDArray:
    '''
    Get cosmic gas-phase metallicity means and functions using the best-fit function per Madau & Fragos 2017.
    ### Parameters:
    - z: redshift
    ### Returns:
    - mean_Z: average absolute metallicity [0<Z<1]
    '''
    return np.power(10, (0.153 - 0.074 * redshift ** 1.34 )) * Zsun


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


# filter pipeline for unwinnowed COSMIC output
def process_cosmic_models(bpp: pd.DataFrame) -> pd.DataFrame:
    '''
    Winnow a COSMIC output file down to BHNS systems.
    '''
    ecc_range, ecc_integral = calculate_ecc_integral(1000)
    
    bhns = get_bhns_systems(bpp)
    t_gw_sec = calculate_gw_timescale(bhns, ecc_range, ecc_integral)
    t_gw = (t_gw_sec * u.s).to(u.Myr)
    
    t_delay = bhns.tphys + t_gw.value)
    data = {"t_gw": t_gw.value, "t_delay": t_delay}
    bhns.assign(data)
    
    return bhns


# filter down to BHNS systems
def get_bhns_systems(bpp: pd.DataFrame) -> pd.DataFrame:
     '''
     Provides a df containing only the following bins:
     - at at least one time, both objects are either a BH or NS
     - the orbital separation is > 0 (non-disrupted)
     - the evol_type is 2 (kstar just changed)
     '''
     bhns = bpp[
               ((bpp.kstar_1 == 13) | (bpp.kstar_1 == 14))\
                    & ((bpp.kstar_2 == 13) | (bpp.kstar_2 == 14))\
                    & (bpp.sep > 0e0)\
                    & (bpp.evol_type == 2)
               ]
     
     return bhns


# functions for postprocessing COSMIC data & calculate GW timescale
def calculate_ecc_integral(n_pts: int = 1000) -> tuple[np.ndarray]:
     '''
     Calculate the integral on the RHS of the Peters Formula (Peters 1963)
     at n_pts points. 100 should be sufficient. As this depends
     only on e0, we can interpolate it to save some cycles.
     '''
     smol = 1e-9
     ecc_integral = np.zeros(n_pts)
     e0_range = np.linspace(0e0, 1e0-smol, n_pts)

     def ecc_integral_func(e: float) -> float:
          num = (e ** (29/19)) * (1 + (121/304) * (e ** 2)) ** (1181/2299)
          denom = (1 - (e ** 2)) ** (3/2)

          return num/denom

     for i, e0 in enumerate(e0_range):
          ecc_integral[i] = quad(ecc_integral_func, a=0e0, b=e0)[0]

     return e0_range, ecc_integral


# Peters (1964) eq. for t_gw
def calculate_gw_timescale(df: pd.DataFrame,
                              ecc_range: np.ndarray, ecc_int: np.ndarray) -> pd.Series:
     '''
     Calculate Peters' gravitational wave timescale (Peters 1963-4)
     from the COSMIC data.
     Outputs t_gw in seconds.
     '''

     G = 6.6743e-8 # cm3 g-1 s-2 (big G)
     C = 2.99792458e10 # cm s-1 (c)

     m1 = df.mass_1 * (1.98841e33) # Msun to g
     m2 = df.mass_2 * (1.98841e33)
     a0 = df.sep * (6.957e10) # Rsun to cm
     e0 = df.ecc

     # calculate beta
     beta = (64e0/5e0) * (m1 * m2 * (m1 + m2)) * (G ** 3) * (C ** -5)

     # get c0 proportionality constant
     c0 = a0 * (1 - (e0 ** 2)) * (e0 ** (-12/19))\
          * ( (1 + (121e0/304e0) * (e0 ** 2)) ** (-870/2299) )
     
     # get thetimescale
     t_gw = (12e0/19e0) * (c0 ** 4) * (beta ** -1)\
          * np.interp(e0, ecc_range, ecc_int)

     return t_gw
