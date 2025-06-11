import numpy as np
from astropy import units as u, constants as const
import pandas as pd
from scipy.stats import norm, truncnorm
from scipy.integrate import quad

from numpy.typing import NDArray, ArrayLike
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

def calc_truncnorm_fractional_SFR(Zbins: ArrayLike,
                                  meanZ: ArrayLike, redshift: ArrayLike, 
                                  SFR_at_redshift: ArrayLike, sigma_logZ: float = 0.5) -> NDArray:
    '''
    Use a truncated log-normal distribution to calculate fractional SFR at each metallicity;
    notably, this method does not 'dump' SF outside the provided Zbins into the edges; instead,
    it rescales the bins to add up to 1.
    '''
    l_i: int = len(redshift)
    l_j: int = len(Zbins)
    lgZbins = np.log10(Zbins)
    
    # test PDFs
    Zpdf: NDArray = np.zeros(shape=(1000, l_i))
    test_range: NDArray = np.logspace(-8, 0, 1000)
    a_arr, b_arr = (lgZbins.min() - np.log10(meanZ)) / sigma_logZ,\
        (lgZbins.max() - np.log10(meanZ)) / sigma_logZ
    for i in range(l_i):
        _pdf = truncnorm.pdf(
            x=np.log10(test_range),
            a=a_arr[i],
            b=b_arr[i],
            loc=np.log10(meanZ[i]),
            scale=sigma_logZ
            )
        Zpdf[:,i] = _pdf / _pdf.sum()
    
    # calc Zbin edges
    Zbin_edges: NDArray = np.zeros(shape=l_j+1, dtype=float)
    for j in range(l_j):
        if j == 0:
            Zbin_edges[j+1] = np.mean(Zbins[j:j+2])
            Zbin_edges[j] = Zbins[j]
        elif j == l_j - 1:
            Zbin_edges[j] = np.mean(Zbins[j-1:j+1])
            Zbin_edges[j+1] = Zbins[j]
        else:
            Zbin_edges[j] = np.mean(Zbins[j-1:j+1])
            Zbin_edges[j+1] = np.mean(Zbins[j:j+2])
    
    # calc truncated pdf
    Zfracs: NDArray = np.zeros(shape=(l_j, l_i), dtype=float)
    for i in range(l_i):
        cdf = np.cumsum(Zpdf[:,i])
        fracs = np.zeros(shape=l_j)
        for j in range(l_j):
            area_below_edges = np.interp(Zbin_edges[j:j+2], test_range, cdf)
            fracs[j] = area_below_edges.max() - area_below_edges.min()
        Zfracs[:,i] = fracs
    
    # multiply result by SFR(z)
    fracSFR = Zfracs * SFR_at_redshift
    
    return fracSFR

def calc_adjusted_mean_for_truncnorm(desired_mean_Z: NDArray, Zmin: float, Zmax: float, sigma: float) -> NDArray:
    '''
    Returns an array of `adjusted` means for desired mean metalicities `desired_mean_Z`. This is necessary to correct
    for the skew of a truncated log-normal metallicity distribution.
    
    ### Parameters:
    - `desired_mean_Z`: NDArray - the `proper` mean metallicities
    - `Zmin`: float - the minimum bin value
    - `Zmax`: float - the maximum bin value
    - `sigma`: float - the log-standard deviation of the log-normal distribution
    ### Returns:
    - NDArray - the "adjusted" absolute metallicity values to use for your truncated log-normal pdf
    '''
    test_log_mu: NDArray = np.log10(np.logspace(-10, 0, 1000))
    fake_log_mu: NDArray = np.zeros(shape=1000)
    log_Zmin, log_Zmax = np.log10(Zmin), np.log10(Zmax)
    a, b = (log_Zmin - test_log_mu) / sigma, (log_Zmax - test_log_mu) / sigma

    for j in range(1000):
        output_log_mu = truncnorm.stats(a[j], b[j], loc=test_log_mu[j], scale=sigma, moments='m')
        fake_log_mu[j] = output_log_mu
    
    means_to_pass = np.interp(np.log10(desired_mean_Z), test_log_mu, fake_log_mu)
    
    return 10 ** (means_to_pass)


def calc_lognorm_fractional_SFR(Zbins: NDArray,
                            meanZ: NDArray, redshift: NDArray, 
                            SFR_at_redshift: NDArray, sigma_logZ: float = 0.5) -> NDArray:
    '''
    Use a log-normal SFR to calculate fractional SFR for each metallicity bin at each redshift/time
    value.
    '''
    l_i: int = len(redshift)
    l_j: int = len(Zbins)
    
    # test PDF over all metallicities
    Z_test_range: NDArray = np.logspace(-10, 0, 1000)
    Zpdf: NDArray = np.zeros(shape=(1000, l_i))
    for i in range(l_i):
        _pdf = norm.pdf(
            x=np.log10(Z_test_range),
            loc=np.log10(meanZ[i]),
            scale=sigma_logZ
            )
        Zpdf[:,i] = _pdf / _pdf.sum()

    # calc Zbin edges
    Zbin_edges: NDArray = np.zeros(shape=l_j+1, dtype=float)
    for j in range(l_j):
        if j == 0:
            Zbin_edges[j] = 0e0
            Zbin_edges[j+1] = np.mean(Zbins[j:j+2])

        elif j == l_j - 1:
            Zbin_edges[j] = np.mean(Zbins[j-1:j+1])
            Zbin_edges[j+1] = 1e0

        else:
            Zbin_edges[j] = np.mean(Zbins[j-1:j+1])
            Zbin_edges[j+1] = np.mean(Zbins[j:j+2])
            
    # get area under Zbin edges
    Zfracs: NDArray = np.zeros(shape=(l_j, l_i), dtype=float)
    for i in range(l_i):
        cdf: NDArray = np.cumsum(Zpdf[:,i])
        fracs = np.zeros(shape=l_j)
        for j in range(l_j):
            area_below_edges = np.interp(Zbin_edges[j:j+2], Z_test_range, cdf)
            fracs[j] = area_below_edges.max() - area_below_edges.min()
        Zfracs[:,i] = fracs
        
    # multiply result by SFR(z)
    fracSFR = Zfracs * SFR_at_redshift
    
    return fracSFR


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
    
    t_delay = bhns.tphys + t_gw.value
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
