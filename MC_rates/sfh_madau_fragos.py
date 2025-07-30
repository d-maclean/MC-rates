import numpy as np
from scipy.stats import norm, truncnorm
from astropy import units as u

from astropy.units import Quantity 
from numpy.typing import NDArray


# Madau & Fragos (2017)

def madau_fragos_SFH(redshift: NDArray, metallicities: NDArray,
                     sigma: float,truncate_lognorm: bool, Zsun: float = 0.017) -> dict:
    '''
    '''    
    Zmin = metallicities.min()
    Zmax = metallicities.max()
    mean_Z: NDArray = calc_mean_metallicity_madau_fragos(redshift, Zsun=Zsun)
    SFR_at_z = calc_SFR_madau_fragos(redshift)
    
    if truncate_lognorm:
        adjusted_mean_Z = calc_adjusted_mean_for_truncnorm(mean_Z, Zmin, Zmax, sigma)
        fracSFR = calc_truncnorm_fractional_SFR(metallicities, adjusted_mean_Z, redshift, SFR_at_z, sigma)
    else:
        fracSFR = calc_lognorm_fractional_SFR(metallicities, mean_Z, redshift, SFR_at_z, sigma)
    
    sfh = {
        "redshift": redshift,
        "metallicities": metallicities,
        "SFR_at_z": SFR_at_z,
        "mean_metallicity": mean_Z,
        "fractional_SFR": fracSFR,
    }
    
    return sfh

def calc_mean_metallicity_madau_fragos(redshift: NDArray, Zsun: float = 0.02) -> NDArray:
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
    units = u.Msun * u.year ** -1 * u.Mpc ** -3 #type: ignore
    psi_z = 1e-2 * numerator/denominator * units

    return psi_z

def calc_truncnorm_fractional_SFR(Zbins: NDArray,
                                  meanZ: NDArray, redshift: NDArray, 
                                  SFR_at_redshift: NDArray, sigma_logZ: float = 0.5) -> NDArray:
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
    n_test: int = 1000
    test_log_mu: NDArray = np.log10(np.logspace(-10, 0, n_test))
    fake_log_mu: NDArray = np.zeros(shape=n_test)
    log_Zmin, log_Zmax = np.log10(Zmin), np.log10(Zmax)
    a, b = (log_Zmin - test_log_mu) / sigma, (log_Zmax - test_log_mu) / sigma

    for j in range(n_test):
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
