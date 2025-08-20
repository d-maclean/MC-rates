import os
import pandas as pd
import numpy as np
from astropy import units as u

from numpy.typing import NDArray

# code to implement the cosmic star formation density histoy (SFRD) method described in
# Chruslinska & Nelemans 2019. All data used in these calculations belong to the authors:
# https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.5300C/abstract
# this module:
# (C) Duncan B Maclean 2025 / BSD 3-Clause License

OPTIONS = {"moderate", "highZ", "lowZ"}


def chr_nel_SFH(comoving_time: NDArray, redshift: NDArray,
                      metallicities: NDArray, option: str = "moderate", Zsun: float = 0.017) -> dict:
    
    if option not in OPTIONS:
        raise ValueError(f"Invalid option. Please provide one of {[x for x in OPTIONS]}.")
    
    n_i = len(comoving_time)
    n_j = len(metallicities)
    
    t_edges = 10 ** (np.log10(comoving_time[:-1].value) + ((np.log10(comoving_time[1:].value) - np.log10(comoving_time[:-1].value))/2))
    t_edges = np.concat([[comoving_time[0]], t_edges * u.Myr, [comoving_time[-1]]])
    Z_edges = 10 ** (np.log10(metallicities[:-1]) + ((np.log10(metallicities[1:]) - np.log10(metallicities[:-1]))/2))
    Z_edges = np.concat([[metallicities[0]], Z_edges, [metallicities[-1]]])
        
    Z_oh = np.arange(5.3, 9.7, 0.022)
    Z_abs = Z_oh_to_Z_abs(Z_oh, Zsun=Zsun)
    
    base_path = os.path.join(os.path.split(__file__)[0])
    timestep_data = pd.read_csv(
            os.path.join(base_path, "chruslinska_nelemans_data", "Time_redshift_deltaT.dat"),
            sep=" ",
            skiprows=[0],
            names=["time", "redshift", "dt"]
            )
    time = timestep_data.time.values * u.Myr
    redz = timestep_data.redshift.values
    if redshift.max() > redz.max() + 1e-1:
        raise ValueError(f"Cannot use Chruslinska+19 SFH beyond z={redz.max():.2f}. Please use a lower redshift value.")
    dt = timestep_data.dt.values * u.Myr
    
    # default
    file = os.path.join(base_path, "chruslinska_data", "moderate_FOH_z_dM.dat")
    if option == "highZ":
        file = os.path.join(base_path, "chruslinska_data", "high-Z-extreme_FOH_z_dM.dat")
    elif option == "lowZ":
        file = os.path.join(base_path, "chruslinska_data", "low-Z_extreme_FOH_z_dM.dat")
    
    sf_data = pd.read_csv(
        file,
        sep = " ",
        skiprows = [0],
        names = Z_abs
        )
    print(sf_data.isnull().any(axis=None))
    print(sf_data.index[sf_data.isnull().any(axis=1)])

    print(sf_data)
    sf = sf_data.to_numpy(dtype=float, na_value=0e0)[:,:] * (u.Msun * u.Mpc**-3)
    print(sf)
    
    SFR_at_z = np.zeros(shape=n_i) * u.Msun * u.Mpc ** -3 * u.yr ** -1
    mean_Z = np.zeros(shape=n_i)
    fracSFR = np.zeros(shape=(n_j, n_i)) * u.Msun * u.Mpc ** -3 * u.yr ** -1
    
    for i in range(n_i):
        t_lo = t_edges[i]
        t_hi = t_edges[i+1]
        time_mask = (time >= t_lo) & (time <= t_hi)
        
        if (time_mask == False).all():
            closest_time = np.argmin(np.abs(time - comoving_time[i]))
            time_mask[closest_time] = True
        
        dt_span = np.sum(dt[time_mask]).to(u.yr)
        sfr = np.sum(sf[time_mask,:], axis=None) / dt_span
        sf_per_Z = np.sum(sf[time_mask,:], axis=0) / dt_span
        
        sf_cdf = np.cumsum(sf_per_Z) / sf_per_Z.sum()

        for j in range(n_j):
            
            cdf_pts = np.interp(Z_edges[j:j+2], Z_abs, sf_cdf)
            fraction = cdf_pts[1] - cdf_pts[0]
            fracSFR[j,i] = (fraction * sfr).to(u.Msun * u.Mpc ** -3 * u.yr ** -1)
            
        mean_Z[i] = np.interp(0.5, sf_cdf, Z_abs)
        SFR_at_z[i] = sfr
        
    sfh = {
        "redshift": redshift,
        "metallicities": metallicities,
        "SFR_at_z": SFR_at_z,
        "mean_metallicity": mean_Z,
        "fractional_SFR": fracSFR,
    }
    
    return sfh


def Z_oh_to_Z_abs(Z_oh: NDArray, Z_ohsun: float = 8.83, Zsun: float = 0.017) -> NDArray:
    return  (10 ** (Z_oh - Z_ohsun)) * Zsun
