import numpy as np
from astropy import units as u
import pandas as pd
from scipy.integrate import quad

from pandas import HDFStore, DataFrame
from numpy.typing import NDArray


def trivial_Pdet(data: dict) -> NDArray:
    '''Returns ones; for calculating intrinsic rates.'''
    n: int = data["bin_num"].shape[0]
    return np.ones(shape=n, dtype=float)


# filter pipeline for unwinnowed COSMIC output
def process_cosmic_models(store, HDFStore) -> tuple[NDArray, DataFrame, DataFrame]:
    '''
    Winnow a COSMIC output file down to BHNS systems.
    '''
    ecc_range, ecc_integral = calculate_ecc_integral(1000)

    bpp = store.get("bpp")
    bcm = store.get("bcm")
    
    cbc_bins, cbc_form_rows, bpp = get_cbc_systems(bpp, bcm)
    t_gw_sec = calculate_gw_timescale(cbc_form_rows, ecc_range, ecc_integral)
    t_gw = (t_gw_sec * u.s).to(u.Myr) #type: ignore
    
    t_delay = cbc_form_rows.tphys + t_gw.value
    data = {"t_gw": t_gw.value, "t_delay": t_delay}
    cbc_form_rows.assign(**data)
    
    return cbc_bins, cbc_form_rows.copy(), bpp


# filter down to BHNS systems
def get_cbc_systems(bpp: DataFrame, bcm: DataFrame) -> tuple[NDArray, DataFrame, DataFrame]:
     '''
     Obtain CBC system bins by merger type.
     '''
     kstars = [13,14]
     merger_types = ["1313", "1413", "1314", "1414"]

     cbc_form_rows = bpp.loc[(bpp['kstar_1'].isin(kstars)) & (bpp['kstar_2'].isin(kstars)) & (bpp['sep'] > 0)].groupby('bin_num', as_index=False).first()
     _bins = cbc_form_rows.bin_num.unique()
     cbc_disrupting_later = bpp.loc[bpp.bin_num.isin(_bins) & (bpp.sep == -1)].bin_num.unique()
     cbc_form_rows = cbc_form_rows.loc[~cbc_form_rows.bin_num.isin(cbc_disrupting_later)]
     bins = cbc_form_rows.bin_num.unique()

     cbc_bpp = bpp.loc[bpp.bin_num.isin(bins)]

     bc_merge_bins = bcm.loc[bcm.merger_type.isin(merger_types)].bin_num.unique()
     bc_wide_bins = bcm.loc[(bcm.bin_state==0) & (bcm.kstar_1.isin(kstars)) & (bcm.kstar_2.isin(kstars))].bin_num.unique()

     n1 = bc_merge_bins[~np.isin(bc_merge_bins, bins)]
     n2 = bc_wide_bins[~np.isin(bc_wide_bins, bins)]

     if len(n1) > 0:
          print(f"WARNING: merging bin numbers {n1} not found in formation filter.")
     if len(n2) > 0:
          print(f"WARNING: wide bin numbers {n2} not found in formation filter.")
     
     return bins, cbc_form_rows, cbc_bpp


# functions for postprocessing COSMIC data & calculate GW timescale
def calculate_ecc_integral(n_pts: int = 1000) -> tuple[NDArray, NDArray]:
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
                              ecc_range: NDArray, ecc_int: NDArray) -> pd.Series:
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
