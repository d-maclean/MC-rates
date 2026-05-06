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
def process_cosmic_models(store: HDFStore) -> tuple[DataFrame, DataFrame, DataFrame]:
     '''
     Winnow a COSMIC output file down to compact binary systems that don't disrupt.
     '''
     ecc_range, ecc_integral = calculate_ecc_integral(1000)

     bpp = store.get("bpp")
     bcm = store.get("bcm")
     initC = store.get("initCond")

     # detect mergers in common envelope phase
     ce_merger_idx = get_mergers_in_pessimistic_ce(bpp) # type: ignore
     print(f"Found {len(ce_merger_idx)} systems merging in CE...")
     bpp = bpp.assign(merge_in_ce = False)
     bpp.loc[bpp.bin_num.isin(ce_merger_idx), "merge_in_ce"] = True

     # filter binaries down to non-disrupted CBCs
     print("Filtering for CBCs...")
     cbc_idx, cbc_form_rows, cbc_bpp,  = get_cbc_systems(bpp, bcm) # type: ignore
     del bpp, bcm
     print(f"Found {len(cbc_idx)} BHNS systems.")
     
     # trying our best
     cbc_initC: DataFrame = initC.loc[cbc_idx].copy() # type: ignore
     del initC

     # get t_gw from Peters Formula
     print("Calculating gw timescales...")
     t_gw = calculate_gw_timescale(cbc_form_rows, ecc_range, ecc_integral)
     t_gw_Myr = t_gw * (3.1688e-14) # s to Myr

     # add t_gw to system age to get total delay time
     system_delay_time = cbc_form_rows.tphys + t_gw_Myr

     # collect data for saving
     cbc_form_rows = cbc_form_rows.assign(
          t_gw=t_gw_Myr.values,
          t_delay=system_delay_time.values,
     )

     return cbc_initC, cbc_form_rows, cbc_bpp


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

     # get mergers in pessimistic CE (where MS/HG donors always merge)
     ce_merger_idx = get_mergers_in_pessimistic_ce(bpp.loc[bpp.bin_num.isin(bins)])
     #print(f"Found {len(ce_merger_idx)} systems merging in CE...")
     cbc_form_rows = cbc_form_rows.assign(merge_in_ce = False)
     cbc_form_rows.loc[cbc_form_rows.bin_num.isin(ce_merger_idx), "merge_in_ce"] = True

     bc_merge_bins = bcm.loc[bcm.merger_type.isin(merger_types)].bin_num.unique()
     bc_wide_bins = bcm.loc[(bcm.bin_state==0) & (bcm.kstar_1.isin(kstars)) & (bcm.kstar_2.isin(kstars))].bin_num.unique()

     n1 = bc_merge_bins[~np.isin(bc_merge_bins, bins)]
     n2 = bc_wide_bins[~np.isin(bc_wide_bins, bins)]

     if len(n1) > 0:
          print(f"WARNING: merging bin numbers {n1} not found in formation filter.")
     if len(n2) > 0:
          print(f"WARNING: wide bin numbers {n2} not found in formation filter.")
     
     return bins.copy(), cbc_form_rows.copy(), cbc_bpp.copy()


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

def get_mergers_in_pessimistic_ce(bpp: pd.DataFrame) -> NDArray:
     '''
     Providees an arrray of bin numbers (unique systems) which
     would merge in CE assuming a pessimistic common envelope
     scenario. This treats systems which would have merged with
     `cemergeflag = 1` in COSMIC.
     
     Code adapted from Michael Zevin's rates code.
     '''
     kstars: list = [0,1,2,7,8,10,11,12] # from BSE/comenv.f

     CE1 = bpp.loc[(bpp.RRLO_1>bpp.RRLO_2) & (bpp.evol_type == 7)]
     CE2 = bpp.loc[(bpp.RRLO_2>bpp.RRLO_1) & (bpp.evol_type == 7)]

     CE1_merg = CE1.loc[CE1.kstar_1.isin(kstars)].bin_num
     CE2_merg = CE2.loc[CE2.kstar_2.isin(kstars)].bin_num
     
     ce_merge_bins = pd.concat([CE1_merg, CE2_merg]).unique()
     return ce_merge_bins
