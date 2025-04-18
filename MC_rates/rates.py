from __future__ import annotations

import os
import sys
import numpy as np
from multiprocessing import get_context
from astropy import units as u, constants as const
from scipy.stats import norm
from astropy.cosmology import Cosmology, Planck18, z_at_value
from dataclasses import dataclass
import pandas as pd

from numpy.typing import NDArray
from astropy.units import Quantity

from rates_functions import calc_mean_metallicity_madau_fragos,\
    calc_SFR_madau_fragos, trivial_Pdet


@dataclass
class CosmicBinaries:
    metallicity: float
    initC: pd.DataFrame
    merger_data: pd.DataFrame
    
    # pop statistics
    init_info: pd.DataFrame
    total_star_mass: Quantity["mass"]
    simulated_mass: Quantity["mass"]
    f_corr: float

def load_cosmic_bins(filepaths: str | list[str]) -> list[CosmicBinaries]:
    '''Load cosmic models from hdf5.'''
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    
    output = []
    for f in filepaths:
        _initC = pd.read_hdf(f, "initC")
        _Zval = _initC.metallicity.values[0]
        _merger_data = pd.read_hdf(f, "merger_data")
        _initInfo = pd.read_hdf(f, "init_info")
        Mpop = _initInfo.Msim.values[0] * u.Msun
        Msim = _initInfo.Msystems.values[0] * u.Msun
        f_corr = Msim/Mpop
        struct = CosmicBinaries(_Zval, _initC, _merger_data, _initInfo, Mpop, Msim, f_corr)
        output.append(struct)
    
    sort_fn = lambda x: x.metallicity
    output = sorted(output, key=sort_fn)
    return output


@dataclass
class MCParams:
    '''Object to contain all info for MC sampling.'''
    cosmology: Cosmology
    cmv_time: NDArray[u.Myr] # (num_pts,)
    redshift: NDArray
    cmv_distance: NDArray[u.Mpc]
    SFR_at_z: NDArray 
    
    # metallicity and binaries
    bins: list[CosmicBinaries] # (j,)
    mean_met_at_z: NDArray # (num_pts,)
    fractional_SFR_at_met: NDArray # shape (j, num_pts)

def init_sampler(t0: Quantity, tf: Quantity,
                 filepaths_to_bins: list[str], **kwargs) -> MCParams:
    '''
    Create an MCParams instance with all the necessary information to
    draw samples and calculate rates.
    ### Parameters:
    - t0, tf: Quantity - earliest and latest comoving time values
    - filepaths_to_bins: list[str] - the location at which to find models
    - **kwargs
    ### Returns:
    - MCParams
    '''
    cosmo: Cosmology = kwargs.get("cosmology", Planck18)
    num_pts: int = kwargs.get("num_pts", 1000)
    sfr_function: function = kwargs.get("SFR_function", calc_SFR_madau_fragos)
    avg_met_function: function = kwargs.get("avg_met_function", calc_mean_metallicity_madau_fragos)
    logZ_sigma_for_SFR: float = kwargs.get("logZ_sigma", 0.5)
    
    comoving_time = np.linspace(t0, tf, num_pts).to(u.Myr)
    redshift = z_at_value(cosmo.age, comoving_time)
    comoving_distance = cosmo.comoving_distance(redshift).to(u.Mpc)
    
    # load binaries from COSMIC
    bins_list = load_cosmic_bins(filepaths_to_bins)
    j = len(bins_list)
    metallicities = [x.metallicity for x in bins_list]
    
    # calculate SFR and the fraction of SFR per metallicity at each time value
    mean_met_at_z = avg_met_function(redshift)
    SFR_at_z = sfr_function(redshift).to(u.Msun * u.yr ** -1 * u.Mpc ** -3)
    fracSFR = np.zeros(shape=(j, num_pts), dtype=float)
    log_Z = np.log10(metallicities)
    log_avgZ =  np.log10(mean_met_at_z)
    
    for n in range(num_pts):
        fracSFR[:,n] = norm.pdf(x=log_Z, loc=log_avgZ[n], scale=logZ_sigma_for_SFR)
        
    # create our object and return
    params = MCParams(
        cosmology=cosmo,
        cmv_time=comoving_time,
        redshift=redshift,
        cmv_distance=comoving_distance,
        SFR_at_z=SFR_at_z,
        bins=bins_list,
        mean_met_at_z=mean_met_at_z,
        fractional_SFR_at_met=fracSFR
    )
    return params

def calc_MC_rates(sampler: MCParams,
                  n: int = 100, seed: int = 0, **kwargs) -> pd.DataFrame:
    '''
    Calculate rates by taking a monte-carlo sum with binaries from `sampler`.
    ### Parameters:
    - n: int - the number of monte-carlo samples to draw from each system
    - seed: int - the pseudorandom seed 
    #### Kwargs:
    - dt: Quantity - the time interval to determine comoving time bins
    - detectability_function: function - the function accounting for detector effects
    ### Returns:
    - DataFrame - a dataframe of MC samples, weights, and detectability
    '''
    bins: list[CosmicBinaries] = sampler.bins
    
    dt: Quantity = kwargs.get("dt", 100 * u.Myr)
    detectability_function: function = kwargs.get("detectability_function", trivial_Pdet)
    time_bins: NDArray = np.arange(sampler.cmv_time[0].value, sampler.cmv_time[-1].value, dt.value)
    
    dicts = []
    
    for j_Z, m in enumerate(bins):
        metallicity: float = m.metallicity
        mergers: pd.DataFrame = m.merger_data
        n_k: int = mergers.index.shape[0]
        
        fracSFR_pdf_at_met = sampler.fractional_SFR_at_met[j_Z,:]
        
        # get system statistics
        bin_num = mergers.bin_num.values
        mass_1 = mergers.mass_1.values
        mass_2 = mergers.mass_2.values
        kstar_1 = mergers.kstar_1.values
        kstar_2 = mergers.kstar_2.values
        t_delay = mergers.t_delay.values
        
        # sample formation times via ITF
        harvest = _inverse_transform_sample(
            (n_k, n), j_Z, fracSFR_pdf_at_met, sampler.cmv_time, seed)
        
        t_max = sampler.cmv_time.max().to(u.Myr).value
        sample_list = np.array([], dtype=float)
        num_valid_samples = np.zeros(n_k, dtype=np.int64)
        
        #print('zbin:', j_Z, 'shape:', harvest.shape)
        for i in range(n_k):
            valid_i = (harvest[i,:] + t_delay[i]) < t_max
            valid_time = harvest[i,:][valid_i]
            k = np.sum(valid_i).astype(int)
            num_valid_samples[i] = k
            sample_list = np.append(sample_list, valid_time, axis=0)
        
            
        columns = {
            "bin_num": np.repeat(bin_num, num_valid_samples),
            "metallicity": np.ones(sample_list.shape[0], dtype=float) * metallicity,
            "mass_1": np.repeat(mass_1, num_valid_samples),
            "mass_2": np.repeat(mass_2, num_valid_samples),
            "kstar_1": np.repeat(kstar_1, num_valid_samples),
            "kstar_2": np.repeat(kstar_2, num_valid_samples),
            "t_delay": np.repeat(t_delay, num_valid_samples),
            "t_form": sample_list,
            "z_form": np.interp(sample_list, sampler.cmv_time.value, sampler.redshift),
            "t_merge": sample_list + np.repeat(t_delay, num_valid_samples),
            "z_merge": np.interp(sample_list, sampler.cmv_time.value, sampler.redshift),
            }
        
        columns["P_det"] = detectability_function(columns)
        columns["weight"] = _calc_intrinsic_weights(sampler, j_Z, columns, n, dt, time_bins)
        
        dicts.append(columns)
        
    final_data = {a:None for a in dicts[0].keys()}
    for di in dicts:
        for k, v in di.items():
            if final_data[k] is None:
                final_data[k] = v
            else:
                final_data[k] = np.append(final_data[k], v)
    
    output = pd.DataFrame(final_data)
    return output

def _calc_intrinsic_weights(sampler: MCParams,
                  j_Z: int, data: dict[str:NDArray], num_draws: int, dt: Quantity, time_bins: NDArray) -> NDArray:
    '''
    We assign each event in our MC sample a 'weight' in the same fashion as described in Dominik et al. 2015,
    Bavera et al. 2020, to calculate rates.
    '''
    dt = dt.to(u.Myr)
    i_t: int = time_bins.shape[0] - 1
    output = np.zeros(data["t_form"].shape[0])
    
    for i in range(i_t):
        t0, t1 = time_bins[i], time_bins[i+1]
        idx = (data["t_merge"] >= t0) & (data["t_merge"] < t1)
        
        z_form = data["z_form"][idx]
        z_event = data["z_merge"][idx]
        
        # star formation history
        SFR_at_z: NDArray =  np.interp(z_form, sampler.redshift, sampler.SFR_at_z)
        met_frac_at_z: float = np.interp(z_form, sampler.redshift, sampler.fractional_SFR_at_met[j_Z,:])
        SFH_jz = SFR_at_z * met_frac_at_z
        
        # comoving distance
        D_z = np.interp(z_event, sampler.redshift, sampler.cmv_distance).to(u.Mpc)
        
        w = (4 * np.pi * const.c) * (1/num_draws) * sampler.bins[j_Z].f_corr *\
            (SFH_jz/sampler.bins[j_Z].total_star_mass) * np.float_power(D_z, 2) * dt
        
        output[idx] = w.to(u.yr ** -1) * data["P_det"][idx] # multiply by Pdet(a number)
    
    return output
    
    

def _inverse_transform_sample(shape: tuple[int,int],
                              j_Z: int, fracSFR_pdf_at_met: NDArray,
                              comoving_time: NDArray, seed: int = 0) -> NDArray:
    '''
    Use inverse transform sampling to sample continuously in time.
    Specifically, this sampler accepts a metallicity bin and calculates the CDF of stars of that
    metallicity forming across cosmic time; it then draws a sample of formation times for stars of that
    particular metallicity.
    **NOTE:** This does not account for the absolute SFR at a given time, only the metallicity distribution! 
    ### Parameters:
    - shape: tuple[int, int] - the shape of the sample to draw
    - j_Z: int - the index of the metallicity at which to sample
    - fracSFR_pdf_at_met: NDArray - the fractional SFR at metallicity j_Z over time
    - comoving_time: the comoving time values over which to interpolate
    - seed: int - the pseudorandom seed
    ### Returns:
    - NDArray - the pseudorandom sample
    '''
    #seed =+ j_Z
    Z_pdf = fracSFR_pdf_at_met
    
    norm_pdf = (Z_pdf / np.sum(Z_pdf)) # normalize
    cdf = np.cumsum(norm_pdf)

    gen = np.random.default_rng(seed=seed)
    choices = gen.random(size=shape)
    harvest = np.interp(choices, cdf, comoving_time.value)
    
    return harvest
