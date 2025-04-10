from __future__ import annotations

import os
import numpy as np
import pandas as pd
from glob import glob
from astropy import units as u, constants as const
from astropy.cosmology import Cosmology, Planck18, z_at_value
from scipy.stats import norm
from multiprocessing import get_context
from functools import partial

from numpy.typing import NDArray
from astropy.units import Quantity

from rates_functions import calc_SFR_madau_fragos, calc_mean_metallicity_madau_fragos


class BinariesBin:
    '''
    Container for a metallicity bin and its associated models.
    Each bin should have a table of initial conditions, statistical
    info, and a corresponding table of DCO mergers.
    '''
    def __init__(self, src: str):

        if not os.path.isfile(src):
            raise FileNotFoundError(f"No data file found at {src}")
        self.filename = src
        self.initC = pd.read_hdf(src, key="initC")
        self.init_info = pd.read_hdf(src, key="init_info")
        self.merger_data = pd.read_hdf(src, key="merger_data")
        
        # metallicity
        self.met = self.initC.metallicity.values[0]
        
        # metadata
        #self.Msim = (np.sum(self.initC.mass_1.values) +\
        #    np.sum(self.initC.mass_2.values)) * u.Msun
        self.Msim = self.init_info.Msim.values[0] * u.Msun
        #self.Mtot = (np.sum(self.init_info.mass_singles.values) +\
        #    np.sum(self.init_info.mass_binaries.values)) * u.Msun
        self.Msystems = self.init_info.Msystems.values[0] * u.Msun
        #self.fcorr = self.Msim/self.Mtot
        self.fcorr = self.Msystems/self.Msim
        
        return
    
    def calc_chirp_mass(self, binaries: pd.DataFrame) -> float | NDArray:
        '''Calculate chirp mass for binary component(s) in the model.'''
        m1 = binaries.mass_1.values
        m2 = binaries.mass_2.values

        mu = (m1 * m2) / (m1 + m2)
        m_chirp = np.float_power(mu, 3/5) * np.float_power(m1+m2, 2/5)
        return m_chirp
    

class MCSampler:
    '''
    Object class for loading data and calculating merger rates for
    a set of stellar models and cosmological parameters.
    '''
    default_seed = 0
    
    def __init__(self,
                tmin: Quantity, tmax: Quantity, bins: list[BinariesBin] | str, **kwargs):
        
        # parse kwargs
        tmin = tmin.to(u.Myr)
        tmax = tmax.to(u.Myr)
        self.cosmo: Cosmology = kwargs.get("cosmology", Planck18)
        self.num_pts: int = kwargs.get("num_pts", 1000)
        self.sfr_function: function = kwargs.get("sfr_function", calc_SFR_madau_fragos)
        self.Z_function: function = kwargs.get("Z_function", calc_mean_metallicity_madau_fragos)
        self.Z_dist_option = "lognorm"
        self.Z_dist_sigma = 0.5
        
        # calculate time, redshift, and comoving valuess
        self.t = np.linspace(tmin, tmax, self.num_pts)
        self.z = z_at_value(self.cosmo.age, self.t)
        self.comoving_distance = self.cosmo.comoving_distance(self.z)
        self.cmV = self.cosmo.comoving_volume(self.z)
        
        # if 'bins' is a str, load binaries from that folder and sort
        if isinstance(bins, str):
            files = glob(os.path.join(bins, "*.hdf5"))
            binary_bins = [BinariesBin(x) for x in files]
            self.bins = sorted(binary_bins, key=lambda x: x.met)
        # or, just take a list and sort it
        elif isinstance(bins, list):
            self.bins = sorted(bins, key=lambda x: x.met)
        else:
            raise ValueError("Please pass a valid option for bins.")
                
        # set metallicity/SFR values
        self.metallicities: list[float] = [x.met for x in self.bins]
        self.SFR: NDArray = None
        self.mean_met: NDArray = None
        self.fSFR_at_metallicity: NDArray = None
        self.calc_SFR_and_Zdist(self.sfr_function, self.Z_function)
        
        return

    def calc_SFR_and_Zdist(self,
                sfr_function: function = calc_SFR_madau_fragos,
                Z_function: function = calc_mean_metallicity_madau_fragos,
                sigma: float = 0.5
                ) -> NDArray:
        
        n = self.num_pts
        m = len(self.metallicities)
        
        self.SFR = sfr_function(self.z)
        self.mean_met = Z_function(self.z)

        met_pdf = np.zeros(shape=(m, n), dtype=float)
        
        # calculate a log-norm distribution, normalize, and tile it
        # we may expand this in future to use different metallicity
        for i in range(n):
            Zdist = norm.pdf(x=np.log10(self.metallicities),\
                loc=np.log10(self.mean_met[i]), scale=sigma)
            Zdist = Zdist / np.sum(Zdist)
            met_pdf[:,i] = Zdist
            
        self.fSFR_at_metallicity = met_pdf
        
        return
    
    def get_z_at_t(self, t: Quantity | NDArray) -> NDArray:
        return np.interp(t, self.t, self.z)

    def _inverse_transform_sample(self,
                                  shape: tuple[int,int], metallicity_bin: int, seed: int = None) -> NDArray:
        '''
        Use inverse transform sampling to sample continuously in time.
        Specifically, this sampler accepts a metallicity bin and calculates the CDF of stars of that
        metallicity forming across cosmic time; it then draws a sample of formation times for stars of that
        particular metallicity.
        **NOTE:** This does not account for the absolute SFR at a given time, only the metallicity distribution! 
        ### Parameters:
        - shape: tuple[int, int] - the shape of the sample to draw
        - metallicity_bin: the metallicity at which to sample
        - seed: int - the pseudorandom seed
        ### Returns:
        - NDArray - the pseudorandom sample
        '''
        if seed is None:
            seed = self.default_seed + metallicity_bin
        else:
            seed =+ metallicity_bin
        
        Z_pdf = self.fSFR_at_metallicity[metallicity_bin,:]
        
        norm_pdf = (Z_pdf / np.sum(Z_pdf)) # normalize
        cdf = np.cumsum(norm_pdf)

        gen = np.random.default_rng(seed=seed)
        choices = gen.random(size=shape)
        harvest = np.interp(choices, cdf, self.t.value)
        
        return harvest
    
    def _dummy_P_det(self, sample: pd.DataFrame) -> NDArray:
        return np.ones(sample.shape[0], dtype=float)
    
    def _sample_worker(self,
                       _bin: BinariesBin, num_draws: int, seed: int) -> pd.DataFrame:
        
        initC = _bin.initC
        mergers = _bin.merger_data
        n: int = mergers.shape[0]
        j_Z: int = np.argmin(np.abs(self.metallicities - _bin.met))
        
        # draw a random sample
        random_sample = self._inverse_transform_sample((n, num_draws), j_Z, seed)
        
        # filter out systems that don't merge in a hubble time
        n_valid_samples = np.zeros(shape=n, dtype=int)
        sample_list = np.array([], dtype=float)
        
        t_delay = mergers.t_delay.values
        
        for i in range(n):
            valid_i = random_sample[i,:] + t_delay[i] < self.t[-1].value
            valid_time = random_sample[i,:][valid_i]
            k = len(valid_time)
            n_valid_samples[i] = k
            sample_list = np.concat([sample_list, valid_time])
            
        # write output
        data = {
            "bin_num": mergers["bin_num"].values,
            #"mass_1": mergers["mass_1"].values,
            #"mass_2": mergers["mass_2"].values,
            #"kstar_1": mergers["kstar_1"].values,
            #"kstar_2": mergers["kstar_2"].values,
            "metallicity": initC["metallicity"].values,
            #"mass_1_init": initC["mass_1"].values,
            #"mass_2_init": initC["mass_2"].values,
            #"t_gw": mergers["t_gw"].values,
            "t_delay": mergers["t_delay"].values,
        }
        
        for k, v in data.items():
            data[k] = np.repeat(v, n_valid_samples)
        data["t_form"] = sample_list
        t_form_quant = sample_list * u.Myr
        data["z_form"] = self.get_z_at_t(t_form_quant)
        t_event_quant = t_form_quant + (data["t_delay"] * u.Myr)
        data["t_merge"] = t_event_quant.value
        data["z_merge"] = self.get_z_at_t(t_event_quant)
        
        output_df = pd.DataFrame(data)
        output_df.reset_index(inplace=True)
        
        return output_df

    def draw_mc_sample(self, num_draws: int = 100, nproc: int = 1, seed: int = None) -> list[pd.DataFrame]:
        '''
        Draw a sample of binaries from the sampler's bins.
        '''
        if nproc > len(self.bins):
            nproc = len(self.bins)
            print(f"Only using {nproc} CPUs... we only have that many bins.")

        if seed is None:
            seed = self.default_seed

        ctx = get_context("spawn")
        with ctx.Pool(nproc) as pool:
            
            args = partial(self._sample_worker, num_draws=num_draws, seed=seed)
            sample_dfs = pool.map(args, self.bins)
            
        return sample_dfs
        
    def _calc_event_weights(self,
                            sample: pd.DataFrame, time_bins: NDArray,
                            det_fn: function, num_draws: int) -> pd.DataFrame:
        '''
        Calculate event weights per Gpc3 per yr based on the Bavera+20 monte carlo sum method.
        ### Parameters:
        - sample: a pd.DataFrame of samples
        - j_Z: an integer representing the metallicity bin
        - tbins: a numpy.ndarray of shape (i, 2) where i  is the number of time bins
        '''
        j_Z: int = np.argmin(np.abs(self.metallicities - sample.metallicity.values[0]))
        _bin: BinariesBin = self.bins[j_Z]
        
        # fractional SFR
        binfrac: float = 0.7
        fcorr: float = _bin.fcorr
        Mtot: Quantity = _bin.Mtot
        
        n: int = sample.shape[0]
        m: int = len(time_bins) - 1
        
        output_df = pd.DataFrame(columns=np.arange(m), index=sample.index.copy())
        
        for i in range(m):
            if i == m:
                break
            
            results = pd.Series(np.zeros(n), index = sample.index, name=i)
            
            t0 = time_bins[i]
            tf = time_bins[i+1]
            dt = (tf - t0).to(u.Myr)
            t_center = (t0 +tf)/2
            z_center = self.get_z_at_t(t_center)
            SFR_at_center = np.interp(t_center, self.t, self.SFR)
            met_frac_at_center = np.interp(t_center, xp=self.t, fp=self.fSFR_at_metallicity[j_Z,:])
            fSFR_Z = met_frac_at_center * SFR_at_center.to(u.Msun * u.yr ** -1 * u.Mpc ** -3)
            
            # get systems that merge in the bin
            filter = (sample.t_merge >= t0) & (sample.t_merge < tf)
            systems = sample[filter]
            f_index = systems.index.values
            
            # comoving distance to each (individual) sample
            D_z = np.interp(systems.z_merge, self.z.value, self.comoving_distance)
            P_det = det_fn(systems)
            
            # we divide the weight by the number of random draws we did for the system;
            # this is to avoid duplicating the contribution (we only simulated it once)
            draw_corr = num_draws ** -1
                        
            w =  num_draws * (fSFR_Z/Mtot) * binfrac * fcorr *\
                (4 * np.pi * const.c) * np.float_power(D_z, 2) * P_det * dt
                
            results.loc[f_index] = w.to(u.yr ** -1)
            output_df[i] = results
        
        return output_df
    
    def calc_rates(self,
                   samples: list[pd.DataFrame], 
                   zmin: float, zmax: float, num_draws: int = 100, nproc = 1, **kwargs) -> pd.DataFrame:
    
        if zmin < self.z.min():
            zmin = self.z.min()
    
        m: int = len(samples)
        dt: Quantity = kwargs.get("dt", 100 * u.Myr)
        time_bin_edges: NDArray = np.arange(self.t[0].value, self.t[-1].value, dt.value) * u.Myr
        
        k: int = len(time_bin_edges) - 1
        t_cntr = [(time_bin_edges[i]+time_bin_edges[i+1])/2 for i in range(k)]
        z_at_cntr: NDArray = np.interp(t_cntr, self.t, self.z)
        
        if "det_function" not in kwargs.keys():
            print(f"No detectability function supplied; Using {1}.")
        det_fn: function = kwargs.get("det_fn", self._dummy_P_det)
        
        ctx = get_context("spawn")
        with ctx.Pool(nproc) as pool:
            args = partial(self._calc_event_weights,\
                time_bins=time_bin_edges, det_fn=det_fn, num_draws=num_draws)
            weights = pool.map(args, samples)
            
        z_filter = (z_at_cntr >= zmin) & (z_at_cntr < zmax)
        bins_in_zrange = np.arange(len(z_at_cntr))[z_filter]
        
        rates = 0e0
        for i in range(m):
            _w = weights[i]
            _w_in_range = _w.loc[:,bins_in_zrange]
            rates += _w_in_range.values.sum() 

        return rates * (u.yr ** -1)

    def run(self, **kwargs) -> pd.DataFrame:
        '''
        Run a rates calculation from start to finish, using the monte-carlo sum method
        explained in Bavera+20: https://doi.org/10.1051/0004-6361/201936204
        ### Arguments:
        - num_draws: int = the number of random samples to draw for each binary system.
            DEFAULT = 100
        - seed: int = the seed to use for pseudorandom number generation while drawing
            samples. DEFAULT = 4242564
        - zlims: tuple = the redshift range in which to calculate rates, formatted as
            (zmin, zmax). DEFAULT = (0, 0.1)
        - nproc: int = the number of CPU cores to use. DEFAULT = 1
        ### Returns:
        - rates: pd.DataFrame = a DataFrame of merger rates
        '''
        num_draws: int = kwargs.get("num_draws", 100)
        seed: int = kwargs.get("seed", 4242564)
        zlims: tuple[float,float] = kwargs.get("zlims", (0.0, 0.1))
        nproc: int = kwargs.get("nproc", 1)
        
        rand_sample: list[pd.DataFrame] = self.draw_mc_sample(num_draws, nproc=nproc, seed=seed)
        rates = self.calc_rates(rand_sample, zlims[0], zlims[1], nproc=nproc)
        
