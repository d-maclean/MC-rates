from __future__ import annotations

import os
import numpy as np
import pandas as pd
from glob import glob
from astropy import units as u, constants as const
from astropy.cosmology import Cosmology, \
    Planck18, z_at_value, CosmologyError
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
    default_seed: int = 0
    abs_zmin: float = 1e-8
    zmax: float = 2e1
    
    def __init__(self,
                tmin: Quantity, tmax: Quantity, bins: list[BinariesBin] | str, **kwargs):
        
        self.num_pts: int = kwargs.get("num_pts", 1000)
        # set time vals
        self.cosmo: Cosmology = kwargs.get("cosmology", Planck18)
        tmin = tmin.to(u.Myr)
        tmax = tmax.to(u.Myr)
        
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
                       _bin: BinariesBin, num_draws: int, seed: int, **kwargs) -> pd.DataFrame:
        
        drop_merge_after_tH: bool = kwargs.get("drop_merge_after_tH", True)
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
            sample_list = np.concatenate([sample_list, valid_time])
            
        # write output
        data_dict = {
            "bin_num": mergers["bin_num"].values,
            "mass_1": mergers["mass_1"].values,
            "mass_2": mergers["mass_2"].values,
            "kstar_1": mergers["kstar_1"].values,
            "kstar_2": mergers["kstar_2"].values,
            "metallicity": initC["metallicity"].values,
            "mass_1_init": initC["mass_1"].values,
            "mass_2_init": initC["mass_2"].values,
            #"t_gw": mergers["t_gw"].values,
            "t_delay": mergers["t_delay"].values,
        }
        data: pd.DataFrame = pd.DataFrame(data_dict)
        data = data.loc[data.index.repeat(n_valid_samples)].reset_index(drop=True)
        
        sample_quant = sample_list * u.Myr
        t_merge_arr = sample_list + data["t_delay"].values
        t_merge_quant = t_merge_arr * u.Myr
        
        data = data.assign(t_form=sample_list)
        data = data.assign(z_form=self.get_z_at_t(sample_quant).value)
        data = data.assign(t_merge=t_merge_arr)
        data = data.assign(z_merge=self.get_z_at_t(t_merge_quant).value)
        
        return data

    def draw_mc_sample(self, num_draws: int = 100, nproc: int = 1, seed: int = None) -> list[pd.DataFrame]:
        '''
        Draw a sample of binaries from the sampler's bins.
        '''
        if nproc > len(self.bins):
            nproc = len(self.bins)
            print(f"Only using {nproc} CPUs... we only have that many bins.")

        if seed is None:
            seed = self.default_seed

        ctx = get_context("forkserver")
        with ctx.Pool(nproc) as pool:
            
            args = partial(self._sample_worker, num_draws=num_draws, seed=seed)
            sample_dfs = pool.map(args, self.bins)
            
        return sample_dfs
        
    def _calc_event_weights(self,
                            sample: pd.DataFrame, time_bins: NDArray,
                            zbins: NDArray, det_fn: function, num_draws: int) -> NDArray:
        '''
        Calculate event weights per Gpc3 per yr based on the Bavera+20 monte carlo sum method.
        ### Parameters:
        - sample: a pd.DataFrame of samples of size `n`
        - j_Z: an integer representing the metallicity bin
        - tbins: a numpy.ndarray of shape `(m,)` defining the edges of the time bins to use
        ### Returns:
        - NDArray: a 2D array of shape `(n, m)` containing the event weights
        '''
        j_Z: int = np.argmin(np.abs(self.metallicities - sample.metallicity.values[0]))
        _bin: BinariesBin = self.bins[j_Z]

        fcorr: float = _bin.fcorr
        Msim: Quantity = _bin.Msim
        
        n: int = sample.shape[0]
        m: int = len(zbins) - 1
        
        output: NDArray = np.zeros(shape=(n, m), dtype=float)
        
        for i in range(m):
            #results = pd.Series(np.zeros(n), index = sample.index, name=i)
            w_m: NDArray = np.zeros(shape=n) * (u.yr ** -1)
            
            t0 = time_bins[i]
            tf = time_bins[i+1]
            dt = (tf - t0) * u.Myr
            t_center = (t0 + tf)/2 * u.Myr
            
            # get systems that merge in the bin
            filter = (sample.t_merge >= t0) & (sample.t_merge < tf)
            systems = sample[filter]
            
            # get SFH for each system
            SFR_per_system = np.interp(systems.t_form, self.t.value, self.SFR)
            SFR_z_per_system = np.interp(systems.t_form, xp=self.t.value, fp=self.fSFR_at_metallicity[j_Z,:])
            SFH_per_system = SFR_z_per_system * SFR_per_system.to(u.Msun * u.yr ** -1 * u.Mpc ** -3)
            
            # comoving distance to each (individual) sample
            D_z = np.interp(systems.z_merge, self.z.value, self.comoving_distance)
            P_det = det_fn(systems)
            
            # we divide the weight by the number of random draws we did for the system;
            # this is to avoid duplicating the contribution (we only simulated it once)
            draw_corr = np.float_power(num_draws, -1)
            
            w_m[filter] = (draw_corr * fcorr * (SFH_per_system/Msim) *\
                (4 * np.pi * const.c) * np.float_power(D_z, 2) * P_det * dt).to(u.yr ** -1)
                
            output[:,i] = w_m.value
        
        return output
    
    def _bin_weights(self,
                    operands: tuple[pd.DataFrame, NDArray], zbins: NDArray, mass_bins: NDArray) -> NDArray:
        
        sample, w_j = operands
        i_redshift: int = len(zbins) - 1
        k_mass: int = len(mass_bins) - 1
        
        output: NDArray = np.zeros(shape=(i_redshift, k_mass), dtype=float)

        for i in range(i_redshift):
            z_lo = zbins[i+1]
            z_hi = zbins[i]
            z_filter = (sample.z_merge < z_hi) & (sample.z_merge >= z_lo)
            
            for k in range(k_mass):
                m_lo = mass_bins[k]
                m_hi = mass_bins[k+1]
                mass_filter = (sample.mass_1 < m_hi) & (sample.mass_1 >= m_lo)

                s = sample[z_filter & mass_filter]
                n = s.index
                w_jik = w_j[n,i]
                output[i,k] = np.sum(w_jik)                

        return output
    
    def calc_rates(self,
                   samples: list[pd.DataFrame], 
                   zrange: list[float, float], mass_bins: NDArray,
                   dt: Quantity, num_draws: int = 100, nproc = 1, **kwargs) -> tuple[Quantity, NDArray]:
        '''
        Calculate and sum the event weights for each system in `samples`.
        ### Parameters:
        - samples: an iterable of samples in DataFrame format
        - zrange: the range of redshifts in which to compute rates; tuple of form (zmin, zmax)
        - mass_bins: the preferred binning of primary mass values; if this is not passed, will bin in steps of 10 Msun.
        - num-draws: the number of MC draws performed in the sampling step.
        - nproc: the number of CPUs to use.
        ### Returns:
        - Quantity: the total rate between `zmin` and `zmax`
        - pd.DataFrame: binned rates as a function of redshift, metallicity. and primary mass
        '''
        verbose: bool = kwargs.get("verbose", False)
        
        if zrange[0] < self.z.min():
            print(f"Truncating redshift to sampler's zmin ({self.z.min()})...")
            zrange[0] = self.z.min()
        if "det_function" not in kwargs.keys():
            print(f"No detectability function supplied; Assuming {1}...")
        det_fn: function = kwargs.get("det_fn", self._dummy_P_det)
        if mass_bins is None:
            mass_bins = np.arange(0, 200, 10) # adding Msun later
        dt: float = kwargs.get("dt", 100)
            
        # prepare comoving time bins and corresponding redshift bins
        if isinstance(dt, Quantity):
            dt = dt.to(u.Myr).value
            
        t0, t1 =  np.interp(zrange, self.z, self.t.value)
        z0, z1 = zrange
        time_bins: NDArray = np.arange(t0, t1, dt)
        redshift_bins: NDArray = self.get_z_at_t((time_bins * u.Myr)).value
                
        # multiprocess
        ctx = get_context("forkserver")
        with ctx.Pool(nproc) as pool:
            args = partial(self._calc_event_weights,\
                time_bins=time_bins, zbins=redshift_bins, det_fn=det_fn, num_draws=num_draws)
            weights = pool.map(args, samples)
        
        # filter redshift bins into rates calc
        valid_i = np.asarray(redshift_bins < z1).nonzero()
        zbins_in_range = redshift_bins[valid_i]
        print(zbins_in_range)
        
        # calculate binned weights for analysis        
        bin_operands = [(s,w) for s,w in zip(samples, weights)]
        #with ctx.Pool(nproc) as pool:
        #    args = partial(self._bin_weights, zbins=zbins_in_range, mass_bins=mass_bins)
        #    binned_weights = pool.map(args, bin_operands)
        
        #binned_weights = np.zeros(shape=(len(self.bins), len(valid_i) - 1, len(mass_bins) - 1))
        #for j in range(len(self.bins)):
        #    binned_weights[j,:,:] = self._bin_weights(bin_operands[j], zbins=zbins_in_range, mass_bins=mass_bins)
        
        #binned_weights = np.stack(binned_weights)
        total_rate = np.sum(np.sum(weights))

        return total_rate, weights

    def run(self, **kwargs) -> tuple[Quantity, pd.DataFrame]:
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
        total_rate, binned_rates = self.calc_rates(rand_sample, zlims[0], zlims[1], nproc=nproc)
        
        return total_rate, binned_rates
 