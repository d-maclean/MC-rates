from __future__ import annotations

import numpy as np
from astropy import units as u, constants as const
from scipy.stats import norm, truncnorm
from scipy.integrate import quad
from functools import partial
from astropy.cosmology import Cosmology, Planck18, z_at_value
from dataclasses import dataclass
import pandas as pd

from numpy.typing import NDArray
from astropy.units import Quantity

from rates_functions import calc_mean_metallicity_madau_fragos,\
    calc_SFR_madau_fragos, trivial_Pdet, process_cosmic_models


@dataclass
class Model:
    metallicity: float
    initCond: pd.DataFrame
    mergers: pd.DataFrame
    
    # pop statistics
    n_singles: int
    n_binaries: int
    binfrac_model: float = 0.7
    total_star_mass: Quantity["mass"] = 0.0 * u.Msun
    simulated_mass: Quantity["mass"] = 0.0 * u.Msun
    imf_f_corr: float = 1.0

    @classmethod
    def load_cosmic_models(cls, filepaths: str | list[str], is_prefiltered: bool = True) -> list[Model]:
        '''
        Load cosmic models and return them as a list
        ### Parameters:
        - filepaths: list - a list of h5 files
        - is_prefiltered: bool = True - whether the data has been pre-winnowed to a list of merging systems
        ### Returns:
        - list[Model]
        '''
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        
        models = []
        for f in filepaths:
            initCond: pd.DataFrame = pd.read_hdf(f, key="initCond")
            metallicity: float = initCond.metallicity.values[0]
            
            if is_prefiltered:
                mergers = pd.read_hdf(f, key="mergers")
            else:
                _bpp = pd.read_hdf(f, key="bpp")
                mergers = process_cosmic_models(_bpp)
            
            n_singles: int = pd.read_hdf(f, key="n_singles").values.sum()
            n_binaries: int = pd.read_hdf(f, key="n_binaries").values.sum()
            binfrac_model: float = n_binaries / n_singles
            
            mass_singles: float = pd.read_hdf(f, key="mass_singles").values.sum()
            mass_binaries: float = pd.read_hdf(f, key="mass_binaries").values.sum()            
            Msim: Quantity = (mass_singles + mass_binaries) * u.Msun
            
            if is_prefiltered:
                Mpop: Quantity = pd.read_hdf(f, key="total_kept_mass").values.sum() * u.Msun
            else:
                Mpop: Quantity = (initCond.mass_1.sum() + initCond.mass_2.sum()) * u.Msun
            
            f_corr = Mpop/Msim
            
            data = {
                "metallicity": metallicity,
                "initCond": initCond,
                "mergers": mergers,
                "n_singles": n_singles,
                "n_binaries": n_binaries,
                "binfrac_model": binfrac_model,
                "total_star_mass": Mpop,
                "simulated_mass": Msim,
                "imf_f_corr": f_corr
                }
            
            struct = cls(**data)
            models.append(struct)
        
        sort_fn = lambda x: x.metallicity
        models = sorted(models, key=sort_fn)
        return models

@dataclass
class MCParams:
    '''Object to contain all info for rates calculation.'''
    cosmology: Cosmology
    comoving_time: NDArray[u.Myr] # (num_pts,)
    redshift: NDArray
    SFR_at_z: NDArray 
    
    # metallicity and binaries
    metallicities: list[float]
    bins: list[Model] # (j,)
    mean_met_at_z: NDArray # (num_pts,)
    fractional_SFR_at_met: NDArray # shape (j, num_pts)
    fcorr_SFR_fracs: NDArray # shape (num_pts) -- for correcting unmodeled SFR

    @classmethod
    def init_sampler(cls, t0: Quantity, tf: Quantity,
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
        
        # load binaries from COSMIC
        bins_list = Model.load_cosmic_models(filepaths_to_bins, is_prefiltered=True)
        l_j = len(bins_list)
        metallicities = [x.metallicity for x in bins_list]
        
        # get Star Formation info
        SFR_at_z = sfr_function(redshift).to(u.Msun * u.yr ** -1 * u.Mpc ** -3)
        mean_metallicity_at_z = avg_met_function(redshift)
        
        # use a lognormal PDF to estimate the fraction of each metallicity bin forming at each time point
        fracSFR = np.zeros(shape=(l_j, num_pts), dtype=float) * (u.Msun * u.yr ** -1 * u.Mpc ** -3)
        for i in range(num_pts):
            log_avgZ = np.log10(mean_metallicity_at_z[i])
            Z_pdf = norm.pdf(x=np.log10(metallicities), loc=log_avgZ, scale=0.5)
            Z_pdf /= Z_pdf.sum()
            fracSFR[:,i] = SFR_at_z[i] * Z_pdf
            
        # calculate fcorr_SFR_fracs for missing metallicity
        # this suggested by Mike to avoid over-modeling mass outside
        # of our model range
        fcorr_SFR_fracs = np.zeros(shape=num_pts)
        
        #minZ, maxZ = metallicities[0], metallicities[-1]
        #test_logZ = np.linspace(-10, 0, 1000)
        #low_pt = np.argmin(np.abs(test_logZ - minZ))
        #hi_pt = np.argmin(np.abs(test_logZ - maxZ))
        #for i in range(num_pts):
        #    total_SFR_pdf = norm.pdf(test_logZ, loc=np.log10(mean_metallicity_at_z[i]), scale=logZ_sigma_for_SFR)
        #    total_SFR_pdf = total_SFR_pdf/np.sum(total_SFR_pdf) # normalize
        #    fcorr_SFR_fracs[i] = np.sum(total_SFR_pdf[low_pt:hi_pt])
        
        # create our object and return
        params = cls(
            cosmology=cosmo,
w            comoving_time=comoving_time,
            redshift=redshift,
            SFR_at_z=SFR_at_z,
            metallicities=metallicities,
            bins=bins_list,
            mean_met_at_z=mean_metallicity_at_z,
            fractional_SFR_at_met=fracSFR,
            fcorr_SFR_fracs=fcorr_SFR_fracs
        )
        return params
    
    def calc_merger_rates_dz(self, zbins: int | NDArray, local_z: float = 0.1) -> tuple:
        '''
        Calculate DCO merger rates for the sample following Dominik et al. 2013 Eq. 16. 
        ### Parameters:
        - zbins: int | NDArray
        - local_z: float = 0.1
        ### Returns:
        - tuple - local Rate, local bbh rate, local bhns rate, local bns rate, raw data
        '''
        if (isinstance(zbins, np.ndarray)):
            if (zbins < 0).any():
                raise ValueError('Redshift must be positive!')
            elif (np.diff(zbins) < 0).any():
                zbins = zbins[::-1]
                if (np.diff(zbins) < 0).any():
                    raise ValueError('Cannot coerce `zbins` to be monotonically increasing...')
            
        if isinstance(zbins, int):
            lgzmin = np.log10(self.redshift.min())
            lgzmax = np.log10(self.redshift.max())
            zbins: NDArray = np.logspace(lgzmin.value, lgzmax.value, zbins)
        
        cosmo: Cosmology = self.cosmology
        n: int = zbins.shape[0] - 1
        
        VOLRATE = u.yr ** -1 * u.Gpc ** -3 # unit
        
        data: dict[str:NDArray] = {
            "z_i": np.zeros(shape=n),
            "z_f": np.zeros(shape=n),
            "z_center": np.zeros(shape=n),
            "t_center": np.zeros(shape=n) * u.Myr,
            "E_z": np.zeros(shape=n),
            "R_bbh": np.zeros(shape=n) * VOLRATE,
            "R_bhns": np.zeros(shape=n) * VOLRATE,
            "R_bns": np.zeros(shape=n) * VOLRATE,
            "R_tot": np.zeros(shape=n) * VOLRATE
        }
        
        for i in range(n):
            z_i = zbins[i]
            z_f = zbins[i+1]
            dz = np.abs(z_f-z_i)
            z_center: float = np.mean(zbins[i:i+2])
            t_center: Quantity = cosmo.age(z_center).to(u.Myr)
        
            # dimensionless Hubble parameter
            E_z: float = np.sqrt(cosmo._Onu0*(1+z_center)**4 + \
                cosmo._Om0*(1+z_center)**3 + cosmo._Ok0*(1+z_center)**2 + cosmo._Ode0)

            R_bbh = 0 * VOLRATE
            R_bns = 0 * VOLRATE
            R_bhns = 0 * VOLRATE

            for j, Zbin in enumerate(self.bins):
                
                f_corr: float = Zbin.imf_f_corr
                Msim: Quantity = Zbin.simulated_mass

                systems: pd.DataFrame = Zbin.mergers[Zbin.mergers.t_delay < t_center.value + self.comoving_time.min().value]
                bbh_systems: NDArray = (systems.kstar_1 == 14) & (systems.kstar_2 == 14)
                bns_systems: NDArray = (systems.kstar_1 == 13) & (systems.kstar_2 == 13)
                bhns_systems: NDArray = ((systems.kstar_1 == 13) & (systems.kstar_2 == 14))\
                    | ((systems.kstar_1 == 14) & (systems.kstar_2 == 13))
                t_form: NDArray = t_center - (systems.t_delay.values * u.Myr)
                frac_SFR_per_sys: NDArray = np.interp(
                    t_form, self.comoving_time, self.fractional_SFR_at_met[j,:]).to(u.Msun * u.yr ** -1 * u.Mpc ** -3)
                
                Rate_per_sys: NDArray = \
                    (f_corr * (frac_SFR_per_sys/Msim) * dz/((1+z_center) * E_z)).to(VOLRATE)
                
                R_bbh += Rate_per_sys[bbh_systems].sum()
                R_bns += Rate_per_sys[bns_systems].sum()
                R_bhns += Rate_per_sys[bhns_systems].sum()

            data["z_i"][i] = z_i
            data["z_f"][i] = z_f
            data["z_center"][i] = z_center
            data["t_center"][i] = t_center
            data["E_z"][i] = E_z
            data["R_bbh"][i] = R_bbh
            data["R_bhns"][i] = R_bhns
            data["R_bns"][i] = R_bns
            data["R_tot"][i] = R_bbh + R_bhns + R_bns
            
        calc_R_local = lambda x: ((1/(cosmo._H0 * cosmo.lookback_time(local_z))) * x.sum()).to(VOLRATE)
        
        R_local_tot = calc_R_local(data["R_tot"][data["z_center"] <= local_z])
        R_local_bbh = calc_R_local(data["R_bbh"][data["z_center"] <= local_z])
        R_local_bhns = calc_R_local(data["R_bhns"][data["z_center"] <= local_z])
        R_local_bns = calc_R_local(data["R_bns"][data["z_center"] <= local_z])
        
        return R_local_tot, R_local_bbh, R_local_bhns, R_local_bns, data


    def calc_MC_rates(self: MCParams,
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
        bins: list[Model] = self.bins
        
        dt: Quantity = kwargs.get("dt", 100 * u.Myr)
        detectability_function: function = kwargs.get("detectability_function", trivial_Pdet)
        time_bins: NDArray = np.arange(self.comoving_time[0].value, self.comoving_time[-1].value, dt.value)
        
        dicts = []
        
        for j_Z, m in enumerate(bins):
            metallicity: float = m.metallicity
            mergers: pd.DataFrame = m.mergers
            n_k: int = mergers.index.shape[0]
            
            # trying with weighted SFR
            fracSFR_pdf_at_met = self.weighted_SFR[j_Z,:]
            
            # get system statistics
            bin_num = mergers.bin_num.values
            mass_1 = mergers.mass_1.values
            mass_2 = mergers.mass_2.values
            kstar_1 = mergers.kstar_1.values
            kstar_2 = mergers.kstar_2.values
            t_delay = mergers.t_delay.values
            
            # sample formation times with ITM
            harvest = self._inverse_transform_sample(
                (n_k, n), fracSFR_pdf_at_met, self.comoving_time, seed)
            
            t_max = self.comoving_time.max().to(u.Myr).value
            sample_list = np.array([], dtype=float)
            num_valid_samples = np.zeros(n_k, dtype=np.int64)
            
            num_valid_samples = np.ones(n_k, dtype=np.int64) * n
            sample_list = np.reshape(harvest, n_k*n)
                
            columns = {
                "bin_num": np.repeat(bin_num, num_valid_samples),
                "metallicity": np.ones(sample_list.shape[0], dtype=float) * metallicity,
                "mass_1": np.repeat(mass_1, num_valid_samples),
                "mass_2": np.repeat(mass_2, num_valid_samples),
                "kstar_1": np.repeat(kstar_1, num_valid_samples),
                "kstar_2": np.repeat(kstar_2, num_valid_samples),
                "t_delay": np.repeat(t_delay, num_valid_samples),
                "t_form": sample_list,
                "z_form": np.interp(sample_list, self.comoving_time.value, self.redshift),
                "t_merge": sample_list + np.repeat(t_delay, num_valid_samples),
                "z_merge": np.interp(sample_list+np.repeat(t_delay, num_valid_samples), self.comoving_time.value, self.redshift, right=np.nan),
                }
            
            columns["P_det"] = detectability_function(columns)
            columns["Rate"] = self._calc_intrinsic_rates(j_Z, columns, n, dt, time_bins)
            
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

    def _calc_intrinsic_rates(self: MCParams,
                    j_Z: int, data: dict[str:NDArray], num_draws: int, dt: Quantity, time_bins: NDArray) -> NDArray:
        '''
        We estimate intrinsic merger rates by binning our sampled events by merger time and multiplying by
        the star formation history SFH(z,Z) and population correction factors.
        For an explanation of the SFH see Bavera et al. 2020.
        '''
        dt = dt.to(u.Myr)
        i_t: int = time_bins.shape[0] - 1
        output = np.zeros(data["t_form"].shape[0]) * (u.yr ** -1 * u.Gpc ** -3)
        
        for i in range(i_t):
            t0, t1 = time_bins[i], time_bins[i+1]
            idx = (data["t_merge"] >= t0) & (data["t_merge"] < t1)
            
            z_form = data["z_form"][idx]
            
            # star formation history
            SFR_at_z: NDArray =  np.interp(z_form, self.redshift[::-1], self.SFR_at_z[::-1]) # TODO
            met_frac_at_z: float = np.interp(z_form, self.redshift[::-1], self.fractional_SFR_at_met[j_Z,:][::-1]) # TODO
            SFH_jz = (SFR_at_z * met_frac_at_z).to(u.Msun * u.yr ** -1 * u.Gpc ** -3)
            fcorr_SFR_fracs = np.interp(z_form, self.redshift[::-1], self.fcorr_SFR_fracs[::-1]) # TODO
            
            # intrinsic merger rate -- see Dominik+13 eq. 16             
            rate = self.bins[j_Z].imf_f_corr * (1/num_draws) * fcorr_SFR_fracs *\
                SFH_jz * (self.bins[j_Z].total_star_mass ** -1)# * \
                            
            rate = rate.to(u.yr ** -1 * u.Gpc ** -3)
            output[idx] = rate
        
        return output
    
    @staticmethod
    def _inverse_transform_sample(shape: tuple[int,int],
                                fracSFR_pdf_at_met: NDArray, comoving_time: NDArray, seed: int = 0) -> NDArray:
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
        Z_pdf = fracSFR_pdf_at_met
        norm_pdf = (Z_pdf / np.sum(Z_pdf)) # normalize
        cdf = np.cumsum(norm_pdf)
        #print(cdf[-1])

        gen = np.random.default_rng(seed=seed)
        choices = gen.random(size=shape)
        harvest = np.interp(choices, cdf, comoving_time.value)
        
        return harvest
