from __future__ import annotations

import numpy as np
from astropy import units as u, constants as const
from astropy.cosmology import Cosmology, Planck15, z_at_value
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

from numpy.typing import NDArray
from astropy.units import Quantity
from typing import ClassVar, Iterable

from rates_functions import process_cosmic_models
from sfh_madau_fragos import madau_fragos_SFH
from sfh_chruslinska import chruslinska19_SFH
from sfh_illustris import illustris_TNG_SFH

SFH_METHODS = {
    "lognorm",
    "truncnorm",
    "illustris",
    "chruslinska19"
    }

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
    def load_cosmic_models(cls, filepaths: str | list[str], is_prefiltered: bool = True) -> Iterable[Model]:
        '''
        Load cosmic models and return them as a list
        ### Parameters:
        - filepaths: list - a list of h5 files
        ### Returns:
        - list[Model]
        '''
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        
        models = []
        for f in filepaths:
            initCond = pd.read_hdf(f, key="initCond")
            metallicity: float = initCond.metallicity.values[0]

            try:
                mergers = pd.read_hdf(f, key="mergers")
            except:
                bpp: pd.DataFrame = pd.read_hdf(f, key="bpp") # type: ignore
                mergers = process_cosmic_models(bpp)
            
            n_singles: int = pd.read_hdf(f, key="n_singles").values.sum()
            n_binaries: int = pd.read_hdf(f, key="n_binaries").values.sum()
            binfrac_model: float = n_binaries / (n_binaries + n_singles)
            
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
        models = np.asarray(sorted(models, key=sort_fn))
        return models

@dataclass
class MCRates:
    '''Object to contain all info for rates calculation.'''
    cosmology: Cosmology
    comoving_time: NDArray[u.Myr] # (num_pts,)
    redshift: NDArray
    SFR_at_z: NDArray
    
    # metallicity and binaries
    metallicities: NDArray
    bins: Iterable[Model] # (j,)
    mean_met_at_z: NDArray # (num_pts,)
    fractional_SFR_at_met: NDArray # shape (j, num_pts)
    RATE: ClassVar[Quantity] =  u.yr ** -1
    VOLRATE: ClassVar[Quantity] = u.yr ** -1 * u.Gpc ** -3

    @classmethod
    def init_sampler(cls, t0: Quantity, tf: Quantity,
                    filepaths_to_models: list[str], model_type: str = "cosmic", **kwargs) -> MCRates:
        '''
        Create an MCRates instance with all the necessary information to
        draw samples and calculate rates.
        ### Parameters:
        - t0, tf: Quantity - earliest and latest comoving time values
        - filepaths_to_models: list[str] - the locations of models
        - model_type: str = "cosmic" - the type of model to use
        - **kwargs
        ### Returns:
        - MCRates
        '''
        cosmo: Cosmology = kwargs.get("cosmology", Planck15)
        num_pts: int = kwargs.get("num_pts", 1000)
        SFH_method = kwargs.get("SFH_method", "truncnorm").lower()
        if SFH_method not in SFH_METHODS:
            raise ValueError(f"Please supply a valid metallicity dispersion method. Valid options are: {[x for x in SFH_METHODS]}")
        logZ_sigma_for_SFH: float = kwargs.get("logZ_sigma", 0.5)
        Zsun: float = kwargs.get("Zsun", 0.017)
        
        comoving_time = np.linspace(t0, tf, num_pts).to(u.Myr)
        redshift = z_at_value(cosmo.age, comoving_time)
        
        # load binaries from COSMIC
        models = Model.load_cosmic_models(filepaths_to_models)
        metallicities = np.asarray([x.metallicity for x in models])
        
        # get Star Formation info
        #FSR_at_z = sfr_function(redshift).to(u.Msun * u.yr ** -1 * u.Mpc ** -3)
        #mean_metallicity_at_z = avg_met_function(redshift)
        
        #modeled_SFR_fracs = np.ones(shape=num_pts)
        
        if SFH_method == "lognorm":
            sfh = madau_fragos_SFH(redshift, metallicities, logZ_sigma_for_SFH, truncate_lognorm=False, Zsun=Zsun)
        elif SFH_method == "truncnorm":
            sfh = madau_fragos_SFH(redshift, metallicities, logZ_sigma_for_SFH, truncate_lognorm=True, Zsun=Zsun)
        elif SFH_method == "illustris":
            sfh = illustris_TNG_SFH(comoving_time, metallicities, filepath=None)
        elif SFH_method == "chruslinska19":
            sfh = chruslinska19_SFH(redshift, metallicities)
        else:
            raise ValueError("Couldn't find your SFH method!")
        SFR_at_z = sfh["SFR_at_z"]
        mean_metallicity_at_z = sfh["mean_metallicity"]
        fractional_SFR = sfh["fractional_SFR"]
        
        # create our object and return
        params = cls(
            cosmology=cosmo,
            comoving_time=comoving_time,
            redshift=redshift,
            SFR_at_z=SFR_at_z,
            metallicities=metallicities,
            bins=models,
            mean_met_at_z=mean_metallicity_at_z,
            fractional_SFR_at_met=fractional_SFR,
        )
        return params
    
    # trying once again
    def calc_merger_rates(self,
                          nbins: int = 200, z_local: float = 0.1,
                          **kwargs) -> tuple[Quantity, Quantity, Quantity, Quantity, pd.DataFrame]:
        '''Calculate dco merger rates per Dominik+2013.'''
        primary_mass_lims: tuple | None = kwargs.get("primary_mass_lims", None)
        secondary_mass_lims: tuple | None = kwargs.get("secondary_mass_lims", None)
        Zlims: tuple | None = kwargs.get("Zlims", None)
        pessimistic_ce: bool = kwargs.get("pessimistic_ce", False)
        
        mass_filter_pri: bool = False
        m_pri_min, m_pri_max = 0, 300
        mass_filter_sec: bool = False
        m_sec_min, m_sec_max = 0, 300
        Z_filter: bool = False
        Z_min, Z_max = 0e0, 1e0
        
        if primary_mass_lims is not None:
            mass_filter_pri = True
            m_pri_min: float = primary_mass_lims[0]
            m_pri_max: float = primary_mass_lims[1]
        if secondary_mass_lims is not None:
            mass_filter_sec = True
            m_sec_min: float = secondary_mass_lims[0]
            m_sec_max: float = secondary_mass_lims[1]
        if Zlims is not None:
            Z_filter = True
            Z_min: float = Zlims[0]
            Z_max: float = Zlims[1]
        
        n: int = nbins
        n_j: int = self.bins.shape
        cosmo: Cosmology = self.cosmology

        t_i: Quantity = self.comoving_time[0]
        t_f: Quantity = self.comoving_time[-1]
        time_bin_edges: NDArray = np.linspace(t_i, t_f, n + 1).to(u.Myr)
        z_at_edges: NDArray = z_at_value(cosmo.age, time_bin_edges)
        time_bin_centers: NDArray = np.zeros(shape=n) * u.Myr
        for i in range(n):
            time_bin_centers[i] = np.mean(time_bin_edges[i:i+2])
        z_at_centers: NDArray = z_at_value(cosmo.age, time_bin_centers)
        E_z_at_centers: NDArray = cosmo.efunc(z_at_centers)
        
        fracSFR_at_bin_centers: NDArray = \
            np.zeros(shape=(n_j, n), dtype=float) * (u.Msun * u.yr ** -1 * u.Mpc ** -3)
        for j in range(n_j):
            fracSFR_at_bin_centers[j,:] = \
                np.interp(time_bin_centers, self.comoving_time, self.fractional_SFR_at_met[j,:])
        
        rate_unit = self.RATE
        
        data: dict[str,NDArray] = {
            "t_center": time_bin_centers.to(u.Myr),
            "t_i": time_bin_edges[:-1].to(u.Myr),
            "t_f": time_bin_edges[1:].to(u.Myr),
            "z_center": z_at_centers,
            "z_i": z_at_edges[:-1],
            "z_f": z_at_edges[1:],
            "E_z": E_z_at_centers,
            "N_bbh": np.zeros(shape=n) * self.VOLRATE,
            "N_bhns": np.zeros(shape=n) * self.VOLRATE,
            "N_bns": np.zeros(shape=n) * self.VOLRATE,
            "N_total": np.zeros(shape=n) * self.VOLRATE,
            "R_bbh": np.zeros(shape=n) * rate_unit, #self.VOLRATE,
            "R_bhns": np.zeros(shape=n) * rate_unit, #self.VOLRATE,
            "R_bns": np.zeros(shape=n) * rate_unit, #self.VOLRATE,
            "R_total": np.zeros(shape=n) * rate_unit, #self.VOLRATE
        }
        
        for i, t_center in enumerate(tqdm(time_bin_centers, desc="Comoving time bins", unit="bins")):
            
            t_i = time_bin_edges[i]
            t_f = time_bin_edges[i+1]
            z_center: float = z_at_centers[i]
            z_i = z_at_edges[i]
            z_f = z_at_edges[i+1]
            dz = np.abs(z_f - z_i)
            
            N_rest_total = 0.0 * self.VOLRATE
            N_rest_bbh = 0.0 * self.VOLRATE
            N_rest_bhns = 0.0 * self.VOLRATE
            N_rest_bns = 0.0 * self.VOLRATE
            
            for j, Zbin in enumerate(self.bins):
                
                if Z_filter:
                    if (Zbin.metallicity > Z_max) or (Zbin.metallicity < Z_min):
                        continue
                
                #met: float = Zbin.metallicity
                binfrac = Zbin.binfrac_model
                Msim = Zbin.simulated_mass
                f_corr = Zbin.imf_f_corr
                
                t_delay_filter = \
                    (Zbin.mergers.t_delay.values < \
                    (t_center.to(u.Myr).value - self.comoving_time[0].to(u.Myr).value))
                systems: pd.DataFrame = Zbin.mergers.loc[t_delay_filter]
                
                if (mass_filter_pri or mass_filter_sec):
                    component_masses = systems[["mass_1", "mass_2"]].to_numpy()
                    primary_mass = component_masses.max(axis=1)
                    secondary_mass = component_masses.min(axis=1)
                    
                    if (mass_filter_pri and mass_filter_sec):
                        mass_filter = (primary_mass >= m_pri_min) & (primary_mass < m_pri_max) &\
                            (secondary_mass >= m_sec_min) & (secondary_mass < m_sec_max)
                    elif (mass_filter_pri):
                        mass_filter = (primary_mass >= m_pri_min) & (primary_mass < m_pri_max)
                    elif (mass_filter_sec):
                        mass_filter = (secondary_mass >= m_sec_min) & (secondary_mass < m_sec_max)
                    else:
                        raise ValueError()
                    systems = systems[mass_filter]
                
                t_form: NDArray = (t_center - systems.t_delay.values * u.Myr).to(u.Myr)
                # get system type for each dco
                bbh, bhns, bns = self.dco_kstar_filter(systems)
                
                # get time bin in which each merging system formed
                SFR_bins_for_systems: NDArray = self._get_bins_from_time(t_form, time_bin_centers)
                frac_SFR_at_t_form: NDArray = fracSFR_at_bin_centers[j,SFR_bins_for_systems]
                
                # calculate the rest-frame merger rate -- see Dominik et al 2013 Eq. 16/17
                Rates_intrinsic: NDArray = (\
                    (binfrac * f_corr) * (frac_SFR_at_t_form / Msim)).to(self.VOLRATE)
                
                N_rest_total += Rates_intrinsic.sum()
                N_rest_bbh += Rates_intrinsic[bbh].sum()
                N_rest_bhns += Rates_intrinsic[bhns].sum()
                N_rest_bns += Rates_intrinsic[bns].sum()
                
            # prepare rates integral
            def rate_integral(n_rest: Quantity) -> Quantity:
                '''Integrate the rest-frame event rates to obtain a local, observable rate.'''
                dV_z = 4 * np.pi * (const.c / cosmo.H(z_center)) *\
                    np.float_power(cosmo.comoving_distance(z_center), 2)
                return (n_rest * dV_z * (dz/(1 + z_center))).to(u.yr ** -1)
            
            data["N_total"][i] = N_rest_total
            data["N_bbh"][i] = N_rest_bbh
            data["N_bhns"][i] = N_rest_bhns
            data["N_bns"][i] = N_rest_bns
            data["R_total"][i] = rate_integral(N_rest_total)
            data["R_bbh"][i] = rate_integral(N_rest_bbh)
            data["R_bhns"][i] = rate_integral(N_rest_bhns)
            data["R_bns"][i] = rate_integral(N_rest_bns)
        
        output_df: pd.DataFrame = pd.DataFrame(data)
    
        local_universe: pd.Series[bool] = output_df.z_center < z_local
        R_tot = output_df.R_total.loc[local_universe].sum() / u.yr #* z_local
        R_bbh = output_df.R_bbh.loc[local_universe].sum() / u.yr #* z_local
        R_bhns = output_df.R_bhns.loc[local_universe].sum() / u.yr #* z_local
        R_bns = output_df.R_bns.loc[local_universe].sum() / u.yr #* z_local

        return R_tot, R_bbh, R_bhns, R_bns, output_df
    
    @staticmethod
    def _get_bins_from_time(t: NDArray | Quantity, time_bins: NDArray) -> NDArray | Quantity:
        '''Return the index of the closest calculated time point from time values.'''
        if not isinstance(t, np.ndarray):
            t = np.asarray([t.to(u.Myr).value]) * u.Myr
        diffs = np.abs(t[:,np.newaxis] - time_bins)
        return np.argmin(diffs, axis=1)
        
        #indices: NDArray = np.zeros(shape=t.shape[0], dtype=int)
        #for k in range(t.shape[0]):
        #    indices[k] = np.argmin(np.abs(t[k] - time_bins))
        #
        #return indices
    
    @staticmethod
    def dco_kstar_filter(s: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:

        bbh = (s.kstar_1 == 14) & (s.kstar_2 == 14)
        bhns = ((s.kstar_1 == 13) & (s.kstar_2 == 14)) | ((s.kstar_1 == 14) & (s.kstar_2 == 13))
        bns = (s.kstar_1 == 13) & (s.kstar_2 == 13)
        
        return bbh, bhns, bns 
    