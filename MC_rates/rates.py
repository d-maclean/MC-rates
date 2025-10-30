from __future__ import annotations

import os
from glob import glob
import numpy as np
from astropy import units as u, constants as const
from astropy.cosmology import Cosmology, Planck15, z_at_value
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray
from pandas import DataFrame, Series
from astropy.units import Quantity

from typing import ClassVar, NamedTuple, Any

from rates_functions import process_cosmic_models
from sfh_madau_fragos import madau_fragos_SFH
from sfh_chruslinska_nelemans import chr_nel_SFH
from sfh_illustris import illustris_TNG_SFH

SFH_METHODS = {
    "lognorm",
    "truncnorm",
    "illustris",
    "chruslinska19"
    }


class RatesResult(NamedTuple):
    total: Quantity
    bbh: Quantity
    nsbh: Quantity
    bns: Quantity
    data: DataFrame

class Model:

    cache: ClassVar[dict[str,dict[str,Any]]] = {} # cache

    def __init__(self, filepath: str, **kwargs):

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"{filepath} not found!")
                        

        self.f_corr = kwargs.get("f_corr", 1.0)
        if filepath in self.cache:
            _data = self.cache[filepath]
            for key, value in _data.items():
                setattr(self, key, value)
            return

        with pd.HDFStore(filepath, mode="r") as store:

            self.initCond = store.get("initCond")
            self.metallicity: float = self.initCond.metallicity.values[0]
            if "/mergers" in store.keys():
                self.mergers = store.get("mergers")
            else:
                self.mergers = process_cosmic_models(store.get("bpp")) #type: ignore
            
            self.n_singles: int = store.get("n_singles").values.max()
            self.n_binaries: int = store.get("n_binaries").values.max()
            self.n_stars: int = store.get("n_stars").values.max()
            self.binfrac_model: float = self.n_binaries / (2 * self.n_binaries + self.n_singles)
            self.mass_singles: float = store.get("mass_singles").values.max()
            self.mass_binaries: float = store.get("mass_binaries").values.max() 
            self.mass_stars: float = store.get("mass_stars").values.max()           
            self.Msim: Quantity = self.mass_stars * u.Msun
            self.Mpop: Quantity = (self.initCond.mass_1.sum() + self.initCond.mass_2.sum()) * u.Msun

        self.cache[filepath] = {
            "metallicity": self.metallicity,
            "initCond": self.initCond,
            "mergers": self.mergers,
            "n_singles": self.n_binaries,
            "n_binaries": self.n_binaries,
            "binfrac_model": self.binfrac_model,
            "mass_singles": self.mass_singles,
            "mass_binaries": self.mass_binaries,
            "mass_stars": self.mass_stars,
            "Msim": self.Msim,
            "Mpop": self.Mpop
        }

        return
    
    @staticmethod
    def get_hash(filepath: str, **kwargs) -> int:
        kw = kwargs.copy()
        kw["filepath"] = filepath
        return hash(frozenset(kw.items()))

    @classmethod
    def load_cosmic_models(cls, files: str | list[str], **kwargs) -> NDArray[Any[Model]]:

        if isinstance(files, str):
            files = glob(os.path.join(files, "*.h*5"))
            if len(files) == 0:
                raise FileNotFoundError(f"Can't find any model files at {files}!")

        models = [cls(f, **kwargs) for f in files]
        sort_fn = lambda x: x.metallicity
        models = np.asarray(sorted(models, key=sort_fn))
        return models


class MCRates:

    RATE: ClassVar[Quantity] =  u.yr ** -1
    VOLRATE: ClassVar[Quantity] = u.yr ** -1 * u.Gpc ** -3

    def __init__(self, t0: Quantity, tf: Quantity, models: str | list[str], **kwargs):

        self.cosmology: Cosmology = kwargs.get("cosmology", Planck15)
        self.Zsun: float = kwargs.get("Zsun", 0.017)
        num_pts: int = kwargs.get("num_pts", 1000)
        self.sfh_method = kwargs.get("sfh_method", "truncnorm")
        if self.sfh_method not in SFH_METHODS:
            raise ValueError(f"Please supply a valid metallicity dispersion method. Valid options are: {[x for x in SFH_METHODS]}")
        logZ_sigma_for_SFH: float = kwargs.get("logZ_sigma", 0.5)
        chruslinska_option: str = kwargs.get("chruslinska_option", "moderate")
        f_corr: float = kwargs.get("f_corr", 1.0)

        self.comoving_time = np.linspace(t0, tf, num_pts).to(u.Myr)
        self.redshift = z_at_value(self.cosmology.age, self.comoving_time)

        self.bins = Model.load_cosmic_models(models, f_corr=f_corr)  
        self.metallicities = np.asarray([x.metallicity for x in self.bins])

        if self.sfh_method == "lognorm":
            sfh = madau_fragos_SFH(self.redshift, self.metallicities, logZ_sigma_for_SFH, truncate_lognorm=False, Zsun=self.Zsun)
        elif self.sfh_method == "truncnorm":
            sfh = madau_fragos_SFH(self.redshift, self.metallicities, logZ_sigma_for_SFH, truncate_lognorm=True, Zsun=self.Zsun)
        elif self.sfh_method == "illustris":
            sfh = illustris_TNG_SFH(self.comoving_time, self.metallicities, filepath=None)
        elif self.sfh_method == "chruslinska19":
            sfh = chr_nel_SFH(self.comoving_time, self.redshift, self.metallicities, option=chruslinska_option, Zsun=self.Zsun)
        else:
            raise ValueError("Couldn't find your SFH method!")
        self.SFR_at_z = sfh["SFR_at_z"]
        self.mean_metallicity_at_z = sfh["mean_metallicity"]
        self.fractional_SFR_at_met = sfh["fractional_SFR"]

        return

    @classmethod
    def init_sampler(cls, *args, **kwargs):
        '''Alias for __init__'''
        cls.__init__(*args, **kwargs)
        return

    # trying once again
    def calc_merger_rates(self,
                          nbins: int = 200, z_local: float = 0.1,
                          **kwargs) -> RatesResult:
        '''Calculate dco merger rates per Dominik+2013.'''
        primary_mass_lims: tuple | None = kwargs.get("primary_mass_lims", None)
        secondary_mass_lims: tuple | None = kwargs.get("secondary_mass_lims", None)
        q_lims: tuple | None = kwargs.get("q_lims", None)
        Zlims: tuple | None = kwargs.get("Zlims", None)
        optimistic_ce: bool = kwargs.get("optimistic_ce", True)
        show_tqdm: bool = kwargs.get("tqdm", True)
        
        mass_filter_pri: bool = False
        m_pri_min, m_pri_max = 0, 300
        mass_filter_sec: bool = False
        m_sec_min, m_sec_max = 0, 300
        do_q_filter: bool = False
        q_min, q_max = 0, 1
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
        if q_lims is not None:
            do_q_filter = True
            q_min: float = q_lims[0]
            q_max: float = q_lims[1]
        if Zlims is not None:
            Z_filter = True
            Z_min: float = Zlims[0]
            Z_max: float = Zlims[1]
        
        n: int = nbins
        n_j: int = self.bins.shape[0]
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
        # SFR at center of each time bin
        for j in range(n_j):
            fracSFR_at_bin_centers[j,:] = \
                np.interp(time_bin_centers, self.comoving_time, self.fractional_SFR_at_met[j,:])
                
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
            "R_bbh": np.zeros(shape=n) * self.RATE,
            "R_bhns": np.zeros(shape=n) * self.RATE,
            "R_bns": np.zeros(shape=n) * self.RATE,
            "R_total": np.zeros(shape=n) * self.RATE,
        }
        
        for i, t_center in enumerate(
            tqdm(time_bin_centers, desc="Comoving time bins", unit="bins", disable=not show_tqdm)):
            
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
                Msim = Zbin.Msim
                f_corr = Zbin.f_corr
                
                t_delay_filter = \
                    (Zbin.mergers.t_delay.values < \
                    (t_center.to(u.Myr).value - self.comoving_time[0].to(u.Myr).value))
                systems: pd.DataFrame = Zbin.mergers.loc[t_delay_filter]

                if not optimistic_ce:
                    if "merge_in_ce" not in systems.columns:
                        print("Warning: Can't find `merge_in_ce` column in your data files. Proceeding with optimistic_ce.")
                    else:
                        systems = systems.loc[~systems.merge_in_ce] # type: ignore
                
                component_masses = systems[["mass_1", "mass_2"]].to_numpy()
                primary_mass: Series = component_masses.max(axis=1)
                secondary_mass: Series = component_masses.min(axis=1)
                if (mass_filter_pri or mass_filter_sec):
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
                if do_q_filter:
                    q = (secondary_mass / primary_mass)
                    q_filter = (q >= q_min) & (q <= q_max)
                    systems = systems[q_filter]
                
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

        return RatesResult(R_tot, R_bbh, R_bhns, R_bns, output_df)
    
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
    