from __future__ import annotations

import os
from glob import glob
from multiprocessing import set_start_method
from multiprocessing.pool import Pool
from functools import partial
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
                self.mergers = process_cosmic_models(store.get("bpp"))
            
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
        self.sfh_method = kwargs.get("sfh_method")
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
        Zlims: tuple | None = kwargs.get("Zlims", None)
        optimistic_ce: bool = kwargs.get("optimistic_ce", True)
        nproc: int = kwargs.get("nproc", 1)
        
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
        
        fracSFR_at_bin_centers: NDArray = \
            np.zeros(shape=(n_j, n), dtype=float) * (u.Msun * u.yr ** -1 * u.Mpc ** -3)
        for j in range(n_j):
            fracSFR_at_bin_centers[j,:] = \
                np.interp(time_bin_centers, self.comoving_time, self.fractional_SFR_at_met[j,:])
                
        columns = [
            "t_center",
            "t_i",
            "t_f",
            "z_center",
            "z_i",
            "z_f",
            "N_bbh",
            "N_bhns",
            "N_bns",
            "N_total",
            "R_bbh",
            "R_bhns",
            "R_bhns",
            "R_total"]
        
        # tqdm the pool here
        set_start_method('spawn')
        with Pool(nproc) as pool:
            # fn_args = (partial)
            bins_idx = range(0, nbins)
            rate_args = partial(
                self._rates_worker,
                t_centers=time_bin_centers,
                z_centers=z_at_centers,
                t_edges=time_bin_edges,
                z_edges=z_at_edges,
                frac_SFR=fracSFR_at_bin_centers,
                optimistic_ce=optimistic_ce,
                Zlims=Zlims,
                m1lims=primary_mass_lims,
                m2lims=secondary_mass_lims
                )
            rates_data = list(tqdm(pool.imap(rate_args, bins_idx)))
        
        output_arr = np.concat(rates_data, axis=0, dtype=object)
        output_df: pd.DataFrame = pd.DataFrame(output_arr, columns=columns)
        # TODO units
        #output_df.loc[:,["t_center", "t_i", "t_f"]] *= u.Myr
        #output_df.loc[:,["N_total", "N_bbh", "N_bhns", "N_bns"]] *= u.yr ** -1 * u.Gpc ** -3
        #output_df.loc[:,[""]]
    
        local_universe: pd.Series[bool] = output_df.z_center < z_local
        R_tot = output_df.R_total.loc[local_universe].sum() / u.yr
        R_bbh = output_df.R_bbh.loc[local_universe].sum() / u.yr
        R_bhns = output_df.R_bhns.loc[local_universe].sum() / u.yr
        R_bns = output_df.R_bns.loc[local_universe].sum() / u.yr

        return RatesResult(R_tot, R_bbh, R_bhns, R_bns, output_df)
    

    def _rates_worker(self,
                     i:int,
                     t_centers: NDArray,
                     z_centers: NDArray,
                     t_edges: NDArray,
                     z_edges: NDArray,
                     frac_SFR: NDArray,
                     optimistic_ce: bool = True,
                     Zlims: tuple | None = None,
                     m1lims: tuple | None = None,
                     m2lims: tuple | None = None) -> NDArray:

        result = np.zeros(15, dtype=float)
        t_center = t_centers[i]
        t_i = t_edges[i]
        t_f = t_edges[i+1]
        z_center: float = z_centers[i].value
        z_i = z_edges[i]
        z_f = z_edges[i+1]
        dz = np.abs(z_f - z_i).value

        N_rest_total = 0.0 * self.VOLRATE
        N_rest_bbh = 0.0 * self.VOLRATE
        N_rest_bhns = 0.0 * self.VOLRATE
        N_rest_bns = 0.0 * self.VOLRATE
            
        for j, Zbin in enumerate(self.bins):
            
            if Zlims is not None:
                Z_min = Zlims[0]
                Z_max = Zlims[1]
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
                    systems = systems.loc[~systems.merge_in_ce]
                
            if (m1lims is not None or m2lims is not None):
                component_masses = systems[["mass_1", "mass_2"]].to_numpy()
                primary_mass = component_masses.max(axis=1)
                secondary_mass = component_masses.min(axis=1)
                    
                if (m1lims is not None and m2lims is not None):
                    mass_filter = (primary_mass >= m1lims[0]) & (primary_mass < m1lims[1]) &\
                        (secondary_mass >= m2lims[0]) & (secondary_mass < m2lims[1])
                elif (m1lims):
                    mass_filter = (primary_mass >= m1lims[0]) & (primary_mass < m1lims[1])
                elif (m2lims):
                    mass_filter = (secondary_mass >= m2lims[0]) & (secondary_mass < m2lims[1])
                else:
                    raise ValueError()
                systems = systems[mass_filter]
            
            t_form: NDArray = (t_center - systems.t_delay.values * u.Myr).to(u.Myr)
            # get system type for each dco
            bbh, bhns, bns = self.dco_kstar_filter(systems)
                
            # get time bin in which each merging system formed
            SFR_bins_for_systems: NDArray = self._get_bins_from_time(t_form, t_centers)
            frac_SFR_at_t_form: NDArray = frac_SFR[j,SFR_bins_for_systems]
            
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
            dV_z = 4 * np.pi * (const.c / self.cosmology.H(z_center)) *\
                np.float_power(self.cosmology.comoving_distance(z_center), 2)
            #rint(n_rest)
            #print(dV_z)
            #print(dz)
            #print(z_center)
            return (n_rest * dV_z * (dz/(1 + z_center))).to(u.yr ** -1)
        
        N_total = N_rest_total.value
        N_bbh = N_rest_bbh.value
        N_bhns = N_rest_bhns.value
        N_bns = N_rest_bns.value
        R_total = rate_integral(N_rest_total).value
        R_bbh = rate_integral(N_rest_bbh).value
        R_bhns = rate_integral(N_rest_bhns).value
        R_bns = rate_integral(N_rest_bns).value

        return np.asarray(
            [t_center, t_i, t_f, z_center, z_i, z_f, N_total,\
            N_bbh, N_bhns, N_bns, R_total, R_bbh, R_bhns, R_bns], dtype=object)
    

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
    