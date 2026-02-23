from __future__ import annotations

import os
from glob import glob
import numpy as np
from astropy import units as u, constants as const
from astropy.cosmology import Cosmology, Planck15, z_at_value
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray
from pandas import DataFrame, Series
from astropy.units import Quantity
import xarray as xr

from typing import ClassVar, NamedTuple, Any
from xarray import Dataset

from rates_functions import process_cosmic_models
from sfh_madau_fragos import madau_fragos_SFH
from sfh_chruslinska_nelemans import chr_nel_SFH
from sfh_illustris import illustris_TNG_SFH


class RatesResult(NamedTuple):
    '''
    Nmaed tuple containing rates output. Local rates are reported in yr^-1 Gpc^-3.

    ### Fields

    `total`: the total local rate of CBCs

    `bbh`: the local binary black hole rate

    `nsbh`: the local neutron star/black hole rate

    `bns`: the local binary neutron star rate

    `data`: a Dataset of per-binary information for each metallicity and time bin
    '''
    total: Quantity
    bbh: Quantity
    nsbh: Quantity
    bns: Quantity
    data: Dataset

SFH_METHODS = {
    "lognorm",
    "truncnorm",
    "illustris",
    "chruslinska19"
    }

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
                self.bpp = None
            else:
                bins, self.mergers, self.bpp = process_cosmic_models(store) #type: ignore
                self.initCond = self.initCond.loc[bins]
            
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
            "bpp": self.bpp,
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
        self.Zsun: float = kwargs.get("Zsun", 0.014)
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

    def calc_merger_rates(self,
                          nbins: int = 200, z_local: float = 0.1,
                          **kwargs) -> RatesResult:
        '''
        Calculate DCO merger rates per Dominik+2013.
        
        ### Parameters
        `nbins`: the number of equally-spaced covmoving time bins to use. default=`200`

        `z_local`: the cutoff for the 'local' universe, used to calculate cosmological rates. default=`0.1`

        `primary_mass_lims`: cutoff limits for primary mass (Msun)

        `secondary_mass_lims`: cutoff limits for sedcondary mass (Msun)

        `q_lims`: cutoff limits for q (m2/m1)

        `Zlims`: cutoff limits for metallicity (absolute)

        `max_ns`: maximum mass for neutron stars. If not passed, BSE kstars are used instead.

        `optimistic_ce`: whether to allow stars with non-differentiated core/enevelope boundary to survive
        common envelope. default=`True`

        `verbose`: whether to output individual rates for each monte-carlo data point

        `show_tqdm`: show a progress bar. default=`True`.

        ### Returns

        `RatesResult`: a named tuple containing the following fields:
            - `total`: sum of all CBCs (yr^-1 Gpc^-3)
            - `bbh`: BBHs (yr^-1 Gpc^-3)
            - `nsbh`: NSBHs (yr^-1 Gpc^-3)
            - `bns`: BNSe (yr^-1 Gpc^-3)
            - `data`: Dataset of detailed rates data, including rates for each binary
        
        '''
        primary_mass_lims: tuple | None = kwargs.get("primary_mass_lims", None)
        secondary_mass_lims: tuple | None = kwargs.get("secondary_mass_lims", None)
        q_lims: tuple | None = kwargs.get("q_lims", None)
        Zlims: tuple | None = kwargs.get("Zlims", None)
        max_ns: float | None = kwargs.get("max_ns", None)
        optimistic_ce: bool = kwargs.get("optimistic_ce", True)
        show_tqdm: bool = kwargs.get("tqdm", True)

        n: int = nbins
        n_j: int = self.bins.shape[0]
        cosmo: Cosmology = self.cosmology
        
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

        n_binaries_per_j = np.array([x.mergers.shape[0] for x in self.bins])
        total_n_binaries = np.sum(n_binaries_per_j)

        try: # get channel data, if available
            channel_data = np.concat([x.mergers.channel for x in self.bins])
        except AttributeError:
            channel_data = np.ones(total_n_binaries, dtype=int) * -1

        get_mass_ratio_reversal = lambda mergers: (mergers.mass_1<mergers.mass_2)
        mass_ratio_reversal_data = np.concat([get_mass_ratio_reversal(x.mergers) for x in self.bins])
        _m1_init = np.concat([x.initCond.mass_1 for x in self.bins])
        _m2_init = np.concat([x.initCond.mass_2 for x in self.bins])

        data = dict(
            # info on comoving time bins
            time_bins = np.arange(0, n),
            t_center = time_bin_centers,
            z_center = z_at_centers,
            E_z = E_z_at_centers,
            is_local = np.zeros(n, dtype=bool),
            # info on binaries
            indices = np.arange(total_n_binaries),
            metallicity = np.repeat(self.metallicities, [x.mergers.shape[0] for x in self.bins]),
            bin_nums = np.concat([x.mergers.bin_num.unique() for x in self.bins]),
            m1 = np.concat([self._calculate_m1_m2_q(x.mergers)[0] for x in self.bins]),
            m2 = np.concat([self._calculate_m1_m2_q(x.mergers)[1] for x in self.bins]),
            q = np.concat([self._calculate_m1_m2_q(x.mergers)[2] for x in self.bins]),
            t_delay = np.concat([x.mergers.t_delay for x in self.bins]),
            channel = channel_data,
            # initial conditions
            m1_i = np.where(mass_ratio_reversal_data, _m2_init, _m1_init),
            m2_i = np.where(mass_ratio_reversal_data, _m1_init, _m2_init),
            porb_i = np.concat([x.initCond.porb for x in self.bins]),
            ecc_i = np.concat([x.initCond.ecc for x in self.bins]),
            # rates per binary per time bin
            rest_frame_rates = [np.zeros((nk, n)) for nk in n_binaries_per_j],
            rates = [np.zeros((nk, n)) for nk in n_binaries_per_j],
            # total rates per time bin
            N_bbh = np.zeros(shape=n) * self.VOLRATE,
            N_bhns = np.zeros(shape=n) * self.VOLRATE,
            N_bns = np.zeros(shape=n) * self.VOLRATE,
            N_total = np.zeros(shape=n) * self.VOLRATE,
            R_bbh = np.zeros(shape=n) * self.RATE,
            R_bhns = np.zeros(shape=n) * self.RATE,
            R_bns = np.zeros(shape=n) * self.RATE,
            R_total= np.zeros(shape=n) * self.RATE,
        )

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
                systems = Zbin.mergers
                nk = systems.shape[0]

                primary_mass, secondary_mass, q = self._calculate_m1_m2_q(systems)

                t_delay_filter = np.ones(nk, dtype=bool)
                mass_filter = t_delay_filter.copy()
                ce_filter = t_delay_filter.copy()
                q_filter = t_delay_filter.copy()
                
                t_delay_filter = \
                    (Zbin.mergers.t_delay.values < \
                    (t_center.to(u.Myr).value - self.comoving_time[0].to(u.Myr).value))

                if not optimistic_ce:
                    if "merge_in_ce" not in systems.columns:
                        print("Warning: Can't find `merge_in_ce` column in your data files. Proceeding with optimistic_ce.")
                    else:
                        ce_filter = ~systems.merge_in_ce
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
                if do_q_filter:
                    q = (secondary_mass / primary_mass)
                    q_filter = (q >= q_min) & (q <= q_max)
                
                merging_filter = (t_delay_filter) & (mass_filter) & (q_filter) & (ce_filter)
                merging_systems = systems[merging_filter]
                t_form: NDArray = (t_center - merging_systems.t_delay.values * u.Myr).to(u.Myr)

                # get system type for each dco
                if max_ns is None:
                    bbh, bhns, bns = self.dco_kstar_filter(systems)
                else:
                    bbh, bhns, bns = self.dco_mass_filter(systems, max_ns=max_ns)
                
                # get time bin in which each merging system formed
                SFR_bins_for_systems: NDArray = self._get_bins_from_time(t_form, time_bin_centers)
                frac_SFR_at_t_form: NDArray = fracSFR_at_bin_centers[j,SFR_bins_for_systems]
                
                # calculate the rest-frame merger rate -- see Dominik et al 2013 Eq. 16/17
                Rates_intrinsic: NDArray = (\
                    (binfrac * f_corr) * (frac_SFR_at_t_form / Msim)).to(self.VOLRATE)
                
                N_rest_total += Rates_intrinsic.sum()
                N_rest_bbh += Rates_intrinsic[bbh[merging_filter]].sum()
                N_rest_bhns += Rates_intrinsic[bhns[merging_filter]].sum()
                N_rest_bns += Rates_intrinsic[bns[merging_filter]].sum()
                
                rate_per_system = np.zeros(nk) * self.VOLRATE
                rate_per_system[merging_filter] = Rates_intrinsic
                data["rest_frame_rates"][j][:,i] = rate_per_system

            # prepare rates integral
            def cosmological_rate_integral() -> Quantity:
                '''Integrate the rest-frame event rates to obtain a local, observable rate.'''
                dV_z = 4 * np.pi * (const.c / cosmo.H(z_center)) *\
                    np.float_power(cosmo.comoving_distance(z_center), 2)
                return (dV_z * (dz/(1 + z_center))).to(u.Gpc ** 3)
            rate_integral = cosmological_rate_integral()
            
            data["N_total"][i] = N_rest_total
            data["N_bbh"][i] = N_rest_bbh
            data["N_bhns"][i] = N_rest_bhns
            data["N_bns"][i] = N_rest_bns
            data["R_total"][i] = rate_integral * N_rest_total
            data["R_bbh"][i] = rate_integral * N_rest_bbh
            data["R_bhns"][i] = rate_integral * N_rest_bhns
            data["R_bns"][i] = rate_integral * N_rest_bns

            for j in range(n_j):
                data["rates"][j][:,i] = \
                    data["rest_frame_rates"][j][:,i] * rate_integral

        #output_df: pd.DataFrame = pd.DataFrame(data)
        local_universe = z_at_centers < z_local
        local_volume = cosmo.comoving_volume(z_local).to(u.Gpc**3)        

        data["rest_frame_rates"] = np.concat(data["rest_frame_rates"], axis=0)
        data["rates"] = np.concat(data["rates"], axis=0)

        dataset = xr.Dataset(
            data_vars = {
                # binary info
                "metallicity": (["indices"], data["metallicity"]),
                "bin_num": (["indices"], data["bin_nums"]),
                "m1": (["indices"], data["m1"]),
                "m2": (["indices"], data["m2"]),
                "q": (["indices"], data["q"]),
                "t_delay": (["indices"], data["t_delay"]),
                "channel": (["indices"], data["channel"]),
                # info on initial conditions
                "m1_i": (["indices"], data["m1_i"]),
                "m2_i": (["indices"], data["m2_i"]),
                "porb_i": (["indices"], data["porb_i"]),
                "ecc_i": (["indices"], data["ecc_i"]),
                # rates for each binry in each time bin
                "rates": (["indices", "comoving_time"], data["rates"]),
                "rest_frame_rates": (["indices", "comoving_time"], data["rest_frame_rates"]),
                # time bin info
                "is_local": (["comoving_time"], local_universe),
                "E_z": (["comoving_time"], data["E_z"]),
                "N_bbh": (["comoving_time"], data["N_bbh"]),
                "N_bhns": (["comoving_time"], data["N_bhns"]),
                "N_bns": (["comoving_time"], data["N_bns"]),
                "N_total": (["comoving_time"], data["N_total"]),
                "R_bbh": (["comoving_time"], data["R_bbh"]),
                "R_bhns": (["comoving_time"], data["R_bhns"]),
                "R_bns": (["comoving_time"], data["R_bns"]),
                "R_total": (["comoving_time"], data["R_total"]),
            },
            coords = {
                "indices": (["indices"], data["indices"]),
                "comoving_time": (["comoving_time"], data["time_bins"]),
                "redshift": (["comoving_time"], z_at_centers)
            },
        )

        R_tot = dataset.R_total[local_universe].sum().data / local_volume
        R_bbh = dataset.R_bbh[local_universe].sum().data / local_volume
        R_bhns = dataset.R_bhns[local_universe].sum().data / local_volume
        R_bns = dataset.R_bns[local_universe].sum().data / local_volume

        return RatesResult(R_tot, R_bbh, R_bhns, R_bns, dataset)


    @staticmethod
    def _calculate_m1_m2_q(b: DataFrame) -> tuple:
        component_masses = b[["mass_1", "mass_2"]].to_numpy()
        primary_mass = component_masses.max(axis=1)
        secondary_mass = component_masses.min(axis=1)
        return primary_mass, secondary_mass, secondary_mass / primary_mass

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
    
    @staticmethod
    def dco_mass_filter(s: pd.DataFrame, max_ns: float = 3.0) -> tuple[pd.Series, pd.Series, pd.Series]:

        bbh = (s.mass_1 > max_ns) & (s.mass_2 > max_ns)
        bhns = ((s.mass_1 <= max_ns) & (s.mass_2 > max_ns)) | ((s.mass_1 > max_ns) & (s.mass_2 <= max_ns))
        bns = (s.mass_1 <= max_ns) & (s.mass_2 <= max_ns)

        return bbh, bhns, bns
