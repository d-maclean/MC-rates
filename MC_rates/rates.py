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
    '''
    Nmaed tuple containing rates output. Local rates are reported in yr^-1 Gpc^-3.

    ### Fields

    `total`: the total local rate of CBCs

    `bbh`: the local binary black hole rate

    `nsbh`: the local neutron star/black hole rate

    `bns`: the local binary neutron star rate

    `data`: a DataFrame of binned rest-frame rates for each comoving time bin

    `hist`: histogram data and bins
    '''
    total: Quantity
    bbh: Quantity
    nsbh: Quantity
    bns: Quantity
    data: DataFrame
    hist: dict

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

        `histogram`: whether to output histogram data. default=`False`.

        `bins_bbh`: histogram bins for BBHs. default=`10`.

        `bins_nsbh`: histogram bins for NSBHs. default=`10`.

        `bins_bns`: histogram bins for BNSe. default=`10`.

        `bins_q`: histogram bins for q. default=`10`.

        `bins_t_delay`: histogram bins for delay time. delay times are reported in log scale. default=`10`.

        `show_tqdm`: show a progress bar. default=`True`.

        ### Returns

        `RatesResult`: a named tuple containing the following fields:
            - `total`: sum of all CBCs (yr^-1 Gpc^-3)
            - `bbh`: BBHs (yr^-1 Gpc^-3)
            - `nsbh`: NSBHs (yr^-1 Gpc^-3)
            - `bns`: BNSe (yr^-1 Gpc^-3)
            - `data`: DataFrame of rates calculations
            - `hist`: dictionary of histogram data and bins
        
        '''
        primary_mass_lims: tuple | None = kwargs.get("primary_mass_lims", None)
        secondary_mass_lims: tuple | None = kwargs.get("secondary_mass_lims", None)
        q_lims: tuple | None = kwargs.get("q_lims", None)
        Zlims: tuple | None = kwargs.get("Zlims", None)
        max_ns: float | None = kwargs.get("max_ns", None)
        optimistic_ce: bool = kwargs.get("optimistic_ce", True)
        histogram: bool = kwargs.get("histogram", False)
        bins_bbh: int | NDArray = kwargs.get("bins_bbh", 10) #type: ignore
        bins_nsbh: int | NDArray = kwargs.get("bins_nsbh", 10) #type: ignore
        bins_bns: int | NDArray = kwargs.get("bins_bns", 10) #type: ignore
        bins_q: int | NDArray = kwargs.get("bins_q", 10) # type: ignore
        bins_t_delay: int | NDArray = kwargs.get("bins_t_delay", 10) # type: ignore
        show_tqdm: bool = kwargs.get("tqdm", True)

        n: int = nbins
        n_j: int = self.bins.shape[0]
        cosmo: Cosmology = self.cosmology

        if histogram:
            _mxns = max_ns if max_ns else 3.0
            if isinstance(bins_bbh, int): bins_bbh = np.linspace(_mxns, 60.1, bins_bbh)
            if isinstance(bins_nsbh, int): bins_nsbh = np.linspace(_mxns, 60.1, bins_nsbh)
            if isinstance(bins_bns, int): bins_bns = np.linspace(1.0, _mxns, bins_bns)
            if isinstance(bins_q, int): bins_q = np.linspace(0.0, 1.0, bins_q)
            if isinstance(bins_t_delay, int):\
                bins_t_delay = np.linspace(0.0, np.log10(cosmo.age(0).to(u.yr).value), bins_t_delay)
        
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

        hist_m1_data = {"bbh": [], "nsbh": [], "bns": []}
        hist_q_data = {"bbh": [], "nsbh": [], "bns": []}
        hist_td_data = {"bbh": [], "nsbh": [], "bns": []}
        
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
            
            hist_dZ_m1 = {"bbh": [], "nsbh": [], "bns": []}
            hist_dZ_q = {"bbh": [], "nsbh": [], "bns": []}
            hist_dZ_td = {"bbh": [], "nsbh": [], "bns": []}

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

                t_delay_filter = np.ones(nk, dtype=bool)
                mass_filter = t_delay_filter.copy()
                ce_filter = t_delay_filter.copy()
                q_filter = t_delay_filter.copy()
                
                t_delay_filter = \
                    (Zbin.mergers.t_delay.values < \
                    (t_center.to(u.Myr).value - self.comoving_time[0].to(u.Myr).value))
                #systems: pd.DataFrame = Zbin.mergers.loc[t_delay_filter]

                if not optimistic_ce:
                    if "merge_in_ce" not in systems.columns:
                        print("Warning: Can't find `merge_in_ce` column in your data files. Proceeding with optimistic_ce.")
                    else:
                        ce_filter = ~systems.merge_in_ce
                        #systems = systems.loc[~systems.merge_in_ce] # type: ignore
                
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
                    #systems = systems[mass_filter]
                if do_q_filter:
                    q = (secondary_mass / primary_mass)
                    q_filter = (q >= q_min) & (q <= q_max)
                    #systems = systems[q_filter]
                
                systems = systems[(t_delay_filter) & (mass_filter) & (q_filter) & (ce_filter)]
                sys_mass_pri, sys_mass_sec, sys_q = self._calculate_m1_m2_q(systems)

                t_form: NDArray = (t_center - systems.t_delay.values * u.Myr).to(u.Myr)
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
                
                if histogram:
                    # bbh
                    _hist_bbh, _ = np.histogram(sys_mass_pri[bbh],
                                                        bins=bins_bbh, weights=(frac_SFR_at_t_form[bbh] / Msim))
                    # nsbh
                    _hist_bhns, _ = np.histogram(sys_mass_pri[bhns],
                                                        bins=bins_nsbh, weights=(frac_SFR_at_t_form[bhns] / Msim))
                    # bns
                    _hist_bns, _ = np.histogram(sys_mass_pri[bns],
                                                        bins=bins_bns, weights=(frac_SFR_at_t_form[bns] / Msim))
                    hist_dZ_m1["bbh"].append(_hist_bbh)
                    hist_dZ_m1["nsbh"].append(_hist_bhns)
                    hist_dZ_m1["bns"].append(_hist_bns)
                    
                    # q
                    _hist_qbbh, _ = np.histogram(sys_q[bbh],
                                                        bins=bins_q, weights=(frac_SFR_at_t_form[bbh] / Msim))
                    _hist_qbhns, _ = np.histogram(sys_q[bhns],
                                                        bins=bins_q, weights=(frac_SFR_at_t_form[bhns] / Msim))
                    _hist_qbns, _ = np.histogram(sys_q[bns],
                                                        bins=bins_q, weights=(frac_SFR_at_t_form[bns] / Msim))
                    hist_dZ_q["bbh"].append(_hist_qbbh)
                    hist_dZ_q["nsbh"].append(_hist_qbhns)
                    hist_dZ_q["bns"].append(_hist_qbns)

                    # t_delay
                    log_td = np.log10((systems.t_delay.values * u.Myr).to(u.yr).value)
                    _hist_tbbh, _ = np.histogram(log_td[bbh],
                                                        bins=bins_t_delay, weights=(frac_SFR_at_t_form[bbh] / Msim))
                    _hist_tbhns, _ = np.histogram(log_td[bhns],
                                                        bins=bins_t_delay, weights=(frac_SFR_at_t_form[bhns] / Msim))
                    _hist_tbns, _ = np.histogram(log_td[bns],
                                                        bins=bins_t_delay, weights=(frac_SFR_at_t_form[bns] / Msim))
                    hist_dZ_td["bbh"].append(_hist_tbbh)
                    hist_dZ_td["nsbh"].append(_hist_tbhns)
                    hist_dZ_td["bns"].append(_hist_tbns)

                N_rest_total += Rates_intrinsic.sum()
                N_rest_bbh += Rates_intrinsic[bbh].sum()
                N_rest_bhns += Rates_intrinsic[bhns].sum()
                N_rest_bns += Rates_intrinsic[bns].sum()
                
            # prepare rates integral
            def rate_integral() -> Quantity:
                '''Integrate the rest-frame event rates to obtain a local, observable rate.'''
                dV_z = 4 * np.pi * (const.c / cosmo.H(z_center)) *\
                    np.float_power(cosmo.comoving_distance(z_center), 2)
                return (dV_z * (dz/(1 + z_center))).to(u.Gpc ** 3)
            
            rate_int = rate_integral()
            
            data["N_total"][i] = N_rest_total
            data["N_bbh"][i] = N_rest_bbh
            data["N_bhns"][i] = N_rest_bhns
            data["N_bns"][i] = N_rest_bns
            data["R_total"][i] = rate_int * N_rest_total
            data["R_bbh"][i] = rate_int * N_rest_bbh
            data["R_bhns"][i] = rate_int * N_rest_bhns
            data["R_bns"][i] = rate_int * N_rest_bns

            if histogram: # multiply by rate_int factor to account for redshift!
                hist_m1_data["bbh"].append(rate_int * np.sum(hist_dZ_m1["bbh"], axis=0))
                hist_m1_data["nsbh"].append(rate_int * np.sum(hist_dZ_m1["nsbh"], axis=0))
                hist_m1_data["bns"].append(rate_int * np.sum(hist_dZ_m1["bns"], axis=0))
                hist_q_data["bbh"].append(rate_int * np.sum(hist_dZ_q["bns"], axis=0))
                hist_q_data["nsbh"].append(rate_int * np.sum(hist_dZ_q["nsbh"], axis=0))
                hist_q_data["bns"].append(rate_int * np.sum(hist_dZ_q["bns"], axis=0))
                hist_td_data["bbh"].append(rate_int * np.sum(hist_dZ_td["bbh"], axis=0))
                hist_td_data["nsbh"].append(rate_int * np.sum(hist_dZ_td["nsbh"], axis=0))
                hist_td_data["bns"].append(rate_int * np.sum(hist_dZ_td["bns"], axis=0))
        
        output_df: pd.DataFrame = pd.DataFrame(data)
    
        local_universe = output_df.z_center < z_local
        local_volume = cosmo.comoving_volume(z_local).to(u.Gpc**3)
        R_tot = output_df.R_total.loc[local_universe].sum() / (u.yr * local_volume)
        R_bbh = output_df.R_bbh.loc[local_universe].sum() / (u.yr * local_volume)
        R_bhns = output_df.R_bhns.loc[local_universe].sum() / (u.yr * local_volume)
        R_bns = output_df.R_bns.loc[local_universe].sum() / (u.yr * local_volume)

        if histogram:
            hist = {}
            _hbbh = np.array(hist_m1_data["bbh"], dtype=object)
            _hbhns = np.array(hist_m1_data["nsbh"], dtype=object)
            _hbns = np.array(hist_m1_data["bns"], dtype=object)
            _hqbbh = np.array(hist_q_data["bbh"], dtype=object)
            _hqnsbh = np.array(hist_q_data["nsbh"], dtype=object)
            _hqbns = np.array(hist_q_data["bns"], dtype=object)
            _htbbh = np.array(hist_td_data["bbh"], dtype=object)
            _htnsbh = np.array(hist_td_data["nsbh"], dtype=object)
            _htbns = np.array(hist_td_data["bns"], dtype=object)

            hist["bbh"] = np.sum(_hbbh[np.nonzero(local_universe)[0]], axis=0).astype(float) * 1e9
            hist["nsbh"] = np.sum(_hbhns[np.nonzero(local_universe)[0]], axis=0).astype(float) * 1e9
            hist["bns"] = np.sum(_hbns[np.nonzero(local_universe)[0]], axis=0).astype(float) * 1e9
            hist["q_bbh"] = np.sum(_hqbbh[np.nonzero(local_universe)[0]], axis=0).astype(float) * 1e9
            hist["q_nsbh"] = np.sum(_hqnsbh[np.nonzero(local_universe)[0]], axis=0).astype(float) * 1e9
            hist["q_bns"] = np.sum(_hqbns[np.nonzero(local_universe)[0]], axis=0).astype(float) * 1e9
            hist["td_bbh"] = np.sum(_htbbh[np.nonzero(local_universe)[0]], axis=0).astype(float) * 1e9
            hist["td_nsbh"] = np.sum(_htnsbh[np.nonzero(local_universe)[0]], axis=0).astype(float) * 1e9
            hist["td_bns"] = np.sum(_htbns[np.nonzero(local_universe)[0]], axis=0).astype(float) * 1e9

            hist["bins_bbh"] = bins_bbh
            hist["bins_nsbh"] = bins_nsbh
            hist["bins_bns"] = bins_bns
            hist["bins_q"] = bins_q
            hist["bins_t_delay"] = bins_t_delay
        else:
            hist = {}

        return RatesResult(R_tot, R_bbh, R_bhns, R_bns, output_df, hist)
    
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
