from __future__ import annotations

import numpy as np
from astropy import units as u, constants as const
from scipy.stats import norm, truncnorm
from scipy.integrate import quad
from functools import partial
from astropy.cosmology import Cosmology, Planck18, z_at_value
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

from numpy.typing import NDArray
from astropy.units import Quantity
from typing import ClassVar

from rates_functions import calc_mean_metallicity_madau_fragos,\
    calc_SFR_madau_fragos, calc_lognorm_fractional_SFR,\
        calc_truncnorm_fractional_SFR, process_cosmic_models


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
        models = sorted(models, key=sort_fn)
        return models

@dataclass
class MCRates:
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
    VOLRATE: ClassVar[Quantity] = u.yr ** -1 * u.Gpc ** -3

    @classmethod
    def init_sampler(cls, t0: Quantity, tf: Quantity,
                    filepaths_to_bins: list[str], **kwargs) -> MCRates:
        '''
        Create an MCRates instance with all the necessary information to
        draw samples and calculate rates.
        ### Parameters:
        - t0, tf: Quantity - earliest and latest comoving time values
        - filepaths_to_bins: list[str] - the location at which to find models
        - **kwargs
        ### Returns:
        - MCRates
        '''
        cosmo: Cosmology = kwargs.get("cosmology", Planck18)
        num_pts: int = kwargs.get("num_pts", 1000)
        sfr_function: function = kwargs.get("SFR_function", calc_SFR_madau_fragos)
        avg_met_function: function = kwargs.get("avg_met_function", calc_mean_metallicity_madau_fragos)
        Zfracs_method: str = kwargs.get("Zfracs_method", "lognorm")
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
        
        modeled_SFR_fracs = np.ones(shape=num_pts)
        
        if Zfracs_method == "lognorm":
            fracSFR = calc_lognorm_fractional_SFR(
                Zbins = metallicities,
                meanZ = mean_metallicity_at_z,
                redshift = redshift, 
                SFR_at_redshift = SFR_at_z,
                sigma_logZ = logZ_sigma_for_SFR
                )
        elif Zfracs_method == "truncnorm":
            fracSFR = calc_truncnorm_fractional_SFR(
                Zbins = metallicities,
                meanZ = mean_metallicity_at_z,
                redshift = redshift, 
                SFR_at_redshift = SFR_at_z,
                sigma_logZ = logZ_sigma_for_SFR
                )
        
        # create our object and return
        params = cls(
            cosmology=cosmo,
            comoving_time=comoving_time,
            redshift=redshift,
            SFR_at_z=SFR_at_z,
            metallicities=metallicities,
            bins=bins_list,
            mean_met_at_z=mean_metallicity_at_z,
            fractional_SFR_at_met=fracSFR,
            fcorr_SFR_fracs=modeled_SFR_fracs
        )
        return params
    
    # trying once again
    def calc_merger_rates(self,
                          nbins: int = 200, z_local: float = 0.1,
                          **kwargs) -> \
                              tuple[Quantity, Quantity, Quantity, Quantity, pd.DataFrame]:
        '''Calculate dco merger rates per Dominik+2013.'''
        primary_mass_lims: tuple | None = kwargs.get("primary_mass_lims", None)
        secondary_mass_lims: tuple | None = kwargs.get("secondary_mass_lims", None)
        Zlims: tuple | None = kwargs.get("Zlims", None)
        use_Zfracs_correction: bool = kwargs.get("use_Zfracs_corr", False)
        
        if use_Zfracs_correction:
            Zfracs_corr = self.fcorr_SFR_fracs
        else:
            Zfracs_corr = np.ones(shape=self.comoving_time.shape[0])
        
        mass_filter_pri: bool = False
        mass_filter_sec: bool = False
        Z_filter: bool = False
        
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
        n_j: int = len(self.bins)
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
        
        data: dict[str:NDArray] = {
            "t_center": time_bin_centers.to(u.Myr),
            "t_i": time_bin_edges[:-1].to(u.Myr),
            "t_f": time_bin_edges[1:].to(u.Myr),
            "z_center": z_at_centers,
            "z_i": z_at_edges[:-1],
            "z_f": z_at_edges[1:],
            "E_z": E_z_at_centers,
            "R_bbh": np.zeros(shape=n) * self.VOLRATE,
            "R_bhns": np.zeros(shape=n) * self.VOLRATE,
            "R_bns": np.zeros(shape=n) * self.VOLRATE,
            "R_total": np.zeros(shape=n) * self.VOLRATE,
            "R_i_bbh": np.zeros(shape=n) * self.VOLRATE,
            "R_i_bhns": np.zeros(shape=n) * self.VOLRATE,
            "R_i_bns": np.zeros(shape=n) * self.VOLRATE,
            "R_i_total": np.zeros(shape=n) * self.VOLRATE
        }
        
        for i, t_center in enumerate(tqdm(time_bin_centers, desc="Comoving time bins", unit="bins")):
            
            t_i = time_bin_edges[i]
            t_f = time_bin_edges[i+1]
            z_center: float = z_at_centers[i]
            z_i = z_at_edges[i]
            z_f = z_at_edges[i+1]
            dz = np.abs(z_i - z_f)
            E_z = E_z_at_centers[i]
            
            R_i_total = 0.0 * self.VOLRATE
            R_i_bbh = 0.0 * self.VOLRATE
            R_i_bhns = 0.0 * self.VOLRATE
            R_i_bns = 0.0 * self.VOLRATE
            
            for j, Zbin in enumerate(self.bins):
                
                if Z_filter:
                    if (Zbin.metallicity > Z_max) or (Zbin.metallicity < Z_min):
                        continue
                
                #met: float = Zbin.metallicity
                binfrac: float = Zbin.binfrac_model
                Msim: float = Zbin.simulated_mass
                f_corr: float = Zbin.imf_f_corr
                
                systems: pd.DataFrame = Zbin.mergers[Zbin.mergers.t_delay.values * u.Myr < t_center]
                
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
                Zfracs_corr_at_t_form: NDArray = Zfracs_corr[SFR_bins_for_systems]
                
                # calculate the rate -- see Dominik et al 2013 Eq. 16/17
                Rates_intrinsic: NDArray = (\
                    (binfrac * f_corr) * (1/Zfracs_corr_at_t_form) * (frac_SFR_at_t_form / Msim)).to(self.VOLRATE)
                
                R_i_total += Rates_intrinsic.sum()
                R_i_bbh += Rates_intrinsic[bbh].sum()
                R_i_bhns += Rates_intrinsic[bhns].sum()
                R_i_bns += Rates_intrinsic[bns].sum()
                
            # append data to output dict
            dzdt: float = dz * (((1 + z_center) * E_z) ** -1)
            data["R_i_total"][i] = R_i_total
            data["R_i_bbh"][i] = R_i_bbh
            data["R_i_bhns"][i] = R_i_bhns
            data["R_i_bns"][i] = R_i_bns
            data["R_total"][i] = R_i_total * dzdt
            data["R_bbh"][i] = R_i_bbh * dzdt
            data["R_bhns"][i] = R_i_bhns * dzdt
            data["R_bns"][i] = R_i_bns * dzdt

        output_df: pd.DataFrame = pd.DataFrame(data)
        R_total_local: Quantity = data["R_total"][z_center < z_local].sum()
        R_bbh_local: Quantity = data["R_bbh"][z_center < z_local].sum()
        R_bhns_local: Quantity = data["R_bhns"][z_center < z_local].sum()
        R_bns_local: Quantity = data["R_bns"][z_center < z_local].sum()

        return R_total_local, R_bbh_local, R_bhns_local, R_bns_local, output_df
    
    @staticmethod
    def _get_bins_from_time(t: NDArray | Quantity, time_bins: NDArray) -> NDArray | Quantity:
        '''Return the index of the closest calculated time point from time values.'''
        diffs = np.abs(t[:,np.newaxis] - time_bins)
        return np.argmin(diffs, axis=1)
        
        #indices: NDArray = np.zeros(shape=t.shape[0], dtype=int)
        #for k in range(t.shape[0]):
        #    indices[k] = np.argmin(np.abs(t[k] - time_bins))
        #
        #return indices
    
    @staticmethod
    def dco_kstar_filter(s: pd.DataFrame) -> tuple[NDArray, NDArray, NDArray]:

        bbh = (s.kstar_1 == 14) & (s.kstar_2 == 14)
        bhns = ((s.kstar_1 == 13) & (s.kstar_2 == 14)) | ((s.kstar_1 == 14) & (s.kstar_2 == 13))
        bns = (s.kstar_1 == 13) & (s.kstar_2 == 13)
        
        return bbh, bhns, bns 
    
    @staticmethod
    def calc_Zpdf(Z_values: NDArray,
                meanZ: NDArray, sigma_logZ: float, num_pts: int) -> tuple[NDArray, NDArray]:
        '''Use an integrated log-normal metallicity PDF to estimate the fractional SFR for each metallicity bin.'''
        Zrange = np.linspace(1e-10, 0.2, num_pts)
        rawZpdf = np.zeros(shape=(num_pts, num_pts), dtype=float)
        Zcdf = rawZpdf.copy()
        
        for i in range(num_pts):
            _pdf = norm.pdf(x=np.log10(Zrange), loc=np.log10(meanZ[i]), scale=sigma_logZ)
            rawZpdf[:,i] = _pdf/_pdf.sum()
            _cdf = np.cumsum(rawZpdf[:,i])
            Zcdf[:,i] = _cdf
        
        # set edges of Zbins
        Zbin_edges: NDArray = np.zeros(shape=len(Z_values)+1)
        for j, jZ  in enumerate(Z_values):
            if j == 0:
                print("aaa!")
                Zbin_edges[j+1] = np.mean(Z_values[j:j+2])
                Zbin_edges[j] = jZ - (Zbin_edges[j+1] - jZ)
            elif i == len(Z_values) - 1:
                Zbin_edges[j] = np.mean(Z_values[j-1:j+1])
                Zbin_edges[j+1] = jZ + (Zbin_edges[j] - jZ)
            else:
                Zbin_edges[j] = np.mean(Zbin_edges[j-1:j+1])
                Zbin_edges[j+1] = np.mean(Zbin_edges[j:j+2])
        
        # calculate fSFR for each bin at each time value
        fSFR_at_z: NDArray = np.zeros(shape=(len(Z_values), num_pts))
        for j, jZ in enumerate(Z_values):
            Z_lo = Zbin_edges[j]
            Z_hi = Zbin_edges[j+1]
            
            for k in range(num_pts):
                Pvals = np.interp([Z_lo, Z_hi], Zrange, Zcdf[:,k])
                P_tz: float = np.abs(np.diff(Pvals))
                fSFR_at_z[j,k] = P_tz

        return Zcdf, Zbin_edges, fSFR_at_z
    