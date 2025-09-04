import os
import h5py
import numpy as np
from astropy import units as u

from numpy.typing import NDArray

def illustris_TNG_SFH(comoving_time: NDArray,
                      metallicities: NDArray, filepath: str | None = None) -> dict:
    '''
    Function to calculate star formation history using a sample of Illustris TNG100 data.
    ### Parameters
    - comoving_time: ArrayLike[Quantity]
    - metallicities: ArrayLike
    - filepath: str | os.PathLike = None - override if desired, else use default
    - cosmo: astropy.cosmology.Cosmology = Planck15
    ### Returns
    - SFH: dict
    ### Note
    We handle the issue of non-modeled mass similarly to how we do w/r/t Madau & Fragos. We renormalize
    the total star formation to ensure that our (incomplete) metallicity range recovers the total SF
    in the TNG simulation.
    '''
    assert (np.diff(comoving_time) > 0).all()
    assert (np.diff(metallicities) > 0).all()
    n_i: int = len(comoving_time)
    n_j: int = len(metallicities)
    VOLUME_FCORR = 100 ** -3
    SFRUNIT = u.Msun * u.yr ** -1 * u.Mpc ** -3 #type: ignore
    time: NDArray = comoving_time.to(u.yr).value #type: ignore
    
    # our bin edges
    Z_edges = 10 ** (np.log10(metallicities[:-1]) + ((np.log10(metallicities[1:]) - np.log10(metallicities[:-1]))/2))
    Z_edges = np.concat([[metallicities[0]], Z_edges, [metallicities[-1]]])
    t_edges = 10 ** (np.log10(time[:-1]) + ((np.log10(time[1:]) - np.log10(time[:-1]))/2))
    t_edges = np.concat([[time[0]], t_edges, [time[-1]]])
    
    print(f"Z_edges: {Z_edges.shape} / t_edges: {t_edges.shape}")
    
    if filepath is None:
        filepath = os.path.join(os.path.split(os.path.abspath(__file__))[0], "TNG100_L75n1820TNG__x-t-log_y-Z-log.hdf5")
        
    with h5py.File(filepath, 'r') as file:
        data = load_illustris_data(file)
        tng_time_edges = np.squeeze(data["time"])[:]
        tng_dt = tng_time_edges[1:] - tng_time_edges[:-1]
        tng_Z_edges = np.squeeze(data["metallicity"])[1:-1]
        tng_mass_formed = np.squeeze(data["mass_formed"])[:,1:-1]
    
    if (np.diff(tng_Z_edges) < 0).any():
        tng_Z_edges = tng_Z_edges[::-1]
        tng_mass_formed = tng_mass_formed[:,::-1]
    if (tng_dt < 0).any():
        tng_time_edges = tng_time_edges[::-1]
        tng_mass_formed = tng_mass_formed[::-1,:]
        
    tng_time = \
        10 ** (np.log10(tng_time_edges[:-1]) + ((np.log10(tng_time_edges[1:]) - np.log10(tng_time_edges[:-1]))/2))
    tng_Z = \
        10 ** (np.log10(tng_Z_edges[:-1])+((np.log10(tng_Z_edges[1:])-np.log10(tng_Z_edges[:-1]))/2))
    
    # total sf in each time bin
    tng_sf = np.sum(tng_mass_formed, axis = 1)
    total_tng_sfr = np.zeros(shape=n_i, dtype=float)

    # re-bin illustris metallicities
    Z_binned_sf = np.zeros(shape=(n_j, tng_time.shape[0]), dtype=float)
    for j in range(n_j):
        Z_lo = Z_edges[j]
        Z_hi = Z_edges[j+1]
        
        Z_bins_mask = (tng_Z >= Z_lo) & (tng_Z <= Z_hi)
        mass_in_dZ = tng_mass_formed[:,Z_bins_mask]
        Z_binned_sf[j,:] = np.squeeze(np.sum(mass_in_dZ, axis = 1))
        
    # re-bin illustris time
    fracSFR = np.zeros(shape=(n_j, n_i), dtype=float)
    mean_Z_at_t = np.zeros(shape=n_i, dtype=float)
    for i in range(n_i):
        t_cntr = time[i]
        t_lo = t_edges[i]
        t_hi = t_edges[i+1]
        
        t_bins_mask = (tng_time >= t_lo) & (tng_time <= t_hi)
        
        if t_bins_mask.sum() < 1: # if no bins are contained, choose the closest
            t_bins_mask = np.zeros(shape=tng_time.shape, dtype=bool)
            closest = np.argmin(np.abs(tng_time - t_cntr))
            t_bins_mask[closest] = True
            
        dt = tng_dt[t_bins_mask].sum()
                
        sfr_ij = Z_binned_sf[:,t_bins_mask].sum(axis = 1) / dt * VOLUME_FCORR
        cdf = np.cumsum(sfr_ij)
        cdf /= cdf.sum()
        _z = np.interp(0.5e0, cdf, metallicities) # TODO
        mean_Z_at_t[i] = _z
        total_tng_sfr[i] = tng_sf[t_bins_mask].sum(axis=None) / dt * VOLUME_FCORR
        fracSFR[:,i] = sfr_ij

    
    sfh = {
        "tng_dt": tng_dt * u.yr, #type: ignore
        "time": time * u.yr, # type: ignore
        "metallicities": metallicities,
        "mean_metallicity": mean_Z_at_t,
        "SFR_at_z": total_tng_sfr * SFRUNIT,
        "fractional_SFR": fracSFR * SFRUNIT
        }
    
    return sfh


def load_illustris_data(h5file: h5py.File) -> dict[str, NDArray]:
    '''What it says on the tin.'''
    tng_data = {
        "time": np.array(h5file["xedges"]),
        "metallicity": np.array(h5file["yedges"]),
        "mass_formed": np.array(h5file["mass"]),
        }
    
    return tng_data
