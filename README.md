# MC-rates

This package provides two classes and some functions to perform transient event rates calculations using
binary populations evolved with [COSMIC](https://github.com/COSMIC-PopSynth/COSMIC). Theoretically it can work with other formats, but I haven't implemented any yet. 

### Installation

```
git clone https://github.com/d-maclean/MC-rates.git
cd MC-rates
pip install .
```

### Usage Example

```
from glob import glob
from astropy import units as u
from MC_rates import MCParams, init_sampler, calc_MC_rates

# point to your data
files = glob("../path-to-files/output-sse/*.hdf5")
t0 = 300 * u.Myr
tf = 13700 * u.Myr

# make a sampler with comoving time values spaced between t0 and tf
sampler = init_sampler(t0, tf, files)

# draw 100 samples in time for each system, weighted by the relative
# star formation density of each metallicity across cosmic time
# use the sample to calculate merger rates between 0 < z < 0.1
rates = calc_MC_rates(sampler, n=100, seed=0)
```

### References & Further Reading

- [Bavera et al. (2020)](https://doi.org/10.1051/0004-6361/201936204)
- [Dominik et al. (2015)](https://iopscience.iop.org/article/10.1088/0004-637X/806/2/263)
- [Madau & Fragos (2017)](https://arxiv.org/abs/1606.07887v2)
