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
from astropy import units as u
from MC_rates import MCSampler

# point to your data
path_to_data = "../path-to-files/output-sse"
t0 = 300 * u.Myr
tf = 13700 * u.Myr

# make a sampler with comoving time values spaced between t0 and tf
sampler = MCSampler(t0, tf, path_to_data)

# draw 100 samples in time for each system, weighted by the relative
# star formation density of each metallicity across cosmic time
sample = sampler.draw_mc_sample(5, nproc=2, seed=0)

# use the sample to calculate merger rates between 0 < z < 0.1
rates = sampler.calc_rates(sample, 0, 0.1, 4)
total_merger_rate = rates.rate.sum()
```

### Acknowledgments


