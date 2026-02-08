# MC-rates

This package provides classes and functions for calculating rates of transient gravitational-wave merger events. 
It is built to take binary populations evolved with [COSMIC](https://iopscience.iop.org/article/10.3847/1538-4357/ab9d85) as input; however,
any monte-carlo population which contains the right information can work.

The code uses a configurable star formation history prescription to calculate the stellar mass and metallicity of stars formed
at a slice of cosmic time, and then obtains the rate of binary coalescenes from the metallicity and delay times of each system
in your monte-carlo model. Thorough explanations of this method are available in [Dominik et al. (2013)](https://iopscience.iop.org/article/10.1088/0004-637X/779/1/72)
and [Bavera et al (2020)](https://www.aanda.org/articles/aa/full_html/2020/03/aa36204-19/aa36204-19.html).

### Installation

```
git clone https://github.com/d-maclean/MC-rates.git
cd MC-rates
pip install .
```

### Usage Example

1. Import the library

```
from astropy import units as u
from MC_rates import MCRates
```

2. Initialize an MCRates object with your chosen models

```
t0 = 600 * u.Myr
tf = 13700 * u.Myr
path_to_models = "/your/path/here"
rates_obj = MCRates(t0, tf, path_to_models, sfh_method="truncnorm")
```

3. Calculate merger rates in some local slice of the universe (supplied by `z_local`):

```
rates = rates_obj.calc_merger_rates(z_local=0.1)
```

4. Access the output object to obtain local merger rates and detaild data:

```
rates.total # total local CBC rate
rates.bbh # local BBH rate
rates.nsbh # local NSBH rate
rates.bns # local BNS rate
rates.data # xarray dataset containing rates for all systems at all time bins
```

#### Star Formation Histories:

- [Madau & Fragos (2017)](https://arxiv.org/abs/1606.07887v2)
- [Illustris TNG (2017-2019)](https://www.tng-project.org/)
- [Chruslinska & Nelemans (2019)](https://academic.oup.com/mnras/pdf-lookup/doi/10.1093/mnras/stz2057)
