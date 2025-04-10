import os
import sys

sys.path.append(
    os.path.split(os.path.abspath(__file__))[0]
    )

from rates_classes import BinariesBin, MCSampler
from rates_functions import calc_SFR_madau_fragos, calc_mean_metallicity_madau_fragos

if __name__ == "__main__":
    
    sys.exit(0)
