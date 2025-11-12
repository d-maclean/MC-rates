import os
import sys
__version__ = "0.4"

sys.path.append(
    os.path.split(os.path.abspath(__file__))[0]
    )

from rates import MCRates, Model
