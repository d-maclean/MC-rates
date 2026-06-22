import os
import sys
__version__ = "0.5.1"

sys.path.append(
    os.path.split(os.path.abspath(__file__))[0]
    )

from rates import *
