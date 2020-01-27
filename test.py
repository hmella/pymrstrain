import numpy as np
from PyMRStrain.Imaging import *

# Coordinates
z = np.linspace(-1,1,1000)
slice_center = 0.0
slice_thickness = 0.2

slice_profile(z, slice_center, slice_thickness, Nb_samples=1e+04)