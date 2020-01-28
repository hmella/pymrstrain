import numpy as np
from PyMRStrain.MRImaging import *

# Image
I = np.zeros([30,60])
pxsz = [0.003, 0.003]

# Image coordinates
[X,Y] = np.meshgrid(np.linspace(-1,1,60),np.linspace(-1,1,30))
R = X**2 + Y**2
I[R <= 0.5**2] = 1

acquisition_artifact(I, pxsz[1], receiver_bandwidth=32*1000, T2star=0.02)