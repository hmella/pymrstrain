import numpy as np
from PyMRStrain.MRImaging import *
import matplotlib.pyplot as plt

# Acquisition matrices
acq_matrix = [256,64]

# kspace
k = np.zeros(acq_matrix, dtype=complex)

# Filter size
s = np.array(k.shape)
s20 = np.round(0.2*s).astype(int)
s0 = np.linspace(0, s20[0]/2, s20[0])
s1 = np.linspace(0, s20[1]/2, s20[1])
w0 = 1.0 - np.power(np.abs(s0/(s20[0]/2)),2)
w1 = 1.0 - np.power(np.abs(s1/(s20[1]/2)),2)

O0 = np.ones([s[0],])
O1 = np.ones([s[1],])

O0[0:s20[0]] *= np.flip(w0)
O0[s[0]-s20[0]:s[0]] *= w0
O1[0:s20[1]] *= np.flip(w1)
O1[s[1]-s20[1]:s[1]] *= w1

plt.imshow(np.outer(O0,O1))
plt.show()
