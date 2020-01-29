import numpy as np
from PyMRStrain.MRImaging import *
import matplotlib.pyplot as plt

# Acquisition matrix
acq_matrix = [64,256]

# kspace
k = np.ones(acq_matrix)

k_test = acq_to_res(k, acq_matrix, [32,128], dir=[1,0])

plt.imshow(k_test.T)
plt.show()