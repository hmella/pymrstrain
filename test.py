import numpy as np
from PyMRStrain.MRImaging import *
import matplotlib.pyplot as plt

# Acquisition matrices
acq_matrix_1 = [256,64]
acq_matrix_2 = [64,256]

# kspace
k_1 = np.zeros(acq_matrix_1, dtype=complex)
k_2 = np.zeros(acq_matrix_2, dtype=complex)
for i in range(256):
    k_1[i,:] = i+100
    k_2[:,i] = i+100
k_test_1 = acq_to_res(k_1, acq_matrix_1, [128,128], dir=[0,1])
k_test_2 = acq_to_res(k_2, acq_matrix_2, [128,128], dir=[1,0])

fig, ax = plt.subplots(1,2)
ax[0].imshow(np.abs(k_test_1).T)
ax[1].imshow(np.abs(k_test_2).T)
plt.show()
