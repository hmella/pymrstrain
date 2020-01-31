import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def Hamming_filter(acq_matrix, dir):
    H0 = signal.hamming(acq_matrix[dir[0]])
    H1 = signal.hamming(acq_matrix[dir[1]])
    return np.outer(H0, H1).flatten('F')


def Riesz_filter(acq_matrix, dir, decay=0.2):
    s = np.array(acq_matrix[dir])
    s20 = np.round(decay*s).astype(int)
    s0 = np.linspace(0, s20[0]/2, s20[0])
    s1 = np.linspace(0, s20[1]/2, s20[1])
    w0 = 1.0 - np.power(np.abs(s0/(s20[0]/2)),2)
    w1 = 1.0 - np.power(np.abs(s1/(s20[1]/2)),2)

    # Filters
    H0 = np.ones([s[0],])
    H0[0:s20[0]] *= np.flip(w0)
    H0[s[0]-s20[0]:s[0]] *= w0

    H1 = np.ones([s[1],])
    H1[0:s20[1]] *= np.flip(w1)
    H1[s[1]-s20[1]:s[1]] *= w1

    # plt.imshow(np.outer(H0,H1))
    # plt.show()

    return np.outer(H0,H1).flatten('F')

def Tukey_filter(acq_matrix, dir, alpha=0.2):
    H0 = signal.tukey(acq_matrix[dir[0]], alpha=alpha)
    H1 = signal.tukey(acq_matrix[dir[1]], alpha=alpha)
    return np.outer(H0,H1).flatten('F')