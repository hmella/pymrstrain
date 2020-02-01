import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def Hamming_filter(acq_matrix, dir):
    H0 = signal.hamming(acq_matrix[dir[1]])
    H1 = np.ones(acq_matrix[dir[0]])
    return np.outer(H0, H1).flatten('C')


def Riesz_filter(acq_matrix, dir, decay=0.2):
    s = acq_matrix[dir[1]]
    s20 = np.round(decay*s).astype(int)
    s1 = np.linspace(0, s20/2, s20)
    w1 = 1.0 - np.power(np.abs(s1/(s20/2)),2)

    # Filters
    H0 = np.ones([s,])
    H0[0:s20] *= np.flip(w1)
    H0[s-s20:s] *= w1
    H1 = np.ones(acq_matrix[dir[0]])

    return np.outer(H0,H1).flatten('C')

def Tukey_filter(acq_matrix, dir, alpha=0.2):
    H0 = signal.tukey(acq_matrix[dir[1]], alpha=alpha)
    H1 = np.ones(acq_matrix[dir[0]])
    return np.outer(H0,H1).flatten('C')
