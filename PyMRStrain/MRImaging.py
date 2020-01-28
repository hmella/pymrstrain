from PyMRStrain.Math import FFT, iFFT
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def slice_profile(z, slice_center, slice_thickness, Nb_samples=10000):

    # Minimum and maximun coordinates
    z_min, z_max = z.min(), z.max()
    
    # Resampled domain
    z_new = np.linspace(z_min, z_max, Nb_samples)

    # Rect function
    rect = np.ones(z_new.shape)
    rect[z_new > slice_center + 0.5*slice_thickness] = 0
    rect[z_new < slice_center - 0.5*slice_thickness] = 0

    # kspace bandwidths
    bw = 1/(z_new[1]-z_new[0])
    bw_rf = 1/slice_thickness
    kz = np.linspace(-0.5*bw, 0.5*bw, Nb_samples)

    # Slice profile
    k_rect = np.ones(z_new.shape)
    k_rect[kz > 0.5*bw_rf]  = 0
    k_rect[kz < -0.5*bw_rf] = 0
    SP = np.interp(z, z_new, iFFT(k_rect*FFT(rect)))

    # # Sinc function
    # fig, ax = plt.subplots(2,2)
    # ax[0,0].plot(np.real(FFT(rect)))
    # ax[0,1].plot(z_new, np.real(iFFT(k_rect*FFT(rect))))
    # ax[1,0].plot(np.real(FFT(rect)))
    # ax[1,1].plot(z, np.real(SP), 'bo', ms=0.2)
    # plt.show()

    return SP