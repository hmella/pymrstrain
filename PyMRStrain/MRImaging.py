from PyMRStrain.Math import FFT, iFFT
import numpy as np
import matplotlib.pyplot as plt

def acquisition_artifact(I, delta, receiver_bandwidth=64*1000,
                         T2star=0.02):

    # kspace bandwith
    k_max = 0.5/delta

    # kspace of the input image
    k = FFT(I)
    km_profiles = k.shape[1]
    kp_profiles = k.shape[0]

    # Receiver bandwidth and acquisition time per line
    Tk = km_profiles*1.0/(2.0*receiver_bandwidth)

    # Acquisition times
    t = acquisition_times(k, Tk, fast_imaging_mode='EPI')

    # MTF
    MTF_trunc = np.ones(I.shape)
    MTF_decay = np.exp(-t/T2star)
    MTF = MTF_trunc*MTF_decay

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(t)
    ax[1].imshow(I)
    ax[2].imshow(np.abs(iFFT(MTF*k)))
    plt.show()


# Generates acquisition times depending of the
# acquisition sequence and imaging mode
def acquisition_times(kspace,Tk,fast_imaging_mode='EPI', multishot=False):

    # kspace of the input image
    km_profiles = kspace.shape[1]
    kp_profiles = kspace.shape[0]

    # Output times

    # Acquisition times for different cartesian techniques:
    # EPI
    if fast_imaging_mode=='EPI':
        t = np.linspace(0,Tk*kp_profiles,km_profiles*kp_profiles)
        t = t.reshape(kspace.shape)
        for i in range(t.shape[0]):
            if i % 2 == 0:
                t[i,:] = t[i,:]
            else:
                t[i,:] = np.flip(t[i,:])

    # Simpler GRE
    if fast_imaging_mode=='None':
        delta = 0.002
        t = np.linspace(0,Tk*kp_profiles,km_profiles*kp_profiles)
        t = t.reshape(kspace.shape)
        for i in range(t.shape[0]):
            t[i,:] += i*Tk + delta

    return t