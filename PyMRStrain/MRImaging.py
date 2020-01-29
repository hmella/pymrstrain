from PyMRStrain.Math import itok, ktoi
import numpy as np
import matplotlib.pyplot as plt

def acquisition_artifact(k, delta, receiver_bandwidth=64*1000,
                         T2star=0.02, fast_imaging_mode='EPI'):

    # kspace bandwith
    k_max = 0.5/delta

    # kspace of the input image
    km_profiles = k.shape[0]
    kp_profiles = k.shape[1]
    ky = np.linspace(-k_max,k_max,kp_profiles)
    for i in range(1,km_profiles):
        ky = np.append(ky, ky[0:kp_profiles])
    ky = np.reshape(ky, k.shape, order='C')

    # Mass transfer functions:
    # Receiver bandwidth and acquisition time per line
    Tk = km_profiles*1.0/(2.0*receiver_bandwidth)

    # Parameters
    df_off = 300        # off-resonance frequency
    dt_esp = Tk         # temporal echo spacing
    ETL = 11            # echo train length

    # Spatial shift
    dy_off = df_off*dt_esp*ETL*delta    # top-down
    # dy_off = 2*df_off*dt_esp*ETL*delta    # center-out

    # Acquisition times
    # TODO: redefine the acquisition times to consider
    # multishot EPI
    t = acquisition_times(k, Tk, fast_imaging_mode='EPI')
    MTF_decay = 1#np.exp(-t/T2star)

    # Truncation
    MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky))
    MTF = MTF_decay*MTF_off

    # from PyMRStrain.MPIUtilities import MPI_rank
    # if MPI_rank == 0:
    #     fig, ax = plt.subplots(2,3)
    #     ax[0,0].imshow(t.T)
    #     ax[0,1].imshow(np.abs(ktoi(k)).T)
    #     ax[0,2].imshow(np.abs(ktoi(MTF*k)).T)
    #     ax[1,1].imshow(np.abs(k).T)
    #     ax[1,2].imshow(np.abs(MTF*k).T)
    #     plt.show()

    return MTF*k


# Generates acquisition times depending of the
# acquisition sequence and imaging mode
def acquisition_times(kspace,Tk,fast_imaging_mode='EPI', multishot=False):

    # kspace of the input image
    km_profiles = kspace.shape[0]
    kp_profiles = kspace.shape[1]

    # Output times

    # Acquisition times for different cartesian techniques:
    # EPI
    if fast_imaging_mode=='EPI':
        t = np.linspace(0,Tk*kp_profiles,km_profiles*kp_profiles)
        t = t.reshape(kspace.shape,order='F')
        for i in range(t.shape[1]):
            if i % 2 == 0:
                t[:,i] = t[:,i]
            else:
                t[:,i] = np.flip(t[:,i])

    # Simpler GRE
    if fast_imaging_mode=='None':
        delta = 0.002
        t = np.linspace(0,Tk*kp_profiles,km_profiles*kp_profiles)
        t = t.reshape(kspace.shape,order='F')
        for i in range(t.shape[1]):
            t[:,i] += i*Tk + delta

    return t


# Transforms the kspace from the acquisition size (acq_matrix)
# to the corresponding resolution
def acq_to_res(k, acq_matrix, resolution, dir=[0,1]):

    # FIRST PART:
    # Correct the kspace for discrepancies in the
    # phase direction.

    # Check phase direction
    zero_fill = resolution[dir[1]] > acq_matrix[dir[1]]

    # Number of additional lines
    n_lines = np.abs(resolution[dir[1]] - acq_matrix[dir[1]])

    # Adapt kspace using the phase-corrected shape
    pshape = acq_matrix
    if zero_fill:
        pshape[dir[1]] += n_lines
        k_new = np.zeros(pshape).flatten('F')
        k_new[int(n_lines/2*acq_matrix[dir[0]]):-int(n_lines/2*acq_matrix[dir[0]])] = k.flatten()
        k_new = np.reshape(k_new, pshape, order='F')
    else:
        pshape[dir[1]] -= n_lines
        k_new = np.copy(k).flatten('F')
        k_new = k_new[int(n_lines/2*acq_matrix[dir[0]]):-int(n_lines/2*acq_matrix[dir[0]])]
        k_new = np.reshape(k_new, pshape, order='F')

    return k_new