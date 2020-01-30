from PyMRStrain.Helpers import order
from PyMRStrain.Math import itok, ktoi
import numpy as np
import matplotlib.pyplot as plt

def EPI_kspace(k, delta, acq_matrix, dir=[0,1],
                  rec_bw=64*1000, T2star=0.02):

    # kspace bandwith
    k_max = 0.5/delta

    # kspace of the input image
    km_profiles = acq_matrix[dir[0]]
    kp_profiles = acq_matrix[dir[1]]
    ky = np.linspace(-k_max,k_max,kp_profiles)
    for i in range(1,km_profiles):
        ky = np.append(ky, ky[0:kp_profiles])

    # Parameters
    df_off = 100                          # off-resonance frequency
    dt_esp = km_profiles*1.0/(2.0*rec_bw) # temporal echo spacing
    ETL = 1                              # echo train length

    # Spatial shifts
    dy_off = df_off*dt_esp*ETL*delta    # top-down
    # dy_off = 2*df_off*dt_esp*ETL*delta    # center-out

    # Acquisition times
    # TODO: redefine the acquisition times to consider
    # multishot EPI
    # t = time_map(k,acq_matrix,dt_esp,dir=dir,ETL=ETL)
    MTF_decay = 1#np.exp(-t/T2star)

    # Truncation
    MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky))
    MTF = MTF_decay*MTF_off

    # out = np.reshape(ky,acq_matrix,order=order[dir[1]])
    # plt.imshow(np.real(out))
    # plt.show()
    # fig, ax = plt.subplots(2,3)
    # ax[0,0].imshow(t.T)
    # ax[0,1].imshow(np.abs(ktoi(k)).T)
    # ax[0,2].imshow(np.abs(ktoi(MTF*k)).T)
    # ax[1,1].imshow(np.abs(k).T)
    # ax[1,2].imshow(np.abs(MTF*k).T)
    # plt.show()

    return MTF*k

# Generates acquisition times depending of the
# acquisition sequence and imaging mode
def time_map(k,acq_matrix,dt_esp,dir=[0,1],ETL=1):

    # kspace of the input image
    km_profiles = acq_matrix[dir[0]]
    kp_profiles = acq_matrix[dir[1]]

    # Acquisition times for different cartesian techniques:
    # EPI
    t = np.linspace(0,dt_esp*kp_profiles,km_profiles*kp_profiles)
    for i in range(kp_profiles):
        if i % 2 != 0:
            a, b = i*km_profiles, (i+1)*km_profiles
            t[a:b] = np.flip(t[a:b])

    t = t.reshape(acq_matrix,order=order[dir[0]])
    # Simpler GRE
    # delta = 0.002
    # t = np.linspace(0,dt_esp*kp_profiles,km_profiles*kp_profiles)
    # t = t.reshape(k.shape,order='F')
    # for i in range(t.shape[1]):
    #     t[:,i] += i*dt_esp + delta

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
    pshape = np.copy(acq_matrix)
    idx = int(n_lines/2*acq_matrix[dir[0]])
    if zero_fill:
        pshape[dir[1]] += n_lines
        k_meas = np.copy(k).flatten(order[dir[0]])[idx:-idx]
    else:
        pshape[dir[1]] -= n_lines
        k_meas = np.copy(k).flatten(order[dir[0]])[idx:-idx]

    # TODO: add EPI artifacts
    k_meas = EPI_kspace(k_meas, 0.003, acq_matrix, dir=dir, rec_bw=128*1000,
                         T2star=0.02)

    # Fill final kspace
    if zero_fill:
        k_new = np.zeros(pshape, dtype=complex).flatten(order[dir[0]])
        k_new[idx:-idx] = k_meas
        k_new = np.reshape(k_new, pshape, order=order[dir[0]])
    else:
        k_new = k_meas.reshape(pshape, order=order[dir[0]])

    # plt.imshow(np.abs(ktoi(k_new)).T)
    # plt.show()

    return k_new
