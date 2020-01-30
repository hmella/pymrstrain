from PyMRStrain.Helpers import order
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import MPI_print, MPI_rank
import numpy as np
import matplotlib.pyplot as plt


# Transforms the kspace from the acquisition size (acq_matrix)
# to the corresponding resolution
def acq_to_res(k, acq_matrix, resolution, delta, dir=[0,1]):

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
    # MPI_print('TODO: arreglar el tamaño del pixel en la dirección de fase')
    # MPI_print(delta)
    k_meas = EPI_kspace(k_meas, delta, acq_matrix, dir=dir, rec_bw=128*1000,
                         T2star=0.02)

    # Fill final kspace
    if zero_fill:
        k_new = np.zeros(pshape, dtype=complex).flatten(order[dir[0]])
        k_new[idx:-idx] = k_meas
        k_new = np.reshape(k_new, pshape, order=order[dir[0]])
    else:
        k_new = k_meas.reshape(pshape, order=order[dir[0]])

    return k_new


def EPI_kspace(k, delta, acq_matrix, dir=[0,1],
                  rec_bw=64*1000, T2star=0.02):

    # kspace bandwith
    k_max = 0.5/delta

    # kspace of the input image
    km_profiles = acq_matrix[dir[0]]
    kp_profiles = acq_matrix[dir[1]]
    grid = np.meshgrid(np.linspace(-k_max,k_max,km_profiles),
                         np.linspace(-k_max,k_max,kp_profiles),
                         indexing='ij')
    ky = grid[dir[1]].flatten(order[dir[0]])

    # Parameters
    df_off = 200                          # off-resonance frequency
    dt_esp = km_profiles*1.0/(2.0*rec_bw) # temporal echo spacing
    ETL = 11                              # echo train length

    # Spatial shifts
    dy_off = df_off*dt_esp*ETL*delta    # top-down
    # dy_off = 2*df_off*dt_esp*ETL*delta    # center-out

    # Acquisition times
    # TODO: redefine the acquisition times to consider
    # multishot EPI
    t = time_map(k,acq_matrix,dt_esp,dir=dir,ETL=ETL)
    MTF_decay = 1#np.exp(-t/T2star)

    # Truncation
    # MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky),order=order[dir[0]])
    # MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky)*t,order=order[dir[0]])
    MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky)*t/T2star,order=order[dir[0]])
    MTF = MTF_decay*MTF_off

    return MTF*k


# Generates acquisition times depending of the
# acquisition sequence and imaging mode
def time_map(k,acq_matrix,dt_esp,dir=[0,1],ETL=1):

    # kspace of the input image
    km_profiles = acq_matrix[dir[0]]
    kp_profiles = acq_matrix[dir[1]]

    # Acquisition times for different cartesian techniques:
    # EPI
    t_train = np.linspace(0,dt_esp*ETL,km_profiles*ETL)
    for i in range(ETL):
        if (i % 2 != 0):
            a, b = i*km_profiles, (i+1)*km_profiles
            t_train[a:b] = np.flip(t_train[a:b])

    t = np.copy(t_train)
    for i in range(int(kp_profiles/ETL)-1):
        t = np.append(t, t_train, axis=0)

    # Simpler GRE
    # delta = 0.002
    # t = np.linspace(0,dt_esp*kp_profiles,km_profiles*kp_profiles)
    # t = t.reshape(k.shape,order='F')
    # for i in range(t.shape[1]):
    #     t[:,i] += i*dt_esp + delta

    return t