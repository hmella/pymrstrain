from PyMRStrain.Helpers import order, build_idx
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import MPI_print, MPI_rank
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal


# Transforms the kspace from the acquisition size (acq_matrix)
# to the corresponding resolution
def acq_to_res(k, acq_matrix, resolution, delta, epi=None, dir=[0,1],
    oversampling=2):

    # FIRST PART:
    # Correct the kspace for discrepancies in the
    # phase direction.

    # Number of additional lines
    n_lines = resolution[dir[1]] - acq_matrix[dir[1]]

    # Adapt kspace using the phase-corrected shape
    idx = build_idx(n_lines, acq_matrix, dir)
    k_meas = np.copy(k).flatten(order[dir[0]])[idx[0]:idx[1]]

    # Add EPI artifacts
    if epi != None:
        k_meas = epi.kspace(k_meas, delta, dir, T2star=0.02)

    # Hamming filter to reduce Gibbs ringing artifacts
    H0 = signal.hamming(acq_matrix[dir[0]])
    H1 = signal.hamming(acq_matrix[dir[1]])
    H = np.outer(H0, H1).flatten('F')
    k_meas = H*k_meas

    # Fill final kspace
    pshape = np.copy(acq_matrix)
    pshape[dir[1]] += n_lines
    k_new = np.zeros(pshape, dtype=np.complex64).flatten(order[dir[0]])
    k_new[idx[0]:idx[1]] = k_meas

    # Remove oversampled values
    pshape[dir[0]] /= oversampling
    k_new = np.reshape(k_new[::oversampling], pshape, order=order[dir[0]])

    return k_new


# EPI class for the generation of EPI-like artifacts
class EPI:
    def __init__(self, receiver_bw = 64*1000,
                 echo_train_length = 1,
                 off_resonance = 100,
                 acq_matrix = [128,64],
                 spatial_shift = 'top-down'):
        self.receiver_bw = receiver_bw
        self.echo_train_length = echo_train_length
        self.off_resonance = off_resonance
        self.acq_matrix = acq_matrix
        self.spatial_shift = spatial_shift
        self.temporal_echo_spacing = acq_matrix[1]*1.0/(2.0*self.receiver_bw)

    # Get kspace with EPI-like artifacts
    def kspace(self,k,delta,dir,T2star):
        # kspace bandwith
        k_max = 0.5/delta

        # kspace of the input image
        m_profiles = self.acq_matrix[dir[0]]
        ph_profiles = self.acq_matrix[dir[1]]
        grid = np.meshgrid(np.linspace(-k_max,k_max,m_profiles),
                            np.linspace(-k_max,k_max,ph_profiles),
                            indexing='ij')
        ky = grid[dir[1]].flatten(order[dir[0]])

        # Parameters
        df_off = self.off_resonance            # off-resonance frequency
        dt_esp = self.temporal_echo_spacing    # temporal echo spacing
        ETL = self.echo_train_length           # echo train length

        # Spatial shifts
        if self.spatial_shift == 'top-down':
            dy_off = df_off*dt_esp*self.echo_train_length*delta
        elif self.spatial_shift == 'center-out':
            dy_off = 2*df_off*dt_esp*self.echo_train_length*delta

        # Acquisition times
        t = self.time_map(k,dir=dir)

        # Truncation
        MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky),order=order[dir[0]])
        # MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky)*t,order=order[dir[0]])
        # MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky)*t/T2star,order=order[dir[0]])
        MTF = MTF_off

        return MTF*k

    # Time maps of the EPI acquisition
    def time_map(self, k, dir):

        # kspace of the input image
        m_profiles = self.acq_matrix[dir[0]]
        ph_profiles = self.acq_matrix[dir[1]]

        # Acquisition times for different cartesian techniques:
        # EPI
        t_train = np.linspace(0,self.temporal_echo_spacing*self.echo_train_length,
                            m_profiles*self.echo_train_length)
        for i in range(self.echo_train_length):
            if (i % 2 != 0):
                a, b = i*m_profiles, (i+1)*m_profiles
                t_train[a:b] = np.flip(t_train[a:b])

        t = np.copy(t_train)
        for i in range(int(ph_profiles/self.echo_train_length)-1):
            t = np.append(t, t_train, axis=0)

        # Simpler GRE
        # delta = 0.002
        # t = np.linspace(0,dt_esp*ph_profiles,m_profiles*ph_profiles)
        # t = t.reshape(k.shape,order='F')
        # for i in range(t.shape[1]):
        #     t[:,i] += i*dt_esp + delta

        return t