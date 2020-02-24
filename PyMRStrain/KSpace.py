import matplotlib.pyplot as plt
import numpy as np
from PyMRStrain.Filters import Hamming_filter, Riesz_filter, Tukey_filter
from PyMRStrain.Helpers import build_idx, order
from PyMRStrain.IO import scale_image, rescale_image
from PyMRStrain.Math import itok, ktoi


# kspace class
class kspace:
    def __init__(self, shape, acq_matrix, oversampling_factor, artifact):
        # Output kspace shape
        self.shape = shape
        self.acq_matrix = acq_matrix
        self.oversampling_factor = oversampling_factor
        self.artifact = artifact
        self.k = np.zeros(shape,dtype=np.complex64)
        self.k_msk = np.zeros(shape,dtype=np.float32)
        self.k_acq = np.zeros(np.append(acq_matrix,shape[2:5]),dtype=np.complex64)
        self.filter = []

    def to_img(self):
        return ktoi(self.k)

    def gen_to_acq(self, k_gen, delta_ph, dir, slice, enc_dir, timestep):

        # Input kspace resolution
        resolution = k_gen.shape
        acq_matrix = self.acq_matrix[dir]

        # Number of additional lines
        n_lines = resolution[dir[1]] - acq_matrix[dir[1]]

        # Adapt kspace using the phase-corrected shape
        idx = build_idx(n_lines, acq_matrix, dir)
        k_acq = np.copy(k_gen).flatten(order[dir[0]])[idx[0]:idx[1]]

        # Add epi artifacts
        if self.artifact != None:
            k_acq = self.artifact.kspace(k_acq, delta_ph, dir, T2star=0.02)

        # Hamming filter to reduce Gibbs ringing artifacts
        # H = Hamming_filter(acq_matrix,dir)
        Hm = Riesz_filter(acq_matrix[dir[0]],width=0.8,lift=0.3)
        Hp = Riesz_filter(acq_matrix[dir[1]],width=0.8,lift=0.3)
        self.filter = np.outer(Hm,Hp).flatten('F')
        k_acq_filt = self.filter*k_acq

        # Fill final kspace
        pshape = np.copy(acq_matrix)
        pshape[dir[1]] += n_lines
        k_new = np.zeros(pshape, dtype=np.complex64).flatten(order[dir[0]])
        k_new[idx[0]:idx[1]] = k_acq_filt

        # Mask
        k_mask = np.zeros(pshape, dtype=float).flatten(order[dir[0]])
        k_mask[idx[0]:idx[1]] = self.filter

        # Remove oversampled values
        pshape[dir[0]] /= self.oversampling_factor

        # Store kspaces
        k_tmp_0 = np.reshape(k_new[::self.oversampling_factor], pshape, order=order[dir[0]])
        k_tmp_1 = np.reshape(k_acq, acq_matrix[dir], order=order[dir[enc_dir]])
        self.k[...,slice,enc_dir,timestep] = k_tmp_0
        self.k_acq[...,slice,enc_dir,timestep] = k_tmp_1

        # Store mask
        k_tmp_msk = np.reshape(k_mask[::self.oversampling_factor], pshape, order=order[dir[0]])
        self.k_msk[...,slice,enc_dir,timestep] = k_tmp_msk

    def scale(self):
        self.k = scale_image(self.k,mag=False,real=True,compl=True)
        self.k_msk = scale_image(self.k_msk,mag=False,real=True,compl=False)
        self.k_acq = scale_image(self.k_acq,mag=False,real=True,compl=True)

    def rescale(self):
        tmp = rescale_image(self.k)
        self.k = tmp['real'] + 1j*tmp['complex']
        tmp = rescale_image(self.k_msk)
        self.k_msk = tmp['real']
        tmp = rescale_image(self.k_acq)
        self.k_acq = tmp['real'] + 1j*tmp['complex']