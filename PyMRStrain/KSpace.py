import matplotlib.pyplot as plt
import numpy as np
from PyMRStrain.Filters import Hamming_filter, Riesz_filter, Tukey_filter
from PyMRStrain.Helpers import build_idx, isodd, order
from PyMRStrain.IO import rescale_image, scale_image
from PyMRStrain.Math import itok, ktoi


# kspace class
class kspace:
    def __init__(self, shape, image, artifact):
        # Output kspace shape
        self.shape = shape
        self.acq_matrix = image.acq_matrix
        self.oversampling_factor = image.oversampling_factor
        self.artifact = artifact
        self.k = np.zeros(shape,dtype=np.complex128)
        self.k_msk = np.zeros(shape,dtype=np.float32)
        self.k_acq = np.zeros(np.append(self.acq_matrix,shape[2:5]),dtype=np.complex128)
        self.filter = image.filter
        self.filter_width = image.filter_width
        self.filter_lift = image.filter_lift

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

        # Filter to reduce Gibbs ringing artifacts
        if self.filter is 'Tukey':
            Hm = Tukey_filter(acq_matrix[dir[0]],width=self.filter_width,
                              lift=self.filter_lift)
            Hp = Tukey_filter(acq_matrix[dir[1]],width=self.filter_width,
                              lift=self.filter_lift)
        if self.filter is 'Riesz':
            Hm = Riesz_filter(acq_matrix[dir[0]],width=self.filter_width,
                              lift=self.filter_lift)
            Hp = Riesz_filter(acq_matrix[dir[1]],width=self.filter_width,
                              lift=self.filter_lift)
        if self.filter is None:
            Hm = np.ones([acq_matrix[dir[0]],])
            Hp = np.ones([acq_matrix[dir[1]],])
        H = np.outer(Hm,Hp).flatten('F')
        k_acq_filt = H*k_acq

        # Fill final kspace
        pshape = np.copy(acq_matrix)
        pshape[dir[1]] += n_lines
        k_new = np.zeros(pshape, dtype=np.complex128).flatten(order[dir[0]])
        k_new[idx[0]:idx[1]] = k_acq_filt

        # Mask
        k_mask = np.zeros(pshape, dtype=float).flatten(order[dir[0]])
        k_mask[idx[0]:idx[1]] = H

        # Remove oversampled values
        pshape[dir[0]] /= self.oversampling_factor

        # Store kspaces
        start = 0
        if isodd(acq_matrix[0]/self.oversampling_factor) and self.oversampling_factor != 1:
            start = 1
        k_tmp_0 = np.reshape(k_new[start::self.oversampling_factor], pshape, order=order[dir[0]])
        k_tmp_1 = np.reshape(k_acq, acq_matrix[dir], order=order[dir[enc_dir]])
        self.k[...,slice,enc_dir,timestep] = k_tmp_0
        self.k_acq[...,slice,enc_dir,timestep] = k_tmp_1

        # Store mask
        k_tmp_msk = np.reshape(k_mask[start::self.oversampling_factor], pshape, order=order[dir[0]])
        self.k_msk[...,slice,enc_dir,timestep] = k_tmp_msk

    def scale(self,dtype=np.uint64):
        self.k = scale_image(self.k,mag=False,real=True,compl=True,dtype=dtype)
        self.k_msk = scale_image(self.k_msk,mag=False,real=True,compl=False,dtype=dtype)
        self.k_acq = scale_image(self.k_acq,mag=False,real=True,compl=True,dtype=dtype)

    def rescale(self):
        tmp = rescale_image(self.k)
        self.k = tmp['real'] + 1j*tmp['complex']
        tmp = rescale_image(self.k_msk)
        self.k_msk = tmp['real']
        tmp = rescale_image(self.k_acq)
        self.k_acq = tmp['real'] + 1j*tmp['complex']
