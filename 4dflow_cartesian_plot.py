import matplotlib.pyplot as plt
import numpy as np

from PyMRStrain.Filters import Tukey_filter
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Plotter import multi_slice_viewer
from PyMRStrain.Noise import add_cpx_noise

if __name__ == '__main__':

  # Import generated data
  seq  = 'FFE'
  VENC = 250
  K = np.load('MRImages/Linear/HCR45/{:s}_V{:d}.npy'.format(seq, VENC))

  # Fix the direction of kspace lines measured in the opposite direction
  if seq == 'EPI':
    for i in range(K.shape[1]):
      if i % 5 == 0:
        K[:,i,...] = K[::-1,i,...]

  # Kspace filtering (as the scanner would do)
  h_meas = Tukey_filter(K.shape[0], width=0.9, lift=0.3)
  h_pha  = Tukey_filter(K.shape[1], width=0.9, lift=0.3)
  h = np.outer(h_meas, h_pha)
  H = np.tile(h[:,:,np.newaxis, np.newaxis, np.newaxis], (1, 1, K.shape[2], K.shape[3], K.shape[4]))
  K_fil = H*K

  # Apply the inverse Fourier transform to obtain the image
  I = ktoi(K_fil[::2,::-1,...],[0,1,2])

  # Get mask
  mask = I > 0.1

  # Add complex noise 
  I = add_cpx_noise(I, relative_std=0.02, mask=1)

  # Show figure
  for fr in range(K.shape[-1]):
    for i in [2]:
      # M = np.transpose(np.abs(I[:,:,:,i,fr]), (0,2,1))
      # P = np.transpose(np.angle(I[:,:,:,i,fr]), (0,2,1))
      M = np.transpose(np.abs(I[:,:,:,i,fr]), (0,1,2))
      P = np.transpose(np.angle(I[:,:,:,i,fr]), (0,1,2))
      multi_slice_viewer([M, P])
      plt.show()
