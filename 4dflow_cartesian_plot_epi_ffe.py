import matplotlib.pyplot as plt
import numpy as np

from PyMRStrain.Filters import Tukey_filter
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Plotter import multi_slice_viewer
from PyMRStrain.Noise import add_cpx_noise

if __name__ == '__main__':

  # Import generated data
  VENC = 150
  Kepi = np.load('MRImages/Linear/HCR45/{:s}_V{:d}.npy'.format('EPI', VENC))
  Kffe = np.load('MRImages/Linear/HCR45/{:s}_V{:d}.npy'.format('FFE', VENC))

  # Fix the direction of kspace lines measured in the opposite direction
  for i in range(K.shape[1]):
    if i % 5 == 0:
      Kepi[:,i,...] = Kepi[::-1,i,...]

  # Kspace filtering (as the scanner would do)
  h_meas = Tukey_filter(Kepi.shape[0], width=0.9, lift=0.3)
  h_pha  = Tukey_filter(Kepi.shape[1], width=0.9, lift=0.3)
  h = np.outer(h_meas, h_pha)
  H = np.tile(h[:,:,np.newaxis, np.newaxis, np.newaxis], (1, 1, Kepi.shape[2], Kepi.shape[3], Kepi.shape[4]))
  Kepi_fil = H*Kepi
  Kffe_fil = H*Kffe

  # Apply the inverse Fourier transform to obtain the image
  Iepi = ktoi(Kepi_fil[::2,::-1,...],[0,1,2])
  Iffe = ktoi(Kffe_fil[::2,::-1,...],[0,1,2])

  # Add complex noise 
  # Iepi = add_cpx_noise(Iepi, relative_std=0.02, mask=1)
  # Iffe = add_cpx_noise(Iffe, relative_std=0.02, mask=1)

  # Show figure
  for fr in range(Kepi.shape[-1]):
    for i in [2]:
      Pepi = np.angle(Iepi[:,:,:,i,fr])
      Pffe = np.angle(Iffe[:,:,:,i,fr])
      mask = np.abs(Iffe[:,:,:,1,1]) > 0.1
      # Pepi = np.transpose(np.angle(Iepi[:,:,:,i,fr]), (0,2,1))
      # Pffe = np.transpose(np.angle(Iffe[:,:,:,i,fr]), (0,2,1))
      # mask = np.transpose(np.abs(Iffe[:,:,:,1,1]) > 0.1, (0,2,1))
      multi_slice_viewer([Pffe, Pepi, mask*np.abs(Pffe - Pepi)])
      plt.show()
