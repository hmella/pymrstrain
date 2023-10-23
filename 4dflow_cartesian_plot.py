import matplotlib.pyplot as plt
import numpy as np

from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Plotter import multi_slice_viewer
from PyMRStrain.Noise import add_cpx_noise

if __name__ == '__main__':

  # Import generated data
  im = 'FFE'
  K = np.load('MRImages/HCR45/{:s}.npy'.format(im))

  # Fix the direction of kspace lines measured in the opposite direction
  if im == 'EPI':
    K[:,1::2,...] = K[::-1,1::2,...]

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
    print(fr)
    for i in [2]:
      # multi_slice_viewer(np.abs(K_fil[::2,:,:,i,fr]))
      # multi_slice_viewer(np.abs(I[:,:,:,i,fr]))
      multi_slice_viewer(np.angle(I[:,:,:,i,fr]),caxis=[-np.pi,np.pi])
      plt.show()
