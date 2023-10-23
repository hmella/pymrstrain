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

  # Apply the inverse Fourier transform to obtain the image
  I = ktoi(K[::2,::-1,...],[0,1,2])

  # Get mask
  mask = I > 0.1

  # Add complex noise 
  I = add_cpx_noise(I, relative_std=0.02, mask=1)

  # Show figure
  for fr in range(K.shape[-1]):
    print(fr)
    for i in [2]:
      # multi_slice_viewer(np.abs(K[::2,:,:,i,fr]))
      # multi_slice_viewer(np.abs(I[:,:,:,i,fr]))
      multi_slice_viewer(np.angle(I[:,:,:,i,fr]),caxis=[-np.pi,np.pi])
      plt.show()
