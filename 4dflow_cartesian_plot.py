import matplotlib.pyplot as plt
import numpy as np

from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Plotter import multi_slice_viewer
from PyMRStrain.Noise import add_cpx_noise

if __name__ == '__main__':

  # Import generated data
  im = 'FFE'
  K1 = np.load('MRImages/HCR35/{:s}.npy'.format(im))
  K2 = np.load('MRImages/HCR45/{:s}.npy'.format(im))

  # Fix the direction of kspace lines measured in the opposite direction
  if im == 'EPI':
    K1[:,1::2,...] = K1[::-1,1::2,...]
    K2[:,1::2,...] = K2[::-1,1::2,...]

  # Apply the inverse Fourier transform to obtain the image
  I1 = ktoi(K1[::2,::-1,...],[0,1,2])
  I2 = ktoi(K2[::2,::-1,...],[0,1,2])

  # Get mask
  mask = I1 > 0.1

  # Calculate the difference between both images
  diff = mask*np.abs(np.angle(I1) - np.angle(I2[...,:-1]))


  # Add noise 
  # I = add_cpx_noise(I, relative_std=0.05, mask=1)

  # Show figure
  for fr in range(K1.shape[-1]):
    print(fr)
    for i in [2]:
      # multi_slice_viewer(np.abs(K[::2,:,:,i,fr]))
      # multi_slice_viewer(np.abs(I1[:,:,:,i,fr]))
      # multi_slice_viewer(np.angle(I1[:,:,:,i,fr]),caxis=[-np.pi,np.pi])
      multi_slice_viewer(diff[:,:,:,i,fr],caxis=[-0.01,0.01])
      plt.show()