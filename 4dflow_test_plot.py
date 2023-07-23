import matplotlib.pyplot as plt
import numpy as np

from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Plotter import multi_slice_viewer
from PyMRStrain.Noise import add_cpx_noise

if __name__ == '__main__':

  # Import generated data
  K = np.load('kspace_test.npy')

  # Generate image
  I = ktoi(K[::2,::-1,:,:],[0,1,2])

  # Add noise 
  # I = add_cpx_noise(I, relative_std=0.05, mask=1)

  # Show figure
  multi_slice_viewer(np.abs(K[::2,:,:,7]))
  multi_slice_viewer(np.abs(I[:,:,:,7]))
  multi_slice_viewer(np.angle(I[:,:,:,7]),caxis=[-np.pi,np.pi])
  plt.show()