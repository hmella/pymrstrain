import matplotlib.pyplot as plt
import numpy as np

from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Plotter import multi_slice_viewer

if __name__ == '__main__':

  # Import generated data
  K = np.load('kspace_test.npy')

  # Generate image
  I = ktoi(K[::2,::-1,:],[0,1,2])

  # Show figure
  multi_slice_viewer(np.abs(K[::2,:,:]))
  multi_slice_viewer(np.abs(I[:,:,:]))
  multi_slice_viewer(np.angle(I[:,:,:]))
  plt.show()