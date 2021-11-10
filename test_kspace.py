import matplotlib.pyplot as plt
import numpy as np
from sigpy.fourier import nufft_adjoint
from TrajToImage import TrajToImage

from PyMRStrain.KSpaceTraj import Cartesian, Radial, Spiral
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Parameters import Parameters
from PyMRStrain.Phantom import Phantom
from PyMRStrain.Spins import Spins

if __name__ == '__main__':

  # Phantom parameters
  p = Parameters(time_steps=18)
  p.h = 0.008
  p.phi_en = -15*np.pi/180
  p.phi_ep = 0*np.pi/180

  # Spins
  spins = Spins(Nb_samples=10000, parameters=p)

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, z_motion=False, write_vtk=False)

  # Imaging parameters
  FOV = np.array([0.3, 0.3], dtype=np.float64)
  res = np.array([64, 64], dtype=np.int64)

  # Phantom
  r = spins.samples
  Mxy = np.zeros([3, r.shape[0]],dtype=np.float)
  Mxy[0,:] = np.real(2.0*np.exp(1j*50*r[:,0]))
  Mxy[1,:] = np.imag(2.0*np.exp(1j*50*r[:,0]))

  # Cartesian kspace
  traj_c = Cartesian(FOV=FOV, res=res, oversampling=2, lines_per_shot=9)
  K_c = TrajToImage(traj_c.points, 1000.0*traj_c.times, Mxy, r, 50.0)
  K_c[:,1::2] = K_c[::-1,1::2]

  # Non-cartesian trajectory
  trajectory = 'radial'
  if trajectory == 'cartesian':
    traj = Cartesian(FOV=FOV, res=res, oversampling=2, lines_per_shot=9)
  elif trajectory == 'radial':
    traj = Radial(FOV=FOV, res=res, oversampling=2, lines_per_shot=9, spokes=32)
  elif trajectory == 'spiral':
    traj = Spiral(FOV=FOV, res=res, oversampling=2, lines_per_shot=2, interleaves=10)
  # traj.plot_trajectory()

  # Non-cartesian kspace
  K = TrajToImage(traj.points, 1000.0*traj.times, Mxy, r, 50.0)

  # Non-uniform fft
  x0 = traj.points[0].flatten().reshape((-1,1))
  x1 = traj.points[1].flatten().reshape((-1,1))
  x0 *= res[0]//2/x0.max()
  x1 *= res[1]//2/x1.max()
  dcf = (x0**2 + x1**2)**0.5              # density compensation function
  kxky = np.concatenate((x0,x1),axis=1)   # flattened kspace coordinates
  y = K.flatten().reshape((-1,1))         # flattened kspace measures
  image = nufft_adjoint(y*dcf, kxky, res) # inverse nufft

  # Show results
  fig, axs = plt.subplots(2, 3, figsize=(10, 10))
  axs[0,0].imshow(np.abs(K_c[::2,:]))
  axs[0,1].imshow(np.abs(ktoi(K_c[::2,:])))
  axs[0,2].imshow(np.angle(ktoi(K_c[::2,:])))
  axs[1,0].imshow(np.abs(itok(image)))
  axs[1,1].imshow(np.abs(image))
  axs[1,2].imshow(np.angle(image))
  plt.show()
