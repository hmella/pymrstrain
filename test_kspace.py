import matplotlib.pyplot as plt
import numpy as np
from sigpy.fourier import nufft_adjoint, ifft
from TrajToImage import TrajToImage

from PyMRStrain.KSpaceTraj import Cartesian, Radial, Spiral
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import gather_image, MPI_rank
from PyMRStrain.Parameters import Parameters
from PyMRStrain.Phantom import Phantom
from PyMRStrain.Spins import Spins

def nufft_recon(traj, K):
  # Non-uniform fft
  x0 = traj.points[0].flatten().reshape((-1,1))
  x1 = traj.points[1].flatten().reshape((-1,1))
  x0 *= res[0]//2/x0.max()
  x1 *= res[1]//2/x1.max()
  dcf = (x0**2 + x1**2)**0.5              # density compensation function
  kxky = np.concatenate((x0,x1),axis=1)   # flattened kspace coordinates
  y = K.flatten().reshape((-1,1))         # flattened kspace measures
  image = nufft_adjoint(y*dcf, kxky, res) # inverse nufft
  return image

def nufft_cart_recon(traj, K):
  # Non-uniform fft
  x0 = traj.points[0].flatten().reshape((-1,1))
  x1 = traj.points[1].flatten().reshape((-1,1))
  x0 *= res[0]//2/x0.max()
  x1 *= res[1]//2/x1.max()
  dcf = 1#(x0**2 + x1**2)**0.5              # density compensation function
  kxky = np.concatenate((x0,x1),axis=1)   # flattened kspace coordinates
  y = K.flatten().reshape((-1,1))         # flattened kspace measures
  image = nufft_adjoint(y*dcf, kxky, res) # inverse nufft
  return image

if __name__ == '__main__':

  # Phantom parameters
  p = Parameters(time_steps=18)
  p.h = 0.008
  p.phi_en = -15*np.pi/180
  p.phi_ep = 0*np.pi/180

  # Spins
  spins = Spins(Nb_samples=300000, parameters=p)

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, z_motion=False, write_vtk=False)

  # Imaging parameters
  FOV = np.array([0.3, 0.3], dtype=np.float64)
  res = np.array([128, 128], dtype=np.int64)

  # Phantom
  r = spins.samples
  Mxy = np.zeros([3, r.shape[0]],dtype=np.float64)
  R = np.sqrt(np.power(r[:,0]-0.4*(p.R_en+p.R_ep),2) + np.power(r[:,1]-0.4*(p.R_en+p.R_ep),2))
  s = (p.R_ep-p.R_en)
  Mxy[0,:] = (1-0.85*np.exp(-np.power(R/s,2)))*np.imag(np.exp(1j*50*(r[:,0]+r[:,1])))
  Mxy[1,:] = (1-0.85*np.exp(-np.power(R/s,2)))*np.real(np.exp(1j*50*(r[:,0]+r[:,1])))

  # Cartesian kspace
  traj_c = Cartesian(FOV=FOV, res=res, oversampling=2, lines_per_shot=9)
  K_c = gather_image(TrajToImage(traj_c.points, 1000.0*traj_c.times, Mxy, r, 50.0))
  K_c[:,1::2] = K_c[::-1,1::2]
  I_cartesian = ktoi(K_c[::2,::-1])
  if MPI_rank==0:
    traj_c.plot_trajectory()
  # I_cartesian = nufft_cart_recon(traj_c, K_c)

  # Non-cartesian trajectory
  trajectory = 'spiral'
  if trajectory == 'cartesian':
    traj = Cartesian(FOV=FOV, res=res, oversampling=2, lines_per_shot=9)
  elif trajectory == 'radial':
    traj = Radial(FOV=FOV, res=res, oversampling=2, lines_per_shot=9, spokes=32)
  elif trajectory == 'spiral':
    parameters = {'k0': 0.1, 'k1': 1.0, 'Slew-rate': 45.0, 'gamma': 63.87e+6}
    traj = Spiral(FOV=FOV, res=res, oversampling=2, lines_per_shot=2, interleaves=10, parameters=parameters)
  if MPI_rank==0:
    traj.plot_trajectory()

  # Non-cartesian kspace
  K = gather_image(TrajToImage(traj.points, 1000.0*traj.times, Mxy, r, 50.0))

  # Non-uniform fft
  I_noncartesian = nufft_recon(traj, K)

  # Show results
  if MPI_rank==0:
    fig, axs = plt.subplots(2, 3)
    axs[0,0].imshow(np.abs(K_c[::2,:]),origin="lower")
    axs[0,1].imshow(np.abs(I_cartesian),origin="lower")
    axs[0,2].imshow(np.angle(I_cartesian),origin="lower")

    axs[1,0].imshow(np.abs(itok(I_noncartesian)),origin="lower")
    axs[1,1].imshow(np.abs(I_noncartesian),origin="lower")
    axs[1,2].imshow(np.angle(I_noncartesian),origin="lower")
    plt.show()
