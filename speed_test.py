import matplotlib.pyplot as plt
import numpy as np
from sigpy.fourier import nufft_adjoint
from TrajToImage import TrajToImage
import time

from PyMRStrain.KSpaceTraj import Cartesian, Radial, Spiral
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Parameters import Parameters
from PyMRStrain.Phantom import Phantom
from PyMRStrain.Spins import Spins
from PyMRStrain.MPIUtilities import ScatterSpins, gather_image, MPI_rank

if __name__ == '__main__':

  # Phantom parameters
  p = Parameters(time_steps=18)
  p.h = 0.008
  p.phi_en = -15*np.pi/180
  p.phi_ep = 0*np.pi/180

  # Imaging parameters
  FOV = np.array([0.3, 0.3], dtype=np.float64)
  res = np.array([32, 32], dtype=np.int64)

  # Number of spins and resolutions
  nb_spins = [1000, 10000, 50000, 100000, 200000]
  res = [16, 32, 64, 128, 256]

  # Arrays to store the runtimes
  number_of_repetitions = 1
  time_spins = np.zeros([len(nb_spins), number_of_repetitions])
  time_res = np.zeros([len(res), number_of_repetitions])

  for (i, nb_s) in enumerate(nb_spins):

    # Spins
    spins = Spins(Nb_samples=nb_s, parameters=p)

    # Create phantom object
    phantom = Phantom(spins, p, patient=False, z_motion=False, write_vtk=False)

    # Phantom
    r = 2*np.pi*spins.samples
    Mxy = np.zeros([2, r.shape[0]],dtype=np.float)
    Mxy[0,:] = np.real(2.0*np.exp(1j*50*r[:,0]))
    Mxy[1,:] = np.imag(2.0*np.exp(1j*50*r[:,0]))

    # Cartesian kspace
    traj_c = Cartesian(FOV=FOV, res=[res[1],res[1]], oversampling=1, lines_per_shot=9)
    for j in range(number_of_repetitions):
      start_time = time.time()
      K_c = TrajToImage(traj_c.points, 1000.0*traj_c.times, Mxy, r, 50.0)
      end_time = time.time()

      # Store runtime
      time_spins[i, j] = end_time - start_time

    # Store times
    if MPI_rank != 1:
      np.savetxt('times_spins_b.txt', time_spins)
    else:
      np.savetxt('times_spins.txt', time_spins)

  # # Resolution test
  # for (i, r) in enumerate(res):

  #   # Resolution
  #   res = np.array([r, r], dtype=np.int64)

  #   # Spins
  #   spins = Spins(Nb_samples=nb_spins[1], parameters=p)

  #   # Create phantom object
  #   phantom = Phantom(spins, p, patient=False, z_motion=False, write_vtk=False)

  #   # Phantom
  #   r = spins.samples
  #   Mxy = np.zeros([3, r.shape[0]],dtype=np.float)
  #   Mxy[0,:] = np.real(2.0*np.exp(1j*50*r[:,0]))
  #   Mxy[1,:] = np.imag(2.0*np.exp(1j*50*r[:,0]))

  #   # Cartesian kspace
  #   traj_c = Cartesian(FOV=FOV, res=res, oversampling=1, lines_per_shot=9)
  #   for j in range(number_of_repetitions):
  #     start_time = time.time()
  #     K_c = TrajToImage(traj_c.points, 1000*traj_c.times, Mxy, r, 50)
  #     end_time = time.time()

  #     # Store runtime
  #     time_res[i, j] = end_time - start_time

  #   # Store times
  #   np.savetxt('times_res.txt', time_res)
