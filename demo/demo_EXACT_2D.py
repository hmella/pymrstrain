from PyMRStrain import *
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__=="__main__":

   # Parameters
  p = Parameters(time_steps=18)
  p.R_en = 0.02
  p.R_ep = 0.03
  p.tau = p.R_ep - p.R_en
  p.h = 0.008
  p.phi_en = -20*np.pi/180
  p.phi_ep = -10*np.pi/180
  p.xi = 0.5
  save_pyobject(p, 'p.pkl')
  p=load_pyobject('p.pkl')

  # Create complimentary image
  I = EXACTImage(FOV=np.array([0.2, 0.2, 0.008]),
            center=np.array([0.0,0.0,0.03]),
            resolution=np.array([100, 100, 1]),
            encoding_frequency=np.array([500.0,500.0,0.0]),
            kspace_factor=15,
            slice_thickness=0.008,
            oversampling_factor=1,
            phase_profiles=100)

  # Spins
  spins = Spins(Nb_samples=100000, parameters=p)

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, z_motion=False, write_vtk=False)

  # Generate images
  kspace, mask = I.generate(None, phantom, p)

  # Image
  I = kspace.to_img()

  # Plot
  if MPI_rank==0:
      multi_slice_viewer(np.abs(I[:,:,0,0,:]))
      multi_slice_viewer(np.angle(I[:,:,0,0,:]))