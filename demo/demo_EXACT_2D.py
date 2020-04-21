from PyMRStrain import *
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__=="__main__":

   # Parameters
  p = Parameters(time_steps=20)
  p.h = 0.008
  p.phi_en = -15*np.pi/180
  p.phi_ep = 0*np.pi/180
  # save_pyobject(p, 'p.pkl')
  # p=load_pyobject('p.pkl')

  # Create complimentary image
  I = EXACTImage(FOV=np.array([0.1, 0.1, 0.008]),
            center=np.array([0.0,0.0,0.0]),
            resolution=np.array([33, 33, 1]),
            encoding_frequency=np.array([200.0,200.0,0.0]),
            M0=np.array([0,0,1]),
            kspace_factor=15,
            slice_thickness=0.008,
            oversampling_factor=1,
            phase_profiles=33)

  # Spins
  spins = Spins(Nb_samples=250000, parameters=p)

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, z_motion=False, write_vtk=False)

  # Generate images
  kspace, mask = I.generate(None, phantom, p)

  # Image
  I = kspace.to_img()
  mask = np.abs(mask.to_img()) > 0.25*np.abs(mask.to_img()).max()

  # Plot
  if MPI_rank==0:
      multi_slice_viewer(mask[:,:,0,0,:])
      multi_slice_viewer(np.abs(I[:,:,0,0,:]))
      multi_slice_viewer(mask[:,:,0,0,:]*np.angle(I[:,:,0,0,:]))