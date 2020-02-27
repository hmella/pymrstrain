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

  # Encoding frequency
  wavelength = [0.0039,0.0078,0.0156]
  ke = 2*np.pi/wavelength[1]
  ke = np.array([ke,ke,0])

  # T1 decay
  decay = 0.5
  T1 = -1.0/np.log(decay)

  # Create complimentary image
  I = SINEImage(FOV=np.array([0.2, 0.2, 0.008]),
            center=np.array([0.0,0.0,0.0]),
            resolution=np.array([200, 200, 1]),
            T1=T1,
            encoding_frequency=ke,
            kspace_factor=15,
            slice_thickness=0.008,
            oversampling_factor=1,
            phase_profiles=200,
            filter='Riesz',
            filter_width=0.8,
            filter_lift=0.3)

  # Spins
  spins = Spins(Nb_samples=2000000, parameters=p)

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, z_motion=False, write_vtk=False)

  # Generate images
  kspace, mask = I.generate(None, phantom, p)

  # Add noise
  kspace.k = add_cpx_noise(kspace.k, mask=kspace.k_msk, sigma=0.06)

  # Image
  I = kspace.to_img()
  m = mask.to_img()

  # Plot
  if MPI_rank==0:
      multi_slice_viewer(np.abs(I[:,:,0,0,:]))
      multi_slice_viewer(np.abs(I[:,:,0,1,:]))
      multi_slice_viewer(np.abs(m[:,:,0,0,:]))
      multi_slice_viewer(np.abs(m[:,:,0,1,:]))
      multi_slice_viewer(np.abs(kspace.k[:,:,0,0,:]))
      multi_slice_viewer(np.abs(kspace.k[:,:,0,1,:]))