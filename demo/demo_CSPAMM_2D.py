import time

import numpy as np
from PyMRStrain import *

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


  # Field inhomogeneity
  phi = lambda X, Y: 0*(X+Y)/0.1*0.2

  # Encoding frequency
  ke = 0.07               # encoding frequency [cycles/mm]
  ke = 1000*2*np.pi*ke    # encoding frequency [rad/m]

  # Create complimentary image
  I = CSPAMMImage(FOV=np.array([0.2, 0.2, 0.008]),
            center=np.array([0.0,0.0,0.0]),
            resolution=np.array([100, 100, 1]),
            encoding_frequency=np.array([ke,ke,0]),
            T1=0.85,
            flip_angle=15*np.pi/180,
            encoding_angle=90*np.pi/180,
            off_resonance=phi,
            kspace_factor=15,
            slice_thickness=0.008,
            oversampling_factor=2,
            phase_profiles=66)

  # Spins
  spins = Spins(Nb_samples=500000, parameters=p)

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, z_motion=False, write_vtk=False)

  # EPI acquisiton object
  epi = EPI(receiver_bw=128*KILO,
            echo_train_length=11,
            off_resonance=200,
            acq_matrix=I.acq_matrix,
            spatial_shift='top-down')

  # Generate images
  start = time.time()
  NSA_1, NSA_2, mask = I.generate(epi, phantom, p, debug=True)
  end = time.time()
  print(end-start)

  # Add noise to images
  sigma = 0.025
  NSA_1.k = add_cpx_noise(NSA_1.k, mask=NSA_1.k_msk, sigma=sigma)
  NSA_2.k = add_cpx_noise(NSA_2.k, mask=NSA_2.k_msk, sigma=sigma)

  # kspace to image
  In1 = NSA_1.to_img()
  In2 = NSA_2.to_img() 
  mask = mask.to_img()

  # Corrected image
  I = In1 - In2

  # Plot
  if MPI_rank==0:
      multi_slice_viewer(np.abs(I[:,:,0,0,:]))
      multi_slice_viewer(np.abs(I[:,:,0,1,:]))
      multi_slice_viewer(np.abs(I[:,:,0,1,:])*np.abs(I[:,:,0,0,:]))
      multi_slice_viewer(np.abs(NSA_1.k[:,:,0,0,:]))
      multi_slice_viewer(np.abs(NSA_1.k[:,:,0,1,:]))
