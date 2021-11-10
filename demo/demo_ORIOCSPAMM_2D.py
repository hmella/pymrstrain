import time

import numpy as np
from PyMRStrain import *

if __name__=="__main__":

  # Parameters
  p = Parameters(time_steps=20)
  p.h = 0.008
  # p.phi_en = -15*np.pi/180
  # p.phi_ep = 0*np.pi/180
  save_pyobject(p, 'p.pkl')
  p=load_pyobject('p.pkl')

  # Field inhomogeneity
  phi = lambda X, Y: 0*(X+Y)/0.1*0.2

  # Encoding frequency
  lmbda = np.sqrt(2)*14        # tag period [mm]
  ke    = 2*np.pi/(lmbda/1000) # encoding frequency [rad/m]

  # Spins
  spins = Spins(Nb_samples=250000, parameters=p)

  # Variable flip angles
  last = 25*np.pi/180
  TR = 1.0/p.time_steps
  FA = np.zeros([p.time_steps],dtype=np.float)
  ramp = np.zeros(FA.shape)
  T1 = 0.9
  C  = np.exp(-TR/T1)
  FA[-1] = last
  gamma = 5*np.pi/180
  for i in range (p.time_steps-2,-1,-1):
    ramp[i]  = gamma*(p.time_steps- i-1)/(p.time_steps-1)
    FA[i]    = np.arctan(np.sin(last)*C) + ramp[i]
    last = FA[i] - ramp[i]

  print((FA-ramp)*180/np.pi)
  plt.plot(180*(FA-ramp)/np.pi)
  plt.plot(180*ramp/np.pi)
  plt.plot(180*FA/np.pi)
  plt.legend(['Fischer','rampdown','Sampath'])
  plt.show()

  # T1 map
  x = spins.samples[spins.regions[:,2],0].reshape((spins.regions[:,2].sum(),1))
  y = spins.samples[spins.regions[:,2],1].reshape((spins.regions[:,2].sum(),1))
  z = spins.samples[spins.regions[:,2],2].reshape((spins.regions[:,2].sum(),1))
  T1a = 1e-10
  T1b = 1e-10
  r = np.sqrt((x+(p.R_en+0.5*p.tau))**2 + y**2)
  s = 0.02
  T1c = 0.9 - 0.35*np.exp(-np.power(r/s, 2))

  # Create complimentary image
  I = ORI_O_CSPAMMImage(FOV=np.array([0.3, 0.3, 0.008]),
            center=np.array([0.0,0.0,0.0]),
            resolution=np.array([150, 150, 1]),
            encoding_frequency=np.array([ke,ke,0]),
            T1=[T1a,T1b,T1c],
            M0=np.array([0,0,1]),
            flip_angle=25*np.pi/180,
            encoding_angle=90*np.pi/180,
            off_resonance=phi,
            kspace_factor=2,
            slice_thickness=0.008,
            oversampling_factor=1,
            phase_profiles=88)

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
  NSA_1, NSA_2, mask, T1 = I.generate(None, phantom, p, debug=True)
  end = time.time()
  print(end-start)

  # Add noise to images
  sigma = 0.025
  # NSA_1.k = add_cpx_noise(NSA_1.k, mask=NSA_1.k_msk, sigma=sigma)
  # NSA_2.k = add_cpx_noise(NSA_2.k, mask=NSA_2.k_msk, sigma=sigma)

  # kspace to image
  In1 = NSA_1.to_img()
  In2 = NSA_2.to_img() 
  mask = mask.to_img()
  T1m  = T1.to_img()

  # CSPAMM image
  Ic = In1 - In2
  It = In1 + In2  

  # Export images
  export_image(In1, path='/home/hernan/Desktop/T1_HARPI/data/NSA_1', name='I1')  
  export_image(In2, path='/home/hernan/Desktop/T1_HARPI/data/NSA_2', name='I2')  
  export_image(mask, path='/home/hernan/Desktop/T1_HARPI/data/mask', name='mask')
  export_image(T1m, path='/home/hernan/Desktop/T1_HARPI/data/T1', name='T1_GT')  
  # Plot
  if MPI_rank==0:
      multi_slice_viewer(np.abs(NSA_1.k[:,:,0,0,:]-NSA_2.k[:,:,0,0,:]))
      multi_slice_viewer(np.abs(Ic[:,:,0,0,:]))
      multi_slice_viewer(np.abs(It[:,:,0,0,:]))
