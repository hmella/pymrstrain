from PyMRStrain import *
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__=="__main__":

  # Parameters
  # p = Parameters(time_steps=18)
  # p['phi_en'] = 20*np.pi/180
  # p['phi_ep'] = 0*np.pi/180
  # # p['R_inner'] = p['R_en']
  # # p['R_outer'] = p['R_ep']
  # np.save("p.npy", p)
  p=np.load('p.npy',allow_pickle=True).item()

  # Field inhomogeneity
  phi = lambda X, Y: 0*(X+Y)/0.1*0.2

  # Encoding frequency
  ke = 0.12               # encoding frequency [cycles/mm]
  ke = 1000*2*np.pi*ke    # encoding frequency [rad/m]

  # Create complimentary image
  I = DENSEImage(FOV=np.array([0.2, 0.2, 0.04]),
            center=np.array([0.0,0.0,0.0]),
            resolution=np.array([70, 70, 1]),
            encoding_frequency=np.array([ke,ke,0]),
            T1=0.85,
            flip_angle=15*np.pi/180,
            off_resonance=phi,
            kspace_factor=15,
            slice_thickness=0.008,
            oversampling_factor=1,
            phase_profiles=40)

  # Spins
  spins = Spins(Nb_samples=100000, parameters=p)

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, write_vtk=False)

  # EPI acquisiton object
  epi = EPI(receiver_bw=128*1000,
            echo_train_length=10,
            off_resonance=200,
            acq_matrix=I.acq_matrix,
            spatial_shift='top-down')

  # Generate images
  start = time.time()
  kspace_0, kspace_1, kspace_in, mask = I.generate(epi, phantom, p, True)
  end = time.time()
  print(end-start)

  # Add noise to images
  sigma = 0.025*1e-32
  kspace_0.k = add_cpx_noise(kspace_0.k, mask=kspace_0.k_msk, sigma=sigma)
  kspace_1.k = add_cpx_noise(kspace_1.k, mask=kspace_1.k_msk, sigma=sigma)

  # kspace to image
  un0 = kspace_0.to_img()
  un1 = kspace_1.to_img() 
  unin = kspace_in.to_img()

  # Corrected image
  u = un0 - un1

  # Plot
  if MPI_rank==0:
      fig, ax = plt.subplots(1, 2)
      tracker = IndexTracker(ax, np.abs(u[:,:,0,0,:]), np.angle(u[:,:,0,0,:]))
      fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
      plt.show()

      fig, ax = plt.subplots(1, 2)
      tracker = IndexTracker(ax, np.abs(kspace_0.k[:,:,0,0,:]),
                             np.abs(kspace_0.k[:,:,0,1,:]))
      fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
      plt.show()