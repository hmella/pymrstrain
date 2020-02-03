from PyMRStrain import *
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__=="__main__":

  # Parameters
  p = Parameters(time_steps=18)
  p['phi_en'] = 20*np.pi/180
  p['phi_ep'] = 0*np.pi/180
  # p['R_inner'] = p['R_en']
  # p['R_outer'] = p['R_ep']
  np.save("p.npy", p)
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
            slice_following=True,
            slice_thickness=0.008,
            oversampling_factor=1,
            phase_profiles=40)

  # Spins
  spins = Spins(Nb_samples=200000, parameters=p)

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, write_vtk=False)

  # EPI acquisiton object
  epi = EPI(receiver_bw=128*1000,
            echo_train_length=10,
            off_resonance=200,
            acq_matrix=I.acq_matrix,
            spatial_shift='top-down')

  # Generator
  g0 = Generator(p, I, epi, debug=True)

  # Generation
  g0.phantom = phantom
  start = time.time()
  u0, u1, uin, mask = g0.get_image()
  end = time.time()
  print(end-start)

  # Add noise to images
  sigma = 0.5e-32
  un0 = add_noise_to_DENSE_(u0, mask, sigma=sigma)
  un1 = add_noise_to_DENSE_(u1, mask, sigma=sigma)

  # Corrected image
  u = un0 - un1

  # Plot
  if MPI_rank==0:
      fig, ax = plt.subplots(1, 2)
      tracker = IndexTracker(ax, np.abs(u[:,:,0,0,:]), np.abs(u[:,:,0,1,:]))
      fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
      plt.show()

      fig, ax = plt.subplots(1, 2)
      tracker = IndexTracker(ax, np.angle(u[:,:,0,0,:]), np.angle(u[:,:,0,1,:]))
      fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
      plt.show()

      fig, ax = plt.subplots(1, 2)
      tracker = IndexTracker(ax, np.abs(itok(u[:,:,0,0,:])),
                             np.abs(itok(u[:,:,0,1,:])),
                             vrange=[0, 1e+03])
      fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
      plt.show()
