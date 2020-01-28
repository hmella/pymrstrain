from PyMRStrain import *
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__=="__main__":

  # Parameters
  p = Parameters(decimals=10, time_steps=18)
#   np.save("p.npy", p)
  # p=np.load('p.npy',allow_pickle=True).item()
  # p['time_steps'] = 2

  # Field inhomogeneity
  phi = lambda X, Y: 0*(X+Y)/0.1*0.2

  # Create complimentary image
  ke = 0.12               # encoding frequency [cycles/mm]
  ke = 1000*2*np.pi*ke    # encoding frequency [rad/m]
  N = 33                  # resolution
  I0 = DENSEImage(FOV=np.array([0.1, 0.1, 0.04]),
            center=np.array([0.0,0.0,-0.03]),
            resolution=np.array([N, N, 1]),
            encoding_frequency=np.array([ke,ke,0]),
            T1=0.85,
            flip_angle=15*np.pi/180,
            off_resonance=phi,
            kspace_factor=5,
            slice_following=True,
            slice_thickness=0.015)

  # Spins
  p['h'] = 0.1
  p['center'] = np.array([0,0,0])
#   p['R_inner'] = p['R_en']
#   p['R_outer'] = p['R_ep']
  spins = Spins(Nb_samples=250000, parameters=p, structured=(False, 20))

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, write_vtk=False)

  # Generator
  g0 = Generator(p, I0, debug=True)

  # Generation
  g0.phantom = phantom
  start = time.time()
  u0, u1, uin, mask = g0.get_image()
  end = time.time()
  print(end-start)

  # Add noise to images
  sigma = 0.5e-2
  un0 = add_noise_to_DENSE_(u0, mask, sigma=sigma)
  un1 = add_noise_to_DENSE_(u1, mask, sigma=sigma)

  # Corrected image
  u = un0 - un1

  if MPI_rank==0:
      fig, ax = plt.subplots(1,2)
      fig0 = ax[0].imshow(np.abs(u[...,0,0,6]),cmap=plt.get_cmap('gray'))
      fig1 = ax[1].imshow(np.angle(u[...,0,0,6]),cmap=plt.get_cmap('gray'))
      fig0.axes.get_xaxis().set_visible(False)
      fig0.axes.get_yaxis().set_visible(False)
      fig1.axes.get_xaxis().set_visible(False)
      fig1.axes.get_yaxis().set_visible(False)
      if I0.slice_following:
          if I0.FOV[-1] > I0.slice_thickness:
              plt.savefig('SF')
          else:
              plt.savefig('SS')
      else:
          plt.savefig('normal')
      plt.show()

  # Plot
  if MPI_rank==0:
      fig, ax = plt.subplots(1, 2)
      tracker = IndexTracker(ax, np.abs(u[:,:,0,0,:]), np.angle(u[:,:,0,0,:]))
      fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
      plt.show()
