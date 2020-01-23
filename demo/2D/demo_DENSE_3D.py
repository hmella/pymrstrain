from PyMRStrain import *
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__=="__main__":

  # Parameters
  # p = Parameters_2D(decimals=10, time_steps=18)
  # np.save("p.npy", p)
  p=np.load('p.npy',allow_pickle=True).item()
  p["mesh_resolution"] = 0.0025

  # Field inhomogeneity
  phi = lambda X, Y: 0*(X+Y)/0.1*0.2

  # Create complimentary image
  ke = 0.12               # encoding frequency [cycles/mm]
  ke = 1000*2*np.pi*ke    # encoding frequency [rad/m]
  N = 33                  # resolution
  I0 = DENSEImage(FOV=np.array([0.1, 0.1, 0.008]),
            center=np.array([0.0,0.0,0.01]),
            resolution=np.array([N, N, 1]),
            encoding_frequency=np.array([ke,ke,0]),
            T1=0.85,
            flip_angle=15*np.pi/180,
            off_resonance=phi,
            kspace_factor=15,
            slice_following=True)

  # Generator
  g0 = Generator(p, I0)

  # Element
  FE = VectorElement("tetrahedron")

  # Mesh and fem space
  p['h'] = 0.05
  # mesh = Mesh('mesh/mesh.msh')
  mesh = fem_ventricle_geometry(p, filename='mesh/mesh.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = DefaultPhantom(p, function_space=V,
                           patient=False, write_vtk=False)

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

  fig, ax = plt.subplots(1,2)
  fig0 = ax[0].imshow(np.abs(u[...,0,0,6]),cmap=plt.get_cmap('gray'))
  fig1 = ax[1].imshow(np.angle(u[...,0,0,6]),cmap=plt.get_cmap('gray'))
  fig0.axes.get_xaxis().set_visible(False)
  fig0.axes.get_yaxis().set_visible(False)
  fig1.axes.get_xaxis().set_visible(False)
  fig1.axes.get_yaxis().set_visible(False)
  if I0.slice_following:
      plt.savefig('SF')
  else:
      plt.savefig('normal')
  plt.show()

  # Plot
  if rank==0:
      # fig, ax = plt.subplots(1, 2)
      # # tracker = IndexTracker(ax, np.abs(u[:,:,0,0,:]), np.abs(u[:,:,1,0,:]))
      # tracker = IndexTracker(ax, np.abs(u[16,:,:,0,:]), np.abs(u[16,:,:,0,:]))
      # fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
      # plt.show()

      fig, ax = plt.subplots(1, 2)
      tracker = IndexTracker(ax, np.abs(u[:,:,0,0,:]), np.angle(u[:,:,0,0,:]))
      fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
      plt.show()
