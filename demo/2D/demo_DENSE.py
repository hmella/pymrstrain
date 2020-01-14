from PyMRStrain import *
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__=="__main__":

  # Parameters
  p = Parameters_2D(decimals=10, time_steps=18)
  np.save("p.npy", p)
  p=np.load('p.npy',allow_pickle=True).item()
  p["mesh_resolution"] = 0.001

  # Field inhomogeneity
  phi = lambda X, Y: (X+Y)/0.1*0.2

  # Create complimentary image
  ke = 0.12               # encoding frequency [cycles/mm]
  ke = 1000*2*np.pi*ke    # encoding frequency [rad/m]
  N = 100                 # resolution
  # I0 = DENSEImage(FOV=np.array([0.1, 0.1, 0.008]),
  #           resolution=np.array([N, N, 1]),
  #           encoding_frequency=np.array([ke,ke,0]),
  #           T1=0.85,
  #           flip_angle=15*np.pi/180,
  #           off_resonance=phi,
  #           kspace_factor=16.5)
  I0 = DENSEImage(FOV=np.array([0.1, 0.1]),
            resolution=np.array([N, N]),
            encoding_frequency=np.array([ke,ke]),
            T1=0.85,
            flip_angle=15*np.pi/180,
            off_resonance=phi,
            kspace_factor=16.5)

  # Generator
  g0 = Generator(p, I0)

  # Element
  FE = VectorElement("triangle")

  # Mesh and fem space
  mesh = fem_ventricle_geometry(p['R_en'], p['tau'], p['h'], p['mesh_resolution'], filename='mesh/mesh.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = DefaultPhantom(p, function_space=V, patient=False)

  # Generation
  g0.phantom = phantom
  start = time.time()
  u0, u1, uin, mask = g0.get_image()
  end = time.time()
  print(end-start)

  # Add noise to images
  sigma = 0.2e-02
  un0 = add_noise_to_DENSE_(u0, mask, sigma=sigma)
  un1 = add_noise_to_DENSE_(u1, mask, sigma=sigma)

  # Corrected image
  u = un0 - un1

  # # Plot kspace
  # fig, ax = plt.subplots(1,2)
  # fig0 = ax[0].imshow(abs(FFT(un0[...,0,10])),cmap=plt.get_cmap('gray'),vmin=0,vmax=3.5e+01)
  # fig1 = ax[1].imshow(abs(FFT(u[...,0,10])),cmap=plt.get_cmap('gray'),vmin=0,vmax=7.0e+01)
  # fig0.axes.get_xaxis().set_visible(False)
  # fig0.axes.get_yaxis().set_visible(False)
  # fig1.axes.get_xaxis().set_visible(False)
  # fig1.axes.get_yaxis().set_visible(False)
  # plt.savefig('DENSE')
  # plt.show()

  # Plot kspace
  # fig, ax = plt.subplots(1,3)
  # fig0 = ax[0].imshow(np.angle(np.angle(uin[...,0,10])),cmap=plt.get_cmap('gray'),vmin=-np.pi,vmax=np.pi)
  # fig1 = ax[1].imshow(np.angle(u[...,0,10]),cmap=plt.get_cmap('gray'),vmin=-np.pi,vmax=np.pi)
  # fig2 = ax[2].imshow(np.angle(u[...,0,10]*(-uin[...,0,10])),cmap=plt.get_cmap('gray'),vmin=-np.pi,vmax=np.pi)
  # fig0.axes.get_xaxis().set_visible(False)
  # fig0.axes.get_yaxis().set_visible(False)
  # fig1.axes.get_xaxis().set_visible(False)
  # fig1.axes.get_yaxis().set_visible(False)
  # fig2.axes.get_xaxis().set_visible(False)
  # fig2.axes.get_yaxis().set_visible(False)
  # plt.savefig('DENSE')
  # plt.show()

  fig = plt.imshow(np.angle(u[...,0,0,7]),cmap=plt.get_cmap('gray'))
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  plt.savefig('new')
  plt.show()

  # fig = plt.imshow(np.angle(u[...,0,10]*np.conj(-uin[...,0,10])),cmap=plt.get_cmap('gray'),vmin=-np.pi,vmax=np.pi)
  # fig.axes.get_xaxis().set_visible(False)
  # fig.axes.get_yaxis().set_visible(False)
  # plt.savefig('DENSE_B0_corrected')
  # plt.show()

  # fig = plt.imshow(np.angle(u[...,0,10]),cmap=plt.get_cmap('gray'),vmin=-np.pi,vmax=np.pi)
  # fig.axes.get_xaxis().set_visible(False)
  # fig.axes.get_yaxis().set_visible(False)
  # plt.savefig('DENSE_B0_free')
  # plt.show()

  # Plot
  if rank==0:
    fig, ax = plt.subplots(1, 2)
    tracker = IndexTracker(ax, np.abs(u[:,:,0,0,:]), np.angle(u[:,:,0,0,:]))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
