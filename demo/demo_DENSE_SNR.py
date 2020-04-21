from PyMRStrain import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *

def FFT(x):
  return fftshift(fftn(ifftshift(x)))

def iFFT(x):
  return fftshift(ifftn(ifftshift(x)))

if __name__=="__main__":

  # Parameters
  p = Parameters_2D(mesh_resolution = 0.00035)
  p['time_steps'] = 18
  np.save("p.npy", p)
  p=np.load('p.npy',allow_pickle=True).item()

  # Field inhomogeneity
  phi = lambda X, Y: 0*(X+Y)/0.1*0.2

  # Create complimentary image
  ke = 0.12               # encoding frequency [cycles/mm]
  ke = 1000*2*np.pi*ke    # encoding frequency [rad/m]
  N = 33                  # resolution
  I0 = DENSEImage(FOV=np.array([0.1, 0.1]),
            resolution=np.array([N, N]),
            encoding_frequency=np.array([ke,ke]),
            T1=np.array([1e-10,1e-10,0.85]),
            M0=np.array([0,0,1]),
            flip_angle=15*np.pi/180,
            off_resonance=phi)

  # Generator
  g0 = Generator(p, I0)

  # Element
  FE = VectorElement("triangle")

  # Mesh and fem space
  mesh = fem_ventricle_geometry(p['R_en'], p['tau'], p['mesh_resolution'], filename='mesh/mesh.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = DefaultPhantom(p, function_space=V, patient=True)

  # Generation
  g0.phantom = phantom
  u0, u1, uin, mask = g0.get_image()

  # Add noise to images
  u0[...,0,:] = mask*u0[...,0,:]
  u0[...,1,:] = mask*u0[...,1,:]
  u1[...,0,:] = mask*u1[...,0,:]
  u1[...,1,:] = mask*u1[...,1,:]
  SNR = [22, 18, 14, 10]
  for s in range(len(SNR)):
    
    un0, noise  = add_noise_to_DENSE_(u0, mask, SNR=SNR[s], recover_noise=True)
    un1  = add_noise_to_DENSE_(u1, mask, SNR=SNR[s])
    unin = add_noise_to_DENSE_(uin, mask, SNR=SNR[s])

    # Corrected image
    u = (un0 - un1)*np.conj(unin)/np.abs(unin)

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

    # fig = plt.imshow(np.angle(u[...,0,10]),cmap=plt.get_cmap('gray'),vmin=-np.pi,vmax=np.pi)
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    # plt.savefig('DENSE_B0')
    # plt.show()

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

    # SNR
    SNR0  = np.zeros([u.shape[-1],1],dtype=np.float_)
    SNR1 = np.zeros([u.shape[-1],1],dtype=np.float_)
    for i in range(u.shape[-1]):
      mu0 = np.abs(u[...,0,i])
      pos = mask[...,i].astype(np.bool_)
      mean = np.mean(mu0[pos])
      std  = np.std(mu0[~pos])
      SNR0[i]  = mean/std

      mu0 = np.abs(u[...,1,i])
      mean = np.mean(mu0[pos])
      std  = np.std(mu0[~pos])
      SNR1[i]  = mean/std

    time  = np.linspace(1,u.shape[-1],u.shape[-1])
    label = "SNR {:0.0f} to {:0.0f}".format(round(SNR0.max()),round(SNR0.min()))
    plt.plot(time,SNR0,label=label)

    # # Plot
    # if rank==0:
    #   fig, ax = plt.subplots(1, 2)
    #   tracker = IndexTracker(ax, np.abs(u[:,:,0,:]), np.angle(u[:,:,0,:]))
    #   fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    #   fig.suptitle('DENSE volunteer')
    #   plt.show()

  plt.legend()
  plt.show()