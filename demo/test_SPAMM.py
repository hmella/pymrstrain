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
  # p = Parameters_2D(mesh_resolution = 0.00025)
  p=np.load('p.npy').item()

  # Create image
  I_Tagging_x_0 = Image(FOV=np.array([0.1, 0.1]),
                  resolution=np.array([70, 70]),
                  type='Tagging',
                  encoding_frequency=[900],
                  encoding_direction=[0],
                  flip_angle = np.pi/2,
                  invert_second_RF=False,
                  T1=0.85)
  I_Tagging_x_1 = Image(FOV=np.array([0.1, 0.1]),
                  resolution=np.array([70, 70]),
                  type='Tagging',
                  encoding_frequency=[900],
                  encoding_direction=[0],
                  flip_angle = np.pi/2,
                  invert_second_RF=True,
                  T1=0.85)
  I_Tagging_y_0 = Image(FOV=np.array([0.1, 0.1]),
                  resolution=np.array([70, 70]),
                  type='Tagging',
                  encoding_frequency=[900],
                  encoding_direction=[1],
                  flip_angle = np.pi/2,
                  invert_second_RF=False,
                  T1=0.85)
  I_Tagging_y_1 = Image(FOV=np.array([0.1, 0.1]),
                  resolution=np.array([70, 70]),
                  type='Tagging',
                  encoding_frequency=[900],
                  encoding_direction=[1],
                  flip_angle = np.pi/2,
                  invert_second_RF=True,
                  T1=0.85)

  # Generator
  gx0 = Generator(p, I_Tagging_x_0)
  gx1 = Generator(p, I_Tagging_x_1)
  gy0 = Generator(p, I_Tagging_y_0)
  gy1 = Generator(p, I_Tagging_y_1)

  # Element
  FE = VectorElement("triangle")

  # Mesh and fem space
  mesh = Mesh('mesh/mesh.msh')
  # mesh = fem_ventricle_geometry(p['R_en'], p['tau'], p['mesh_resolution'], filename='mesh/mesh.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = DefaultPhantom(p, function_space=V)

  # Generation
  gx0.phantom = phantom
  gx1.phantom = phantom
  gy0.phantom = phantom
  gy1.phantom = phantom
  tx0, mask = gx0.get_image()
  tx1, mask = gx1.get_image()
  ty0, mask = gy0.get_image()
  ty1, mask = gy1.get_image()

  # Add noise
  add_noise_to_SPAMM(tx0, mask, 0.03)
  add_noise_to_SPAMM(tx1, mask, 0.03)
  add_noise_to_SPAMM(ty0, mask, 0.03)
  add_noise_to_SPAMM(ty1, mask, 0.03)

  # Generate C-SPAMM images
  tx = np.zeros(tx0.shape, dtype=complex)
  ty = np.zeros(ty0.shape, dtype=complex)
  if rank==0:
    for i in range(30):
      tx[...,i] = iFFT(FFT(tx0[:, :, i]) - FFT(tx1[:, :, i]))
      ty[...,i] = iFFT(FFT(ty0[:, :, i]) - FFT(ty1[:, :, i]))

    tx -= tx.min()
    tx /= tx.max()
    ty -= ty.min()
    ty /= ty.max()

  # Plot
  if rank==0:
    u_min, u_max = tx.min(), tx.max()
    for i in range(30):
      t = np.real(tx[...,i])*np.real(ty[...,i])
      fig = plt.imshow(t, cmap=plt.get_cmap('Greys_r'), 
                      interpolation='nearest', vmin=u_min, vmax=u_max)
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
      plt.tight_layout()
      plt.savefig('tagging_{:04d}'.format(i))

    fig, ax = plt.subplots(1, 2)
    tracker = IndexTracker(ax, np.real(tx[:,:,:]), np.real(ty[:,:,:]))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.suptitle('Tagging volunteer {:d}'.format(rank))
    plt.show()
