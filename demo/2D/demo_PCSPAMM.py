from PyMRStrain import *
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt

def FFT(x):
  return fftshift(fftn(ifftshift(x)))

def iFFT(x):
  return fftshift(ifftn(ifftshift(x)))

if __name__=="__main__":

  # Parameters
  p = Parameters_2D(mesh_resolution = 0.00035)

  # Create complimentary images
  FOV = 0.1
  N   = 100
  ke  = 900
  kv  = 100
  alpha = 15*np.pi/180
  beta  = 90*np.pi/180
  I0 = PCSPAMMImage(FOV=np.array([FOV, FOV]),
            resolution=np.array([N, N]), 
            v_encoding_frequency=np.array([kv, kv]),
            encoding_frequency=np.array([ke, ke]),
            flip_angle=alpha,
            encoding_angle=beta,
            T1=0.85)

  # Element
  FE = VectorElement("triangle")

  # Mesh and fem space
#  mesh_v = Mesh('mesh/mesh.msh')
  mesh = fem_ventricle_geometry(p['R_en'], p['tau'], p['mesh_resolution'], filename='mesh/mesh.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = DefaultPhantom(p, function_space=V)

  # Generator
  g0 = Generator(p, I0, phantom=phantom, debug=True)

  # Generation
  t0, t1, mask = g0.get_image()

  # Add noise
  tn0 = add_noise_to_SPAMM_(t0, mask, sigma=0.03)
  tn1 = add_noise_to_SPAMM_(t1, mask, sigma=0.03)

  if rank==0:
    # Phase correction
    tn1 = np.conj(tn1)

    # Generate C-SPAMM and PC images
    t = np.zeros(t0.shape, dtype=complex)
    v = np.zeros(t0.shape, dtype=np.float)

    for i in range(t.shape[-1]):
      # C-SPAMM
      t[...,0,i] = iFFT(FFT(tn0[...,0,i]) - FFT(tn1[...,0,i]))
      t[...,1,i] = iFFT(FFT(tn0[...,1,i]) - FFT(tn1[...,1,i]))

      # PC
      v[...,0,i] = np.angle(iFFT(FFT(tn0[...,0,i]) + FFT(tn1[...,0,i])))
      v[...,1,i] = np.angle(iFFT(FFT(tn0[...,1,i]) + FFT(tn1[...,1,i])))

    # CSPAMM phase correction
    t = np.abs(t)*np.exp(1j*(np.angle(t)-v))

    # Scale image
    o = scale_image(t,mag=True,pha=True,real=False,compl=False)
    t = o["magnitude"]["Image"]

    fig, ax = plt.subplots(1,2)
    step = 13
    im0 = ax[0].imshow(np.abs(FFT(t0[...,0,step])))
    im1 = ax[1].imshow(np.abs(FFT(t0[...,0,step]) + FFT(t1[...,0,step])))
    ax[0].set_title('SPAMM')
    ax[1].set_title('CSPAMM')
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)
    ax[1].axes.get_xaxis().set_visible(False)
    ax[1].axes.get_yaxis().set_visible(False)
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    plt.tight_layout()
    plt.show()

  # Plot
  if rank==0:
#    u_min, u_max = t_v.min(), t_v.max()
#    for i in range(30):
#      fig = plt.imshow(t_v[:, :, i, 0], cmap=plt.get_cmap('Greys_r'), 
#                       interpolation='nearest', vmin=0.5*u_min, vmax=u_max)
#      fig.axes.get_xaxis().set_visible(False)
#      fig.axes.get_yaxis().set_visible(False)
#      plt.savefig('tagging_{:04d}'.format(i))

    fig, ax = plt.subplots(1, 2)
    tracker = IndexTracker(ax, np.real(t[:,:,1,:]), v[:,:,1,:])
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)
    ax[1].axes.get_xaxis().set_visible(False)
    ax[1].axes.get_yaxis().set_visible(False)
    plt.show()