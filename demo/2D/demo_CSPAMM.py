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
  p["time_steps"] = 18

  # Field inhomogeneity
  phi = lambda X, Y: (X+Y)/0.1*2.87

  # T1 map
  T1 = lambda X, Y: 0.85 + 0.05*np.sin(200*X)*np.sin(200*Y)

  # Create complimentary images
  FOV = 0.1
  N   = 70
  ke  = 785
  alpha = 15*np.pi/180
  beta  = 90*np.pi/180
  I0 = CSPAMMImage(FOV=np.array([FOV, FOV]),
            resolution=np.array([N, N]), 
            encoding_frequency=np.array([ke, ke]),
            flip_angle=alpha,
            encoding_angle=beta,
            T1=T1)

  # Element
  FE = VectorElement("triangle")

  # Mesh and fem space
  mesh = fem_ventricle_geometry(p['R_en'], p['tau'], p['mesh_resolution'], filename='mesh/mesh.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = DefaultPhantom(p, time_steps=p["time_steps"],function_space=V, patient=False)

  # Generator
  g0 = Generator(p, I0, phantom=phantom, debug=True)

  # Generation
  t0, t1, mask = g0.get_image()

  # Add noise
  tn0 = add_noise_to_SPAMM_(t0, mask, sigma=5e-08)
  tn1 = add_noise_to_SPAMM_(t1, mask, sigma=5e-08)

  # Generate C-SPAMM images
  t = np.zeros(t0.shape, dtype=complex)
  if rank==0:
    for i in range(t.shape[-1]):
      t[...,0,i] = tn0[...,0,i] + tn1[...,0,i]
      t[...,1,i] = tn0[...,1,i] + tn1[...,1,i]

    # Scale image
    o = scale_image(t,mag=True,pha=False,real=False,compl=False)
    # t = o["magnitude"]["Image"]

    # Plot k-space
    fig, ax = plt.subplots(1,2)
    step = -1
    im0 = ax[0].imshow(np.abs((t[...,0,0])))
    im1 = ax[1].imshow(np.abs((t[...,0,step])))
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

    # Plot generated images
    fig, ax = plt.subplots(1, 2)
    tracker = IndexTracker(ax, np.abs(t[:,:,0,:]), np.imag(t[:,:,0,:]))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.suptitle('Tagging volunteer {:d}'.format(rank))
    plt.show()