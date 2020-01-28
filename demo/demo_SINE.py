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
  p = Parameters_2D(mesh_resolution = 0.0003)
  p["time_steps"] = 5
  np.save("p.npy", p)
  p = np.load('p.npy').item()
  p["sigma"] = 1

  # Create complimentary images
  FOV = 0.1
  N   = 100
  ke  = 785
  T1  = -1.0/np.log(0.5)
  I0 = SINEImage(FOV=np.array([FOV, FOV]),
            resolution=np.array([N, N]), 
            encoding_frequency=np.array([ke, ke]),
            T1=T1)

  # Create phantom object
  phantom = PixelledPhantom(p, time_steps=p["time_steps"], image=I0, patient=False)

  # Generator
  g0 = Generator(p, I0, phantom=phantom, debug=True)

  # Generation
  t, mask = g0.get_image()

  # Add noise
  tn = add_noise_to_SPAMM_(t, mask, sigma=1e-06)

  # Generate C-SPAMM images
  if rank==0:

    # Scale image
    o = scale_image(t,mag=True,pha=False,real=False,compl=False)
    # t = o["magnitude"]["Image"]

    # Plot k-space
    fig, ax = plt.subplots(1,2)
    step = -1
    im0 = ax[0].imshow(np.abs(FFT(t[...,0,step])))
    im1 = ax[1].imshow(np.abs(FFT(t[...,1,step])))
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
    tracker = IndexTracker(ax, np.real(tn[:,:,0,:]), np.abs(tn[:,:,0,:]*t[:,:,1,:]))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.suptitle('Tagging volunteer {:d}'.format(rank))
    plt.show()