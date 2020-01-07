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
  p = {'t_end': 1.0,
       'dt': 1.0/20.0,
       't': 0.0,
       'time_steps': 20,
       'mesh_resolution': 0.0001}

  # imaging paramenters
  res = np.array([100, 100, 25])
  FOV = np.array([0.15, 0.15, 0.2])
  cen = np.array([0.01,-0.01,-0.03])
  alpha = 15*np.pi/180
  beta  = 90*np.pi/180

  # encoding frequency
  n = 5                          # number of pixels per wavelength
  pxsz = FOV[0]/res[0]           # pixel size [m]
  lmbda = n*pxsz                 # wavelength [m]
  ke = np.double(2*np.pi/lmbda)  # encoding frequency [rad/m]

  # Create complimentary images
  I0 = CSPAMMImage(FOV=FOV,
            resolution=res, 
            center=cen,
            encoding_frequency=np.array([ke,ke]),
            flip_angle=alpha,
            encoding_angle=beta,
            T1=0.85)

  # Element
  FE = VectorElement('tetrahedron10')

  # Mesh and fem space
  mesh = Mesh('mesh/mesh10.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = Phantom3D(function_space=V, time_steps=p['time_steps'], path='files/')

  # Generators
  g0 = Generator(p, I0, phantom=phantom, debug=True)

  # Generation
  t0, t1, mask = g0.get_image()

  # Add noise
  tn0 = add_noise_to_SPAMM_(t0, mask, sigma=0.05)
  tn1 = add_noise_to_SPAMM_(t1, mask, sigma=0.05)

  # Generate C-SPAMM images
  t = np.zeros(t0.shape, dtype=complex)
  if rank==0:
    for i in range(t.shape[-1]):
      for z in range(t.shape[2]):
        t[...,z,0,i] = iFFT(FFT(tn0[...,z,0,i]) - FFT(tn1[...,z,0,i]))
        t[...,z,1,i] = iFFT(FFT(tn0[...,z,1,i]) - FFT(tn1[...,z,1,i]))
        t[...,z,2,i] = iFFT(FFT(tn0[...,z,2,i]) - FFT(tn1[...,z,2,i]))

    # Scale image
    o = scale_image(t,mag=True,pha=False,real=False,compl=False)
    oo = scale_image(tn0,mag=True,pha=False,real=False,compl=False)

  # Plot
  if rank==0:
    # Export file
    export_image(oo,path='I_SPAMM',name='Is')
    export_image(o,path='I_CSPAMM',name='I')
    export_image(mask,path='mask', name='mask')