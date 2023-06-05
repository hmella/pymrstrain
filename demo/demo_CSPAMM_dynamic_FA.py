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
  p = Parameters(time_steps=18)
  p.h = 0.008
  p.phi_en = -15*np.pi/180
  p.phi_ep = 0*np.pi/180
  # save_pyobject(p, 'p.pkl')
  # p=load_pyobject('p.pkl')

  # Field inhomogeneity
  phi = lambda X, Y: 0*(X+Y)/0.1*0.2

  # Encoding frequency
  ke = 0.07               # encoding frequency [cycles/mm]
  ke = 1000*2*np.pi*ke    # encoding frequency [rad/m]

  # Variable flip angles
  TR      = 1.0/p.time_steps            # repetition time
  T1      = 0.85                        # reference T1 value
  decay   = np.exp(-TR/T1)              # decay term
  FA      = np.zeros([p.time_steps,])   # array of flip angles
  last_FA = 10*np.pi/180                # flip angle of the last frame
  FA[-1]  = last_FA                     # set flip angle of the last frame
  ramp    = np.zeros([p.time_steps,])   # array of ramp values
  gamma   = 5*np.pi/180
  for i in range (p.time_steps-2,-1,-1):
    ramp[i] = gamma*(p.time_steps - i - 1)/(p.time_steps - 1)
    FA[i]   = np.arctan(np.sin(last_FA)*decay) + ramp[i]
    last_FA = FA[i] - ramp[i]

  # Show variable flip angles
  plt.figure(1)
  plt.plot(180*(FA-ramp)/np.pi)
  plt.plot(180*ramp/np.pi)
  plt.plot(180*FA/np.pi)
  plt.legend(['Fischer','rampdown','Sampath'])
  plt.show()

  # Create complimentary image
  I = CSPAMMImage(FOV=np.array([0.3, 0.3, 0.008]),
            center=np.array([0.0,0.0,0.0]),
            resolution=np.array([164, 164, 1]),
            encoding_frequency=np.array([ke,ke,0]),
            T1=np.array([1e-10,1e-10,0.85]),
            M0=np.array([0,0,1]),
            flip_angle=FA,
            encoding_angle=90*np.pi/180,
            off_resonance=phi,
            kspace_factor=2,
            slice_thickness=0.008,
            oversampling_factor=1,
            phase_profiles=66)

  # Spins
  spins = Spins(Nb_samples=75000, parameters=p)

  # Create phantom object
  phantom = Phantom(spins, p, patient=False, z_motion=False, write_vtk=False)

  # EPI acquisiton object
  epi = EPI(receiver_bw=128*KILO,
            echo_train_length=11,
            off_resonance=200,
            acq_matrix=I.acq_matrix,
            spatial_shift='top-down')

  # Generate images
  start = time.time()
  NSA_1, NSA_2, mask = I.generate(None, phantom, p, debug=True)
  end = time.time()
  print(end-start)

  # Add noise to images
  sigma = 0.025
  # NSA_1.k = add_cpx_noise(NSA_1.k, mask=NSA_1.k_msk, sigma=sigma)
  # NSA_2.k = add_cpx_noise(NSA_2.k, mask=NSA_2.k_msk, sigma=sigma)

  # kspace to image
  In1 = NSA_1.to_img()
  In2 = NSA_2.to_img() 
  mask = mask.to_img()

  # CSPAMM image
  I = In1 - In2

  # Plot
  if MPI_rank==0:
      multi_slice_viewer(np.abs(NSA_1.k[:,:,0,0,:]-NSA_2.k[:,:,0,0,:] + 
                         (NSA_1.k[:,:,0,1,:]-NSA_2.k[:,:,0,1,:])))
      multi_slice_viewer(np.abs(I[:,:,0,1,:]))
