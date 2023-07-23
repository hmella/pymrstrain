import matplotlib.pyplot as plt
import numpy as np

from PyMRStrain.MPIUtilities import scatterKspace

# Generic tracjectory
class Trajectory:
  def __init__(self, FOV=np.array([0.3, 0.3]), res=np.array([100, 100]),
              oversampling=2, max_Gro_amp=30, Gro_slew_rate=195,lines_per_shot=7, gamma=42.58, VENC=None):
      self.FOV = FOV
      self.res = res
      self.oversampling = oversampling
      self.max_Gro_amp = max_Gro_amp         # [mT/m]
      self.max_Gro_amp_ = 1.0e-3*max_Gro_amp # [T/m]
      self.Gro_slew_rate = Gro_slew_rate     # [mT/(m*ms)]
      self.gamma = gamma                     # [MHz/T]
      self.gamma_ = 2.0*np.pi*1.0e+6*gamma   # [rad Hz/T]
      self.lines_per_shot = lines_per_shot
      self.pxsz = FOV/res
      self.kspace_bw = 1.0/self.pxsz
      self.kspace_spa = 1.0/(oversampling*self.FOV)
      self.ro_samples = oversampling*res[0]
      self.VENC = VENC  # [m/s]

  def check_ph_enc_lines(self, ph_samples):
    ''' Verify if the number of lines in the phase encoding direction
    satisfies the multishot factor '''
    lps = self.lines_per_shot
    if ph_samples % lps != 0:
      ph_samples = int(lps*np.floor(ph_samples/lps)) 

    return ph_samples

  def velenc_time(self):
    ''' Calculate the time needed to apply the velocity encoding gradients
    based on the values of max_Gro_amp and Gro_slew_rate'''

    # Bipolar lobes areas without rectangle part
    dur_to_max = self.max_Gro_amp_/self.Gro_slew_rate
    alpha = np.pi/(self.gamma_*self.VENC*self.max_Gro_amp_)
    t1 = (0.5*alpha*dur_to_max)**(1/3)

    # Check if rectangle parts of the gradient are needed
    if t1 <= dur_to_max:
      # Calculate duration of the first velocity encoding gradient lobe
      dur2 = 2.0*t1
      enc_time = 2.0*dur2

      # Plot
      t = np.array([0, t1, 2*t1, 3*t1, 4*t1])
      G_max = t1/dur_to_max*self.max_Gro_amp
      G = np.array([0, G_max, 0, -G_max, 0])
      plt.figure(1)
      plt.plot(1000.0*t,G)
      plt.show()

    else:
      # Estimate the duration of the rectangular part of the gradient
      a = 1.0 
      b = 3.0*dur_to_max
      c = 2.0*dur_to_max**2 - alpha
      t2 = np.array([(-b + (b**2 - 4*a*c)**0.5)/(2*a),
                     (-b - (b**2 - 4*a*c)**0.5)/(2*a)])

      # Remove negative solutions
      t2[t2 < 0] = 1e+10
      t2 = t2.min()

      # Gradients duration
      dur2 = 2.0*dur_to_max + t2
      enc_time = 2*dur2

      # Plot
      t = np.array([0, dur_to_max, dur_to_max+t2, 2*dur_to_max+t2, 3*dur_to_max+t2, 3*dur_to_max+2*t2, 4*dur_to_max+2*t2])
      G_max = self.max_Gro_amp
      G = np.array([0, G_max, G_max, 0, -G_max, -G_max, 0])
      plt.figure(1)
      plt.plot(1000.0*t,G)
      plt.show()

    # Store velocity encoding time
    self.vel_enc_time = enc_time

    return enc_time


# Cartesian trajectory
class Cartesian(Trajectory):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.ph_samples = self.check_ph_enc_lines(self.res[1])
      (self.points, self.times) = self.kspace_points()

    def readout_time(self):
      return 1

    def kspace_points(self):
      ''' Get kspace points '''
      # Time needed to acquire one line
      # It depends on the kspcae bandwidth, the gyromagnetic constant, and
      # the maximun gradient amplitude
      dt_line = (self.kspace_bw[0]*2*np.pi)/(self.gamma_*self.max_Gro_amp_)
      dt = np.linspace(0.0, dt_line, self.ro_samples)

      # Readout gradient timings
      slope = self.max_Gro_amp_/self.Gro_slew_rate
      lenc  = (0.5*self.kspace_bw[0]*2*np.pi - self.gamma_*slope*self.max_Gro_amp_)/(self.gamma_*self.max_Gro_amp_)

      # Update timings to include bipolar gradients and pre-positioning gradient before the readout
      venc_time = 0.0
      if self.VENC != None:
        venc_time = self.velenc_time()

      # kspace locations
      kx = np.linspace(-0.5*self.kspace_bw[0], 0.5*self.kspace_bw[0], self.ro_samples)
      ky = 0.5*self.kspace_bw[1]*np.ones(kx.shape)
      kspace = (np.zeros([self.ro_samples, self.ph_samples]),
                np.zeros([self.ro_samples, self.ph_samples]))

      # kspace times and locations
      t = np.zeros([self.ro_samples, self.ph_samples])
      for ph in range(0,self.ph_samples):
        # Fill locations
        if (ph+1) % 2 != 0:
          kspace[0][:,ph] = kx
          kspace[1][:,ph] = ky - self.kspace_bw[1]*(ph/(self.ph_samples-1))
        else:
          kspace[0][::-1,ph] = kx
          kspace[1][::-1,ph] = ky - self.kspace_bw[1]*(ph/(self.ph_samples-1))

        # Update timings
        if ph % self.lines_per_shot == 0:
          t[:,ph] = venc_time + slope + lenc + slope + slope + dt
        else:
          t[:,ph] = t[-1,ph-1] + slope + slope + dt

      # Calculate echo time
      self.echo_time = 0.5*(t[:].max() - 3*slope - lenc - venc_time) + 3*slope + lenc + venc_time

      # Send the information to each process if running in parallel
      kspace, t, local_idx = scatterKspace(kspace, t)
      self.local_idx = local_idx

      return (kspace, t)

    def plot_trajectory(self):
      # Plot kspace locations and times
      fig, axs = plt.subplots(1, 2, figsize=(10, 5))
      axs[0].plot(self.points[0].flatten('F'),self.points[1].flatten('F'))
      im = axs[1].scatter(self.points[0],self.points[1],c=1000*self.times,s=1.5)
      axs[0].set_xlabel('k_x (1/m)')
      axs[0].set_ylabel('k_y (1/m)')
      axs[1].set_xlabel('k_x (1/m)')
      axs[1].set_ylabel('k_y (1/m)')
      cbar = fig.colorbar(im, ax=axs[1])
      cbar.ax.tick_params(labelsize=8) 
      cbar.ax.set_title('Time [ms]',fontsize=8)
      plt.show()


# Radial trajectory
class Radial(Trajectory):
    def __init__(self, *args, spokes=20, **kwargs):
      super().__init__(*args, **kwargs)
      self.spokes = self.check_ph_enc_lines(spokes)
      (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
      ''' Get kspace points '''
      # Time needed to acquire one line
      dt_line = (self.kspace_bw[0]*2*np.pi)/(self.gamma_*self.max_Gro_amp_)
      dt = np.linspace(0.0, dt_line, self.ro_samples)

      # kspace locations
      kx = np.linspace(-0.5*self.kspace_bw[0], 0.5*self.kspace_bw[0], self.ro_samples)
      ky = np.zeros(kx.shape)
      kspace = (np.zeros([self.ro_samples, self.spokes]),
                np.zeros([self.ro_samples, self.spokes]))

      # Angles for each ray
      theta = np.linspace(0, np.pi, self.spokes+1)
      theta = theta[0:-1]

      # kspace times and locations
      t = np.zeros([self.ro_samples, self.spokes])
      for sp in range(0, self.spokes):
        # Rotation matrix
        if (sp+1) % 2 != 0:
          kspace[0][:,sp] = kx*np.cos(theta[sp]) + ky*np.sin(theta[sp])
          kspace[1][:,sp] = -kx*np.sin(theta[sp]) + ky*np.cos(theta[sp])
        else:
          kspace[0][::-1,sp] = kx*np.cos(theta[sp]) + ky*np.sin(theta[sp])
          kspace[1][::-1,sp] = -kx*np.sin(theta[sp]) + ky*np.cos(theta[sp])

        if sp % self.lines_per_shot == 0:
          t[:,sp] = dt
        else:
          t[:,sp] = t[-1,sp-1] + dt

      # Send the information to each process if running in parallel
      kspace, t, local_idx = scatterKspace(kspace, t)
      self.local_idx = local_idx

      return (kspace, t)

    def plot_trajectory(self):
      ''' Show kspace points '''
      # Plot kspace locations and times
      fig, axs = plt.subplots(1, 2, figsize=(10, 5))
      axs[0].plot(self.points[0].flatten('F'),self.points[1].flatten('F'))
      im = axs[1].scatter(self.points[0],self.points[1],c=1000*self.times,s=1.5)
      axs[0].set_xlabel('k_x (1/m)')
      axs[0].set_ylabel('k_y (1/m)')
      axs[1].set_xlabel('k_x (1/m)')
      axs[1].set_ylabel('k_y (1/m)')
      cbar = fig.colorbar(im, ax=axs[1])
      cbar.ax.tick_params(labelsize=8) 
      cbar.ax.set_title('Time [ms]',fontsize=8)
      plt.show()


# Radial trajectory
class Spiral(Trajectory):
    def __init__(self, *args, interleaves=20, parameters=[], **kwargs):
      super().__init__(*args, **kwargs)
      self.interleaves = self.check_ph_enc_lines(interleaves)
      self.parameters = parameters
      (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
      ''' Get kspace points '''
      # Spiral parameters
      N = self.interleaves        # Number of interleaves
      f = self.FOV[0]             # Field-of-view
      k0 = self.parameters['k0']
      k1 = self.parameters['k1']
      S = self.parameters['Slew-rate']    # Slew-rate [T/m/s]
      gamma = self.gamma
      r = self.pxsz[0]

      # Radial distance definition
      kr0 = k0*0.5*self.kspace_bw[0]
      kr1 = k1*0.5*self.kspace_bw[0]
      kr = np.linspace(0, 1, self.ro_samples)
      kr = kr1*(kr**1)
      phi = 2*np.pi*f*kr/N

      # Complex trajectory
      K = kr*np.exp(1j*phi)

      # Time needed to acquire one interleave
      t_sc = np.sqrt(2)*np.pi*f/(3*N*np.sqrt(1e+6*gamma*1e-3*S)*r**(3/2))
      dt = np.linspace(0.0, t_sc, self.ro_samples)    

      # kspace locations
      kx = np.real(K)
      ky = np.imag(K)
      kspace = (np.zeros([self.ro_samples, self.interleaves]),
                np.zeros([self.ro_samples, self.interleaves]))

      # Angles for each ray
      theta = np.linspace(0, 2*np.pi, self.interleaves+1)
      theta = theta[0:-1]

      # kspace times and locations
      t = np.zeros([self.ro_samples, self.interleaves])
      for sp in range(0, self.interleaves):
        # Rotation matrix
        kspace[0][:,sp] = kx*np.cos(theta[sp]) + ky*np.sin(theta[sp])
        kspace[1][:,sp] = -kx*np.sin(theta[sp]) + ky*np.cos(theta[sp])

        if sp % self.lines_per_shot == 0:
          t[:,sp] = dt
        else:
          t[:,sp] = t[-1,sp-1] + dt

      # Send the information to each process if running in parallel
      kspace, t, local_idx = scatterKspace(kspace, t)
      self.local_idx = local_idx

      return (kspace, t)

    def plot_trajectory(self):
      ''' Show kspace points '''
      # Plot kspace locations and times
      fig, axs = plt.subplots(1, 2, figsize=(10, 5))
      axs[0].plot(self.points[0].flatten('F'),self.points[1].flatten('F'))
      im = axs[1].scatter(self.points[0],self.points[1],c=1000*self.times,s=1.5)
      axs[0].set_xlabel('k_x (1/m)')
      axs[0].set_ylabel('k_y (1/m)')
      axs[1].set_xlabel('k_x (1/m)')
      axs[1].set_ylabel('k_y (1/m)')
      cbar = fig.colorbar(im, ax=axs[1])
      cbar.ax.tick_params(labelsize=8) 
      cbar.ax.set_title('Time [ms]',fontsize=8)
      plt.show()


def SpiralCalculator():
  return True
