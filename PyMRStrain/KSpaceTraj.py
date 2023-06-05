import matplotlib.pyplot as plt
import numpy as np


# Generic tracjectory
class Trajectory:
  def __init__(self, FOV=np.array([0.3, 0.3]), res=np.array([100, 100]),
              oversampling=2, max_Gro_amp=35, lines_per_shot=8, gamma=42.58):
      self.FOV = FOV
      self.res = res
      self.oversampling = oversampling
      self.max_Gro_amp = max_Gro_amp # [mT/m]
      self.gamma = gamma             # [MHz/T]
      self.lines_per_shot = lines_per_shot
      self.pxsz = FOV/res
      self.kspace_bw = 1.0/self.pxsz
      self.kspace_spa = self.kspace_bw/res
      self.ro_samples = oversampling*res[0]

  def check_ph_enc_lines(self, ph_samples):
    ''' Verify if the number of lines in the phase encoding direction
    satisfies the multishot factor '''
    lps = self.lines_per_shot
    if ph_samples % lps != 0:
      ph_samples = int(lps*np.floor(ph_samples/lps))
    
    return ph_samples


# Cartesian trajectory
class Cartesian(Trajectory):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.ph_samples = self.check_ph_enc_lines(self.res[1])
      (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
      ''' Get kspace points '''
      # Time needed to acquire one line
      # It depends on the kspcae bandwidth, the gyromagnetic constant, and
      # the maximun gradient amplitude
      dt_line = (self.kspace_bw[0]*2*np.pi)/(1e+6*self.gamma*1e-3*self.max_Gro_amp)
      dt = np.linspace(0.0, dt_line, self.ro_samples)

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

        if ph % self.lines_per_shot == 0:
          t[:,ph] = dt
        else:
          t[:,ph] = t[-1,ph-1] + dt

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
      dt_line = (self.kspace_bw[0]*2*np.pi)/(1e+6*self.gamma*1e-3*self.max_Gro_amp)
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
