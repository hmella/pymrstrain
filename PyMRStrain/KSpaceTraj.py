import matplotlib.pyplot as plt
import numpy as np

from PyMRStrain.MPIUtilities import scatterKspace

# plt.rcParams['text.usetex'] = True

# Gradient
class Gradient:
  def __init__(self, slope=1.0, lenc=1.0, G=1.0, Gr_max=30.0, Gr_sr=195.0, gammabar=42.58, t_ref=0.0):
    self.slope = slope   # [ms]
    self.lenc = lenc     # [ms]
    self.G = G           # [mT/m]
    self.slope_ = 1.0e-3*slope   # [s]
    self.lenc_ = 1.0e-3*lenc     # [s]
    self.G_ = 1.0e-3*G           # [T/m]
    self.Gr_max = Gr_max         # [mT/m]
    self.Gr_max_ = 1.0e-3*Gr_max # [T/m]
    self.Gr_sr = Gr_sr     # [mT/(m*ms)]
    self.Gr_sr_ = Gr_sr    # [T/(m*s)]
    self.gammabar = gammabar               # [MHz/T]
    self.gammabar_ = 1.0e+6*gammabar       # [Hz/T]
    self.gamma_ = 2.0*np.pi*1e+6*gammabar  # [rad/T]
    self.t_ref = t_ref
    self.timings, self.amplitudes = self.group_timings()

  def __add__(self, g):
    # Concatenate gradients
    return 1

  def group_timings(self):
    if self.lenc <= 0.0:
      timings = np.array([0.0, 
                          self.slope,
                          self.slope+self.slope])
      amplitudes = np.array([0.0, 
                            self.G,
                            0.0])
    else:
      timings = np.array([0.0, 
                          self.slope, 
                          self.slope+self.lenc, 
                          self.slope+self.lenc+self.slope])
      amplitudes = np.array([0.0, 
                            self.G, 
                            self.G, 
                            0.0])

    # Add reference time
    timings += self.t_ref

    return timings, amplitudes

  def calculate(self, bw_hz, receiver_bw=None, ro_samples=None, ofac=None):
    ''' Calculate gradient based on target area 
                                             __________
                _1_ /|\                    /|          |\
     slew rate |  /  |  \                /  |          |  \
               |/    G    \            /    G          G    \
              /      |      \        /      |          |      \
               slope   slope          slope     lenc     slope
    '''
    # Calculate gradient
    if receiver_bw != None:
      ''' If the receiver bandwidth is given, the lenc should be fixed and
      calculated accordingly
      '''
      self.lenc_  = (ro_samples/ofac)/receiver_bw       # [s]
      self.G_     = bw_hz/(self.gammabar_*self.lenc_)   # [T/m]
      self.slope_ = np.abs(self.G_)/self.Gr_sr_ # [s]
    else:
      ''' Calcualte everything as if only the ramps are needed '''
      # Time needed to reach the maximun amplitude
      slope_req_ = np.sqrt(np.abs(bw_hz)/(self.gammabar_*self.Gr_sr_)) # [s]

      # Time needed for the 
      slope_ = self.Gr_max_/self.Gr_sr_ # [s]

      # Build gradient
      if slope_req_ < slope_:
        self.slope_ = slope_req_                # [s]
        self.G_     = self.Gr_sr_*slope_req_    # [T/m]
        self.lenc_  = self.slope_ - slope_req_  # [s]
      else:
        # Assign slope and gradient amplitude
        self.slope_  = slope_            # [s]
        self.G_      = self.Gr_max_ # [T/m]

        # Calculate lenc
        bw_hz_slopes_ = self.gammabar_*self.Gr_sr_*slope_**2
        self.lenc_    = (np.abs(bw_hz) - bw_hz_slopes_)/(self.G_*self.gammabar_)

      # Account for area sign
      self.G_ *= np.sign(bw_hz)

    # Store variables in mT - ms
    self.lenc  = 1.0e+3*self.lenc_    # [ms]
    self.G     = 1.0e+3*self.G_       # [mT/m]
    self.slope = 1.0e+3*self.slope_   # [ms]
    if self.lenc < 0:
      self.dur_ = self.slope_ + self.slope_ # [ms]
    else:
      self.dur_ = self.slope_ + self.lenc_ + self.slope_ # [ms]
    self.dur = 1.0e+3*self.dur_ # [s]

    # Update timings and amplitudes in array
    self.timings, self.amplitudes = self.group_timings()

  def calculate_bipolar(self, VENC):
    ''' Calculate the time needed to apply the velocity encoding gradients
    based on the values of Gr_max and Gr_sr'''

    # Bipolar lobes areas without rectangle part
    # If lenc = 0, 2*pi = gamma*G(t)*VENC*slope^2
    # which is equivalent to: 2*pi = gamma*SR*VENC*slope^3
    slope_ = self.Gr_max_/self.Gr_sr_ # [s]
    slope_req_ = np.cbrt(np.pi/(2*self.gamma_*VENC*self.Gr_sr_))

    # Check if rectangle parts of the gradient are needed
    if slope_req_ <= slope_:
      # Calculate duration of the first velocity encoding gradient lobe
      self.slope_ = slope_req_                # [s]
      self.G_     = self.Gr_sr_*slope_req_    # [s]
      self.lenc_  = self.slope_ - slope_req_  # [s]
    else:
      # Lobe area (only ramps)
      area_ramps = slope_*self.Gr_max_

      # Estimate remaining area 
      # If lenc != 0:
      #     pi = gamma*Gmax*VENC*(lenc + slope)*(lenc + 2*slope)
      # which is equivalent to
      #     pi = gamma*VENC*(Gmax*lenc + Ar)*(lenc + 2*slope)
      a = self.Gr_max_
      b = area_ramps + 2*slope_*self.Gr_max_
      c = (2*slope_*area_ramps - np.pi/(self.gamma_*VENC))
      t2 = np.array([(-b + np.sqrt(b**2 - 4*a*c))/(2*a),
                     (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])

      # Remove negative solutions
      t2[t2 < 0] = 1e+10
      t2 = t2.min()

      # Gradients parameters
      self.slope_ = slope_
      self.G_     = self.Gr_max_
      self.lenc_  = t2

    # Store variables in mT - ms
    self.lenc  = 1.0e+3*self.lenc_    # [ms]
    self.G     = 1.0e+3*self.G_       # [mT/m]
    self.slope = 1.0e+3*self.slope_   # [ms]
    if self.lenc < 0:
      self.dur_ = 2*(self.slope_ + self.slope_) # [s]
    else:
      self.dur_ = 2*(self.slope_ + self.lenc_ + self.slope_) # [s]
    self.dur = 1.0e+3*self.dur_ # [ms]

    # Update timings and amplitudes in array
    timings, amplitudes = self.group_timings()  
    self.timings = np.concatenate((timings, timings[-1] + timings))
    self.amplitudes = np.concatenate((amplitudes, -amplitudes))


  def plot(self, linestyle='-', axes=[]):
    ''' Plot gradient '''
    fig = plt.figure(2)
    plt.plot(self.timings, self.amplitudes, linestyle)
    plt.show()

    return fig

# Generic tracjectory
class Trajectory:
  def __init__(self, FOV=np.array([0.3, 0.3, 0.08]), res=np.array([100, 100, 1]), oversampling=2, Gr_max=30, Gr_sr=195, lines_per_shot=7, gammabar=42.58, VENC=None, receiver_bw=128.0e+3, plot_seq=False, MPS_ori=np.eye(3)):
      self.FOV = FOV
      self.res = res
      self.oversampling = oversampling
      self.Gr_max = Gr_max          # [mT/m]
      self.Gr_max_ = 1.0e-3*Gr_max  # [T/m]
      self.Gr_sr  = Gr_sr     # [mT/(m*ms)]
      self.Gr_sr_ = Gr_sr     # [T/(m*s)]
      self.gammabar = gammabar                # [MHz/T]
      self.gammabar_ = 1e+6*gammabar          # [Hz/T]
      self.gamma_ = 2*np.pi*1e+6*gammabar     # [rad/T]
      self.lines_per_shot = lines_per_shot
      self.pxsz = FOV/res
      self.kspace_bw = 1.0/self.pxsz
      self.kspace_spa = self.kspace_bw/np.array([oversampling*res[0]-1, res[1]-1, res[2]])
      self.ro_samples = oversampling*res[0] # number of readout samples
      self.slices = res[2]               # number of slices
      self.VENC = VENC  # [m/s]
      self.plot_seq = plot_seq
      self.receiver_bw = receiver_bw          # [Hz]
      self.MPS_ori = MPS_ori  # orientation

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
      # Bipolar gradient
      if self.VENC != None:
        bipolar = Gradient(t_ref=0.0, Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
        bipolar.calculate_bipolar(self.VENC)

      # k-space positioning gradients
      ph_grad = Gradient(t_ref=bipolar.timings[-1], Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
      ph_grad.calculate(0.5*self.kspace_bw[1])

      ro_grad0 = Gradient(t_ref=bipolar.timings[-1], Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
      ro_grad0.calculate(-0.5*self.kspace_bw[0] - 0.5*ro_grad0.gammabar_*ro_grad0.G_*ro_grad0.slope_)

      blip_grad = Gradient(t_ref=0.0, Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
      blip_grad.calculate(-self.kspace_bw[1]/self.ph_samples)

      ro_gradients = [bipolar, ro_grad0, ]
      ph_gradients = [ph_grad, ]
      for i in range(self.lines_per_shot):
        # Calculate readout gradient
        ro_grad = Gradient(t_ref=ro_gradients[i+1].timings[-1], Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
        ro_grad.calculate((-1)**i*self.kspace_bw[0], receiver_bw=self.receiver_bw, ro_samples=self.ro_samples, ofac=self.oversampling)
        ro_gradients.append(ro_grad)

        # Calculate blip gradient
        ref = ro_gradients[-1].t_ref + ro_gradients[-1].dur - 0.5*blip_grad.dur
        blip_grad = Gradient(t_ref=ref, Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
        blip_grad.calculate(-self.kspace_bw[1]/self.ph_samples)
        ph_gradients.append(blip_grad)

      if self.plot_seq:
        plt.rcParams['text.usetex'] = True
        fig = plt.figure(figsize=(8,3))
        for gr in ph_gradients[:-1]:
          ax1 = plt.plot(gr.timings, gr.amplitudes, 'r-', linewidth=2)
        for gr in ro_gradients:
          ax2 = plt.plot(gr.timings, gr.amplitudes, 'b-', linewidth=2)
        for i in range(2, self.lines_per_shot+2):
          a = [ro_gradients[i].t_ref + ro_gradients[i].slope,
               ro_gradients[i].t_ref + ro_gradients[i].slope + ro_gradients[i].lenc]
          ax3 = plt.plot(a, [0, 0], 'm:', linewidth=2)
        plt.tick_params('both', length=5, width=1, which='major', labelsize=16)
        plt.tick_params('both', length=3, width=1, which='minor', labelsize=16)
        plt.minorticks_on()
        plt.xlabel('$t~\mathrm{(ms)}$', fontsize=20)
        plt.ylabel('$G~\mathrm{(mT/m)}$', fontsize=20)
        ax1[0].set_label('PH')
        ax2[0].set_label('RO')
        ax3[0].set_label('ADC')
        plt.legend(fontsize=14, loc='upper right', ncols=3)
        plt.axis([0, ro_gradients[-1].t_ref + ro_gradients[-1].dur, -1.4*self.Gr_max, 1.4*self.Gr_max])
        plt.yticks([-40, -20, 0, 20, 40])
        plt.tight_layout()
        fig.savefig('sequence.pdf')
        plt.show()

      # Time needed to acquire one line
      # It depends on the kspcae bandwidth, the gyromagnetic constant, and
      # the maximun gradient amplitude
      dt = np.linspace(0.0, ro_grad.lenc_, self.ro_samples)

      # Update timings to include bipolar gradients and pre-positioning gradient before the readout
      venc_time = 0.0
      if self.VENC != None:
        venc_time = bipolar.dur_

      # kspace locations
      kx = np.linspace(-0.5*self.kspace_bw[0], 0.5*self.kspace_bw[0], self.ro_samples)
      ky = -0.5*self.kspace_bw[1]*np.ones(kx.shape)
      kz = np.linspace(-0.5*self.kspace_bw[2], 0.5*self.kspace_bw[2], self.slices)
      kspace = (np.zeros([self.ro_samples, self.ph_samples, self.slices],dtype=np.float32),
                np.zeros([self.ro_samples, self.ph_samples, self.slices],dtype=np.float32),
                np.zeros([self.ro_samples, self.ph_samples, self.slices],dtype=np.float32))

      # Fix kspace shifts
      if (self.ro_samples/self.oversampling) % 2 == 0:
        kx = kx - 0.5*self.kspace_spa[0]
      if self.ph_samples % 2 == 0:
        ky = ky - 0.5*self.kspace_spa[1]
      if self.slices % 2 == 0:
        kz = kz - 0.5*self.kspace_spa[2]

      # kspace times and locations
      t = np.zeros([self.ro_samples, self.ph_samples], dtype=np.float32)
      for ph in range(self.ph_samples):

        # Evaluate readout direction
        if ph % self.lines_per_shot == 0:
          ro = 1

        # Fill locations
        kspace[0][::ro,ph,:] = np.tile(kx[:,np.newaxis], [1, self.slices])
        kspace[1][::ro,ph,:] = np.tile(ky[:,np.newaxis] + self.kspace_spa[1]*ph, [1, self.slices])

        # Update timings
        if ph % self.lines_per_shot == 0:
          t[::ro,ph] = venc_time + ro_grad0.dur_ + ro_grad.slope_ + dt
        else:
          t[::ro,ph] = t[::-ro,ph-1][-1] + ro_grad.slope_ + ro_grad.slope_ + dt

        # Reverse readout
        ro = -ro

      # Fill kz coordinates
      for s in range(self.slices):
        kspace[2][:,:,s] = kz[s]

      # Calculate echo time
      self.echo_time = venc_time + ro_grad0.dur_ + 0.5*self.lines_per_shot*ro_grad.dur_

      # Send the information to each process if running in parallel
      kspace, t, local_idx = scatterKspace(kspace, t)
      self.local_idx = local_idx

      return (kspace, t)

    def plot_trajectory(self):
      # Plot kspace locations and times
      plt.figure(1)
      for i in range(int(self.points[0].shape[1]/self.lines_per_shot)):
        idx = [i*self.lines_per_shot, (i+1)*self.lines_per_shot]
        kxx = np.concatenate((np.array([0]), self.points[0][:,idx[0]:idx[1]].flatten('F')))
        kyy = np.concatenate((np.array([0]), self.points[1][:,idx[0]:idx[1]].flatten('F')))
        # kxx = self.points[0][:,idx[0]:idx[1]].flatten('F')
        # kyy = self.points[1][:,idx[0]:idx[1]].flatten('F')
        plt.plot(kxx,kyy)
      plt.xlabel('$k_x ~(1/m)$')
      plt.ylabel('$k_y ~(1/m)$')
      plt.axis('equal')

      plt.figure(2)
      im = plt.scatter(self.points[0],self.points[1],c=1000*self.times,s=1.5)
      plt.xlabel('$k_x ~(1/m)$')
      plt.ylabel('$k_y ~(1/m)$')
      cbar = plt.colorbar(im, ax=plt.gca())
      cbar.ax.tick_params(labelsize=8) 
      cbar.ax.set_title('Time [ms]',fontsize=8)
      plt.axis('equal')
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
      dt_line = (self.kspace_bw[0]*2*np.pi)/(self.gamma_*self.Gr_max_)
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
