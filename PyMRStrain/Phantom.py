import matplotlib.pyplot as plt
import numpy as np
from PyMRStrain.Spins import Function
from PyMRStrain.IO import write_vtk
from scipy.ndimage.filters import gaussian_filter


#########################################################
# Base class for phantoms
class PhantomBase:
  def __init__(self, spins):
    # Parameters must be defined by subclass
    self.t    = 0.0
    self.spins = spins
    self.x      = self.spins.samples
    self.r      = np.linalg.norm(self.x[:,0:2], axis=1)
    self.scos   = self.x[:,0]/self.r
    self.ssin   = self.x[:,1]/self.r

    # Volunteers
    self.u  = Function(self.spins)
    self.v  = Function(self.spins)
    self.u_real = np.zeros(self.x.shape)

  # Temporal modulation function
  def TemporalModulation(self, t, tA, tB, tC):

    # time resampling
    t_tmp = np.linspace(t[0],t[-1],3*t.size)

    # modulation
    Tp = np.zeros(t_tmp.shape,dtype=t.dtype)

    for i in range(t_tmp.size):
        if t_tmp[i] < tA:
            Tp[i] = 0.0
        elif t_tmp[i] >= tA and t_tmp[i] < tB:
            Tp[i] = 1.005 - 1.005*np.exp(-5*(t_tmp[i] - tA)/(tB-tA))
        elif t_tmp[i] >= tB and t_tmp[i] < tC:
            Tp[i] = 1.0
        else:
            Tp[i] = np.exp(-11*(t_tmp[i] - tC))

    # # filtering
    Tp = gaussian_filter(Tp, sigma=3)
    Tp = Tp/Tp.max()

    # modulation resampling
    Tp = Tp[::3]

    # # # # modulation
    # # # Tp = np.zeros(t.shape,dtype=t.dtype)

    # # # for i in range(t.size):
    # # #   Tp[i] = t[i]/1.0

    return Tp


#########################################################



#########################################################
# Phantom Class
class Phantom(PhantomBase):
  def __init__(self, spins, p, patient=False, z_motion=True, write_vtk=False):
    super().__init__(spins)
    self.p = p
    self.patient = patient
    self.write_vtk = write_vtk
    self.z_motion = z_motion

  def get_data(self, i):

    # Shorthand notation for the parameters
    p = self.p

    # Time steps
    # t = np.linspace(p.t_end/p['time_steps'],p.t_end,p['time_steps'])
    t = np.linspace(0.0, p.t_end, p.time_steps+1)

    # Ventricle region
    ventricle = self.spins.regions[:,-1]

    # Create base displacement
    mu = (p.R_ep - self.r)/(p.R_ep - p.R_en)
    mu[ventricle] = mu[ventricle] - mu[ventricle].min()
    mu[ventricle] = mu[ventricle]/mu[ventricle].max()
    mu[ventricle] = np.power(mu[ventricle], p.sigma)

    ##########################
    # plt.scatter(self.x[:,0],self.x[:,1],c=mu,cmap="jet",edgecolors=None,marker='.')
    # plt.colorbar()
    # plt.show()

    # End-systolic endocardial displacement
    d_en  = (1 - p.S_en)*p.R_en

    # End-systolic epicardial displacement
    d_ep  = p.R_ep - ((p.R_ep**2 - p.R_en**2)/p.S_ar
          + (p.R_en - d_en)**2)**0.5

    # End systolic radial displcements
    phi_n = ((1 - mu)*p.phi_ep + mu*p.phi_en)*np.sin(2*np.pi/p.t_end*t[i])
    d_n   = (1 - mu)*d_ep + mu*d_en

    # End-diastolic and end-systolic radius
    R_ED = self.r
    R_ES = self.r - d_n

    # If the phantom is 3D add longitudinal displacement
    # and angle scaling factor
    if self.z_motion:
        # Normalize coordinates
        z = np.copy(self.x[:,2])
        z_max = z.max()
        z_min = z.min()
        z = -(z - z_max)/(z_max - z_min) # z=0 at base, z=1 at appex

        # Scale in-plane components (keeps displacement at base but
        # increases displacement at appex)
        scale = (0.35-z)/0.35

        # Define through-plane displacement
        self.u_real[:,2] = (z - 1)*0.02

    else:
        scale = 1.0


    # Get in-plane displacements
    if self.patient:
      # Abnormality
      Phi = p.xi*(1 - (self.scos*np.cos(p.psi) + self.ssin*np.sin(p.psi)))

      # Create displacement and velocity for patients
      self.u_real[:,0] = Phi*(R_ES*(self.scos*np.cos(phi_n*scale) \
                            - self.ssin*np.sin(phi_n*scale)) - R_ED*self.scos)
      self.u_real[:,1] = Phi*(R_ES*(self.ssin*np.cos(phi_n*scale) \
                            + self.scos*np.sin(phi_n*scale)) - R_ED*self.ssin)

    else:
      # Create displacement and velocity for volunteers
      self.u_real[:,0] = R_ES*(self.scos*np.cos(phi_n*scale) \
                            - self.ssin*np.sin(phi_n*scale)) - R_ED*self.scos
      self.u_real[:,1] = R_ES*(self.ssin*np.cos(phi_n*scale) \
                            + self.scos*np.sin(phi_n*scale)) - R_ED*self.ssin

    # # # # Inclusion
    # # # f = Function(self.V)
    # # # R = np.sqrt(np.power(self.x[:,0]-0.4*(p.R_en+p.R_ep),2) + np.power(self.x[:,1],2))
    # # # s = (p.R_ep-p.R_en)
    # # # f.vector()[dofmap_x] = (1-0.55*np.exp(-np.power(R/s,2)))
    # # # f.vector()[dofmap_y] = (1-0.55*np.exp(-np.power(R/s,2)))
    # # # write_scalar_vtk(f, path='output/f.vtk', name='f')

    # Modulation function
    dt = 1e-08

    # Displacements at different time-steps
    g  = self.TemporalModulation(t, p.tA, p.tB, p.tC)
    self.u.vector()[:] = g[i]*self.u_real
    # # # self.u.vector()[:] = g[i]*(np.multiply(self.u_real,f.vector()))
    # # # write_scalar_vtk(self.u, path='output/u.vtk', name='u')

    # Velocity at different time-steps
    dgdt = - self.TemporalModulation(t + 2*dt, p.tA, p.tB, p.tC) \
         + 8*self.TemporalModulation(t + dt, p.tA, p.tB, p.tC) \
         - 8*self.TemporalModulation(t - dt, p.tA, p.tB, p.tC) \
         + self.TemporalModulation(t - 2*dt, p.tA, p.tB, p.tC)
    dgdt /= 12*dt
    self.v.vector()[:] = dgdt[i]*self.u_real

  # Displacement
  def displacement(self, i):
    self.get_data(i)

    # Export generated displacement field
    if self.write_vtk:
        write_vtk(self.u, path="output/u_{:04d}.vtk".format(i), name='u')

    return self.u

  # Velocity
  def velocity(self, i):
    self.get_data(i)
    return self.u, self.v


# Pixelled Phantom Class
class PixelledPhantom(PhantomBase):
  def __init__(self, p, image=None, t=0.0, T=1.0, time_steps=30, patient=False):

    self.t    = 0.0
    self.X    = image._grid[0]
    self.Y    = image._grid[1]
    self.r    = np.sqrt(np.power(self.X, 2) + np.power(self.Y, 2))
    self.r[np.isnan(self.r)] = 1
    self.scos = self.X/self.r
    self.ssin = self.Y/self.r

    # Volunteers
    self.u  = np.zeros(np.append(self.X.shape,2))
    self.v  = np.zeros(np.append(self.X.shape,2))
    self.u_real = np.zeros(np.append(self.X.shape,2))
    self.patient = patient
    self.image = image
    self.p = p

  def get_data(self, i):

    # Parameters
    p = self.p

    # Time steps
    # t = np.linspace(p.t_end/p['time_steps'],p.t_end,p['time_steps'])
    t = np.linspace(0.0,p.t_end,p['time_steps']+1)

    # Create base displacement
    mu = (p.R_ep - self.r)/(p.R_ep - p.R_en)
    # mu = mu - mu.min()
    # mu = mu/mu.max()
    # mu[np.logical_or(mu < 0.0, mu > 1.0)] = 0
    # mu = np.power(mu, p.sigma)
    # print(p.sigma)

    # #########################
    # plt.imshow(mu,cmap="jet")
    # plt.colorbar()
    # plt.show()

    # End-systolic endocardial displacement
    d_en  = (1 - p.S_en)*p.R_en

    # End-systolic epicardial displacement
    d_ep  = p.R_ep - ((p.R_ep**2 - p.R_en**2)/p.S_ar + (p.R_en - d_en)**2)**0.5

    # End systolic radial displcements
    phi_n = ((1 - mu)*p.phi_ep + mu*p.phi_en)*np.sin(2*np.pi/p.t_end*t[i])
    d_n   = (1 - mu)*d_ep + mu*d_en

    # End-diastolic and end-systolic radius
    R_ED = self.r
    R_ES = self.r - d_n

    if self.patient:
      # Abnormality
      # Phi = 0.5*p.xi*(1.0 - (self.scos*np.cos(p.psi) + self.ssin*np.sin(p.psi)))
      Phi = 0.5*p.xi*(0.5 - 0.5*(self.scos*np.cos(p.psi) + self.ssin*np.sin(p.psi))) + 0.5

      # Create displacement and velocity for patients
      self.u_real[...,0] = Phi*(R_ES*(self.scos*np.cos(phi_n) \
                            - self.ssin*np.sin(phi_n)) - R_ED*self.scos)
      self.u_real[...,1] = Phi*(R_ES*(self.ssin*np.cos(phi_n) \
                            + self.scos*np.sin(phi_n)) - R_ED*self.ssin)

    else:
      # Create displacement and velocity for volunteers
      self.u_real[...,0] = R_ES*(self.scos*np.cos(phi_n) \
                            - self.ssin*np.sin(phi_n)) - R_ED*self.scos
      self.u_real[...,1] = R_ES*(self.ssin*np.cos(phi_n) \
                            + self.scos*np.sin(phi_n)) - R_ED*self.ssin


    # # Inclusion
    # f = np.zeros(self.u_real.shape)
    # R = np.sqrt(np.power(self.X-0.5*(p.R_en+p.R_ep),2) + np.power(self.Y,2))
    # s = 2*(p.R_ep-p.R_en)
    # f[...,0] = (0.35*np.exp(-np.power(R/s,2)))
    # f[...,1] = (0.35*np.exp(-np.power(R/s,2)))

    # Modulation function
    dt = 1e-08

    # Displacements at different time-steps
    g  = TemporalModulation(t, p.tA, p.tB, p.tC)
    self.u[...] = g[i]*self.u_real
    # self.u[...] = g[i]*(self.u_real-f*self.u_real)

    # Velocity at different time-steps
    dgdt = - TemporalModulation(t + 2*dt, p.tA, p.tB, p.tC) \
         + 8*TemporalModulation(t + dt, p.tA, p.tB, p.tC) \
         - 8*TemporalModulation(t - dt, p.tA, p.tB, p.tC) \
         + TemporalModulation(t - 2*dt, p.tA, p.tB, p.tC)
    dgdt /= 12*dt
    self.v[:] = dgdt[i]*self.u_real

  # Displacement
  def displacement(self, i):
    self.get_data(i)
    return self.u

  # Velocity
  def velocity(self, i):
    self.get_data(i)
    return self.u, self.v

  # # Temporal modulation function
  # def TemporalModulation(self, t, tA, tB, tC):

  #   # # # time resampling
  #   # # t_tmp = np.linspace(t[0],t[-1],3*t.size)

  #   # # # modulation
  #   # # Tp = np.zeros(t_tmp.shape,dtype=t.dtype)

  #   # # for i in range(t_tmp.size):
  #   # #     if t_tmp[i] < tA:
  #   # #         Tp[i] = 0.0
  #   # #     elif t_tmp[i] >= tA and t_tmp[i] < tB:
  #   # #         Tp[i] = 1.005 - 1.005*np.exp(-5*(t_tmp[i] - tA)/(tB-tA))
  #   # #     elif t_tmp[i] >= tB and t_tmp[i] < tC:
  #   # #         Tp[i] = 1.0
  #   # #     else:
  #   # #         Tp[i] = np.exp(-11*(t_tmp[i] - tC))

  #   # # # filtering
  #   # # # Tp = gaussian_filter(Tp, sigma=3)
  #   # # Tp = Tp/Tp.max()

  #   # # # modulation resampling
  #   # # Tp = Tp[::3]

  #   # modulation
  #   Tp = np.zeros(t.shape,dtype=t.dtype)

  #   for i in range(t.size):
  #     Tp[i] = t[i]/1.0

  #   return Tp

  # Temporal modulation function
  def TemporalModulation(self, t, tA, tB, tC):

    # time resampling
    t_tmp = np.linspace(t[0],t[-1],3*t.size)

    # modulation
    Tp = np.zeros(t_tmp.shape,dtype=t.dtype)

    for i in range(t_tmp.size):
        if t_tmp[i] < tA:
            Tp[i] = 0.0
        elif t_tmp[i] >= tA and t_tmp[i] < tB:
            Tp[i] = 1.005 - 1.005*np.exp(-5*(t_tmp[i] - tA)/(tB-tA))
        elif t_tmp[i] >= tB and t_tmp[i] < tC:
            Tp[i] = 1.0
        else:
            Tp[i] = np.exp(-11*(t_tmp[i] - tC))

    # filtering
    Tp = gaussian_filter(Tp, sigma=3)
    Tp = Tp/Tp.max()

    # modulation resampling
    Tp = Tp[::3]

    # # # # modulation
    # # # Tp = np.zeros(t.shape,dtype=t.dtype)

    # # # for i in range(t.size):
    # # #   Tp[i] = t[i]/1.0

    return Tp
