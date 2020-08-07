import matplotlib.pyplot as plt
import numpy as np
from PyMRStrain.IO import write_vtk
from PyMRStrain.Spins import Function
from scipy.ndimage.filters import gaussian_filter


# Base class for phantoms
class PhantomBase:
  def __init__(self, spins):
    # Parameters must be defined by subclass
    self.t    = 0.0
    self.spins = spins
    self.x      = self.spins.samples
    if spins.orientation == 0:
        self.r      = np.linalg.norm(self.x[:,[1, 2]], axis=1)
        self.scos   = self.x[:,1]/self.r
        self.ssin   = self.x[:,2]/self.r
    elif spins.orientation == 1:
        self.r      = np.linalg.norm(self.x[:,[0, 2]], axis=1)
        self.scos   = self.x[:,0]/self.r
        self.ssin   = self.x[:,2]/self.r
    elif spins.orientation == 2:
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

    # Filtering
    Tp = gaussian_filter(Tp, sigma=2)
    # print('Using non-filtered time modulation for debugging')
    Tp = Tp/Tp.max()

    # Temporal resampling
    Tp = Tp[::3]

    return Tp


# Phantom Class
class Phantom(PhantomBase):
  def __init__(self, spins, p, patient=False, z_motion=True, 
              phi_en_apex=10*np.pi/180, write_vtk=False):
    super().__init__(spins)
    self.p = p
    self.patient = patient
    self.write_vtk = write_vtk
    self.z_motion = z_motion
    self.phi_en_apex = phi_en_apex

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
        z = np.copy(self.x[:,self.spins.orientation])
        z_max = z.max()
        z_min = z.min()
        z = -(z - z_max)/(z_max - z_min) # z=0 at base, z=1 at appex

        # Scale in-plane components (keeps displacement at base but
        # increases displacement at appex)
        # Obs: zero_twist controls where the transition clockwise to
        # counterclockwise twist is going to happen. If the length of
        # the long-axis is 1.0, and distances are measured from base
        # to appex, zero_twist is the distance from base where this
        # transition happens.
        # # scale = self.phi_en_apex/p.phi_en
        # # self.zero_twist = -1.0/scale/(1.0-1.0/scale)
        zero_twist = 1/(1 - self.phi_en_apex/p.phi_en)

        # Angular scaling factor defining the rotations in the Z-direction
        scale = (zero_twist-z)/zero_twist

        # Define through-plane displacement
        self.u_real[:,self.spins.orientation] = (z - 1)*0.02

    else:
        scale = 1.0


    # Get in-plane displacements
    if self.patient:
      # Abnormality
      # Phi = p.xi*0.5*(1 - (self.scos*np.cos(p.psi) + self.ssin*np.sin(p.psi)))
      Phi = p.xi*0.5*(1 - (self.scos*np.cos(p.psi) + self.ssin*np.sin(p.psi)))

      # Create displacement and velocity for patients
      if self.spins.orientation == 0:
          self.u_real[:,1] = Phi*(R_ES*(self.scos*np.cos(phi_n*scale) \
                                - self.ssin*np.sin(phi_n*scale)) - R_ED*self.scos)
          self.u_real[:,2] = Phi*(R_ES*(self.ssin*np.cos(phi_n*scale) \
                                + self.scos*np.sin(phi_n*scale)) - R_ED*self.ssin)
      elif self.spins.orientation == 1:
          self.u_real[:,0] = Phi*(R_ES*(self.scos*np.cos(phi_n*scale) \
                                - self.ssin*np.sin(phi_n*scale)) - R_ED*self.scos)
          self.u_real[:,2] = Phi*(R_ES*(self.ssin*np.cos(phi_n*scale) \
                                + self.scos*np.sin(phi_n*scale)) - R_ED*self.ssin)
      elif self.spins.orientation == 2:
          self.u_real[:,0] = Phi*(R_ES*(self.scos*np.cos(phi_n*scale) \
                                - self.ssin*np.sin(phi_n*scale)) - R_ED*self.scos)
          self.u_real[:,1] = Phi*(R_ES*(self.ssin*np.cos(phi_n*scale) \
                                + self.scos*np.sin(phi_n*scale)) - R_ED*self.ssin)

    else:
      # Create displacement and velocity for volunteers
      if self.spins.orientation == 0:
          self.u_real[:,1] = (R_ES*(self.scos*np.cos(phi_n*scale) \
                                - self.ssin*np.sin(phi_n*scale)) - R_ED*self.scos)
          self.u_real[:,2] = (R_ES*(self.ssin*np.cos(phi_n*scale) \
                                + self.scos*np.sin(phi_n*scale)) - R_ED*self.ssin)
      elif self.spins.orientation == 1:
          self.u_real[:,0] = (R_ES*(self.scos*np.cos(phi_n*scale) \
                                - self.ssin*np.sin(phi_n*scale)) - R_ED*self.scos)
          self.u_real[:,2] = (R_ES*(self.ssin*np.cos(phi_n*scale) \
                                + self.scos*np.sin(phi_n*scale)) - R_ED*self.ssin)
      elif self.spins.orientation == 2:
          self.u_real[:,0] = (R_ES*(self.scos*np.cos(phi_n*scale) \
                                - self.ssin*np.sin(phi_n*scale)) - R_ED*self.scos)
          self.u_real[:,1] = (R_ES*(self.ssin*np.cos(phi_n*scale) \
                                + self.scos*np.sin(phi_n*scale)) - R_ED*self.ssin)

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
        write_vtk([self.u], path="output/u_{:04d}.vtk".format(i), name=['u'])

    return self.u

  # Velocity
  def velocity(self, i):
    self.get_data(i)
    return self.u, self.v


# PCSPAMM phantom
class PCSPAMM_Phantom(PhantomBase):
  def __init__(self, spins, p, z_motion=True, write_vtk=False):
    super().__init__(spins)
    self.p = p
    self.write_vtk = write_vtk
    self.z_motion = z_motion

  def Fluid_TemporalModulation(self, t, tA, tB, tC):
    return t

  def get_data(self, i):

    # Shorthand notation for the parameters
    p = self.p

    # Time steps
    # t = np.linspace(p.t_end/p['time_steps'],p.t_end,p['time_steps'])
    t = np.linspace(0.0, p.t_end, p.time_steps+1)

    # Ventricle region
    fluid = self.spins.regions[:,0]

    # Modulation function
    dt = 1e-08

    # Displacements at different time-steps
    g  = self.Fluid_TemporalModulation(t, p.tA, p.tB, p.tC)
    # self.u.vector()[:] = g[i]*self.u_real
    # # # self.u.vector()[:] = g[i]*(np.multiply(self.u_real,f.vector()))
    # # # write_scalar_vtk(self.u, path='output/u.vtk', name='u')

    # Add helicity pattern
    factor = 0.1*(1-np.exp(-(self.p.R_en-self.r[fluid])/self.p.R_en))
    if self.spins.orientation == 0:
        self.u_real[fluid,1] = ((self.scos[fluid]*np.cos(factor) \
                              - self.ssin[fluid]*np.sin(factor)) - self.scos[fluid])
        self.u_real[fluid,2] = ((self.ssin[fluid]*np.cos(factor) \
                              + self.scos[fluid]*np.sin(factor)) - self.ssin[fluid])
    elif self.spins.orientation == 1:
        self.u_real[fluid,0] = ((self.scos[fluid]*np.cos(factor) \
                              - self.ssin[fluid]*np.sin(factor)) - self.scos[fluid])
        self.u_real[fluid,2] = ((self.ssin[fluid]*np.cos(factor) \
                              + self.scos[fluid]*np.sin(factor)) - self.ssin[fluid])
    elif self.spins.orientation == 2:
        self.u_real[fluid,0] = ((self.scos[fluid]*np.cos(factor) \
                              - self.ssin[fluid]*np.sin(factor)) - self.scos[fluid])
        self.u_real[fluid,1] = ((self.ssin[fluid]*np.cos(factor) \
                              + self.scos[fluid]*np.sin(factor)) - self.ssin[fluid])
    self.u.vector()[:] = g[i]*self.u_real

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
        write_vtk([self.u], path="output/u_{:04d}.vtk".format(i), name=['u'])

    return self.u

  # Velocity
  def velocity(self, i):
    self.get_data(i)
    return self.u, self.v