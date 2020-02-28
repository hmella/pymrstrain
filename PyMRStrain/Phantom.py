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
    self.r      = np.linalg.norm(self.x[:,0:2], axis=1)
    self.scos   = self.x[:,0]/self.r
    self.ssin   = self.x[:,1]/self.r

    # Volunteers
    self.u  = Function(self.spins)
    self.v  = Function(self.spins)
    self.u_real = np.zeros(self.x.shape)

  # Temporal modulation function
  def TemporalModulation(self, t, tA, tB, tC):

    # # # # time resampling
    # # # t_tmp = np.linspace(t[0],t[-1],3*t.size)

    # # # # modulation
    # # # Tp = np.zeros(t_tmp.shape,dtype=t.dtype)

    # # # for i in range(t_tmp.size):
    # # #     if t_tmp[i] < tA:
    # # #         Tp[i] = 0.0
    # # #     elif t_tmp[i] >= tA and t_tmp[i] < tB:
    # # #         Tp[i] = 1.005 - 1.005*np.exp(-5*(t_tmp[i] - tA)/(tB-tA))
    # # #     elif t_tmp[i] >= tB and t_tmp[i] < tC:
    # # #         Tp[i] = 1.0
    # # #     else:
    # # #         Tp[i] = np.exp(-11*(t_tmp[i] - tC))

    # # # # # filtering
    # # # Tp = gaussian_filter(Tp, sigma=3)
    # # # Tp = Tp/Tp.max()

    # # # # modulation resampling
    # # # Tp = Tp[::3]

    # modulation
    Tp = np.zeros(t.shape,dtype=t.dtype)

    for i in range(t.size):
      Tp[i] = t[i]/1.0

    return Tp


# Phantom Class
class Phantom(PhantomBase):
  def __init__(self, spins, p, patient=False, z_motion=True, zero_twist=0.35,
              write_vtk=False, add_inclusion=False):
    super().__init__(spins)
    self.p = p
    self.patient = patient
    self.write_vtk = write_vtk
    self.z_motion = z_motion
    self.zero_twist = zero_twist
    self.add_inclusion = add_inclusion

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
        # Obs: zero_twist controls where the transition clockwise to
        # counterclockwise twist is going to happen. If the length of
        # the long-axis is 1.0, and distances are measured from base
        # to appex, zero_twist is the distance from base where this
        # transition happens.
        scale = (self.zero_twist-z)/self.zero_twist

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

    # Modulation function
    dt = 1e-08

    # Displacements at different time-steps
    g  = self.TemporalModulation(t, p.tA, p.tB, p.tC)
    self.u.vector()[:] = g[i]*self.u_real

    # Inclusion
    if self.add_inclusion:
        f = np.zeros(self.u_real.shape)
        R = np.sqrt(np.power(self.x[:,0]-0.4*(p.R_en+p.R_ep),2) + np.power(self.x[:,1],2))
        s = (p.R_ep-p.R_en)
        f[:,0] = (1-0.55*np.exp(-np.power(R/s,2)))
        f[:,1] = (1-0.55*np.exp(-np.power(R/s,2)))
        # self.u.vector()[:] = g[i]*(np.multiply(self.u_real,f))
        self.u.vector()[:] *= f

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
