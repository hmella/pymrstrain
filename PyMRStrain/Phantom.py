from PyMRStrain.FiniteElement import *
from PyMRStrain.FunctionSpace import *
from PyMRStrain.Function import *
from PyMRStrain.Mesh import *
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from PyMRStrain.IO import write_vtk

# NOTE: 1 [cycle] = 2*np.pi [rad]

class Phantom3D(object):
  def __init__(self, function_space=None, time_steps=30, path=None):
    self.V = function_space                       # Function space
    self.mesh = function_space.mesh()             # Mesh
    self._time_steps = time_steps                 # Number of time steps
    self.path = path
    self.t = 0.0

  def displacement(self, i):
    # Function
    u = Function(self.V)

    # Load data
    u.vector()[:] = Mesh(self.path+'u_{:04d}.msh'.format(i)).mesh.point_data['u'].flatten()

    return u

  def velocity(self, i):
    # Function
    v = Function(self.V)

    # Load data
    v.vector()[:] = Mesh(self.path+'v_{:04d}.msh'.format(i)).mesh.point_data['v'].flatten()

    return self.displacement(i), v

#########################################################
# Base class for phantoms
class PhantomBase:
  def __init__(self, mesh, V):
    # Parameters must be defined by subclass
    self.mesh = mesh
    self.V    = V
    self.t    = 0.0
    self.dofmap = self.V.vertex_to_dof_map()
    self.x      = np.copy(self.V.dof_coordinates()[self.dofmap[0::self.V.element_shape()[0]]])
    self.r      = np.copy(np.sqrt(np.power(self.x[:,0], 2) + np.power(self.x[:,1], 2)))
    self.cos_   = np.copy(self.x[:,0]/self.r)
    self.sin_   = np.copy(self.x[:,1]/self.r)

    # Volunteers
    self.u  = Function(self.V)
    self.v  = Function(self.V)
    self.u_real = self.u.copy_values()

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
class DefaultPhantom(PhantomBase):
  def __init__(self, p, function_space=None, t=0.0, T=1.0, time_steps=30, patient=False):

    # Init parameters
    self.sigma = p['sigma']
    self.R_en  = p['R_en']
    self.R_ep  = p['R_ep']
    self.tA    = p['tA']
    self.tB    = p['tB']
    self.tC    = p['tC']
    self.S_en  = p['S_en']
    self.S_ar  = p['S_ar']
    self.phi_en = p['phi_en']
    self.phi_ep = p['phi_ep']
    self.psi = p['psi']
    self.xi  = p['xi']
    self.patient = patient
    self.T = T
    self.time_steps = p["time_steps"]

    # Init parameters from base class
    super().__init__(function_space.mesh(), function_space)

  def get_data(self, i):

    # Time steps
    # t = np.linspace(self.T/self.time_steps,self.T,self.time_steps)
    t = np.linspace(0.0,self.T,self.time_steps+1)

    # Component dofmaps
    dofmap_x = self.dofmap[0::self.V.element_shape()[0]]
    dofmap_y = self.dofmap[1::self.V.element_shape()[0]]

    # Create base displacement
    mu = (self.R_ep - self.r)/(self.R_ep - self.R_en)
    mu = mu - mu.min()
    mu = mu/mu.max()
    # mu[mu < 0.0] = 0.0
    # mu[mu > 1.0] = 1.0
    mu = np.power(mu, self.sigma)

    ##########################
    # plt.scatter(self.x[:,0],self.x[:,1],c=mu,cmap="jet",edgecolors=None,marker='.')
    # plt.colorbar()
    # plt.show()

    # End-systolic endocardial displacement
    d_en  = (1 - self.S_en)*self.R_en

    # End-systolic epicardial displacement
    d_ep  = self.R_ep - ((self.R_ep**2 - self.R_en**2)/self.S_ar + (self.R_en - d_en)**2)**0.5

    # End systolic radial displcements
    phi_n = ((1 - mu)*self.phi_ep + mu*self.phi_en)*np.sin(2*np.pi/self.T*t[i])
    d_n   = (1 - mu)*d_ep + mu*d_en

    # End-diastolic and end-systolic radius
    R_ED = self.r
    R_ES = self.r - d_n

    if self.patient:
      # Abnormality
      # Phi = 0.5*self.xi*(1.0 - (self.cos_*np.cos(self.psi) + self.sin_*np.sin(self.psi)))
      Phi = 0.5*self.xi*(0.5 - 0.5*(self.cos_*np.cos(self.psi) + self.sin_*np.sin(self.psi))) + 0.5

      # Create displacement and velocity for patients
      self.u_real[dofmap_x] = Phi*(R_ES*(self.cos_*np.cos(phi_n) \
                            - self.sin_*np.sin(phi_n)) - R_ED*self.cos_)
      self.u_real[dofmap_y] = Phi*(R_ES*(self.sin_*np.cos(phi_n) \
                            + self.cos_*np.sin(phi_n)) - R_ED*self.sin_)

    else:
      # Create displacement and velocity for volunteers
      self.u_real[dofmap_x] = R_ES*(self.cos_*np.cos(phi_n) \
                            - self.sin_*np.sin(phi_n)) - R_ED*self.cos_
      self.u_real[dofmap_y] = R_ES*(self.sin_*np.cos(phi_n) \
                            + self.cos_*np.sin(phi_n)) - R_ED*self.sin_

    # # # # Inclusion
    # # # f = Function(self.V)
    # # # R = np.sqrt(np.power(self.x[:,0]-0.4*(self.R_en+self.R_ep),2) + np.power(self.x[:,1],2))
    # # # s = (self.R_ep-self.R_en)
    # # # f.vector()[dofmap_x] = (1-0.55*np.exp(-np.power(R/s,2)))
    # # # f.vector()[dofmap_y] = (1-0.55*np.exp(-np.power(R/s,2)))
    # # # write_scalar_vtk(f, path='output/f.vtk', name='f')

    # Modulation function
    dt = 1e-08

    # Displacements at different time-steps
    g  = self.TemporalModulation(t, self.tA, self.tB, self.tC)
    self.u.vector()[:] = g[i]*self.u_real
    # # # self.u.vector()[:] = g[i]*(np.multiply(self.u_real,f.vector()))
    # # # write_scalar_vtk(self.u, path='output/u.vtk', name='u')

    # Velocity at different time-steps
    dgdt = - self.TemporalModulation(t + 2*dt, self.tA, self.tB, self.tC) \
         + 8*self.TemporalModulation(t + dt, self.tA, self.tB, self.tC) \
         - 8*self.TemporalModulation(t - dt, self.tA, self.tB, self.tC) \
         + self.TemporalModulation(t - 2*dt, self.tA, self.tB, self.tC)
    dgdt /= 12*dt
    self.v.vector()[:] = dgdt[i]*self.u_real

  # Displacement
  def displacement(self, i):
    self.get_data(i)
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
    self.cos_ = self.X/self.r
    self.sin_ = self.Y/self.r

    # Volunteers
    self.u  = np.zeros(np.append(self.X.shape,2))
    self.v  = np.zeros(np.append(self.X.shape,2))
    self.u_real = np.zeros(np.append(self.X.shape,2))

    # Init parameters
    self.sigma = p['sigma']
    self.R_en  = p['R_en']
    self.R_ep  = p['R_ep']
    self.tA    = p['tA']
    self.tB    = p['tB']
    self.tC    = p['tC']
    self.S_en  = p['S_en']
    self.S_ar  = p['S_ar']
    self.phi_en = p['phi_en']
    self.phi_ep = p['phi_ep']
    self.psi = p['psi']
    self.xi  = p['xi']
    self.patient = patient
    self.T = T
    self.time_steps = p["time_steps"]
    self.image = image

  def get_data(self, i):

    # Time steps
    # t = np.linspace(self.T/self.time_steps,self.T,self.time_steps)
    t = np.linspace(0.0,self.T,self.time_steps+1)

    # Create base displacement
    mu = (self.R_ep - self.r)/(self.R_ep - self.R_en)
    # mu = mu - mu.min()
    # mu = mu/mu.max()
    # mu[np.logical_or(mu < 0.0, mu > 1.0)] = 0
    # mu = np.power(mu, self.sigma)
    # print(self.sigma)

    # #########################
    # plt.imshow(mu,cmap="jet")
    # plt.colorbar()
    # plt.show()

    # End-systolic endocardial displacement
    d_en  = (1 - self.S_en)*self.R_en

    # End-systolic epicardial displacement
    d_ep  = self.R_ep - ((self.R_ep**2 - self.R_en**2)/self.S_ar + (self.R_en - d_en)**2)**0.5

    # End systolic radial displcements
    phi_n = ((1 - mu)*self.phi_ep + mu*self.phi_en)*np.sin(2*np.pi/self.T*t[i])
    d_n   = (1 - mu)*d_ep + mu*d_en

    # End-diastolic and end-systolic radius
    R_ED = self.r
    R_ES = self.r - d_n

    if self.patient:
      # Abnormality
      # Phi = 0.5*self.xi*(1.0 - (self.cos_*np.cos(self.psi) + self.sin_*np.sin(self.psi)))
      Phi = 0.5*self.xi*(0.5 - 0.5*(self.cos_*np.cos(self.psi) + self.sin_*np.sin(self.psi))) + 0.5

      # Create displacement and velocity for patients
      self.u_real[...,0] = Phi*(R_ES*(self.cos_*np.cos(phi_n) \
                            - self.sin_*np.sin(phi_n)) - R_ED*self.cos_)
      self.u_real[...,1] = Phi*(R_ES*(self.sin_*np.cos(phi_n) \
                            + self.cos_*np.sin(phi_n)) - R_ED*self.sin_)

    else:
      # Create displacement and velocity for volunteers
      self.u_real[...,0] = R_ES*(self.cos_*np.cos(phi_n) \
                            - self.sin_*np.sin(phi_n)) - R_ED*self.cos_
      self.u_real[...,1] = R_ES*(self.sin_*np.cos(phi_n) \
                            + self.cos_*np.sin(phi_n)) - R_ED*self.sin_


    # # Inclusion
    # f = np.zeros(self.u_real.shape)
    # R = np.sqrt(np.power(self.X-0.5*(self.R_en+self.R_ep),2) + np.power(self.Y,2))
    # s = 2*(self.R_ep-self.R_en)
    # f[...,0] = (0.35*np.exp(-np.power(R/s,2)))
    # f[...,1] = (0.35*np.exp(-np.power(R/s,2)))

    # Modulation function
    dt = 1e-08

    # Displacements at different time-steps
    g  = self.TemporalModulation(t, self.tA, self.tB, self.tC)
    self.u[...] = g[i]*self.u_real
    # self.u[...] = g[i]*(self.u_real-f*self.u_real)

    # Velocity at different time-steps
    dgdt = - self.TemporalModulation(t + 2*dt, self.tA, self.tB, self.tC) \
         + 8*self.TemporalModulation(t + dt, self.tA, self.tB, self.tC) \
         - 8*self.TemporalModulation(t - dt, self.tA, self.tB, self.tC) \
         + self.TemporalModulation(t - 2*dt, self.tA, self.tB, self.tC)
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
