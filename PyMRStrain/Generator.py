from PyMRStrain.Geometry import *
from PyMRStrain.FiniteElement import *
from PyMRStrain.FunctionSpace import *
from PyMRStrain.Function import *
from PyMRStrain.Image import *
from PyMRStrain.Math import *
from PyMRStrain.MPIUtilities import *
from ImageUtilities import *
from SpinBasedutils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal

###################
# Common Generator
###################
class Generator:
  def __init__(self, parameters, Image, phantom=None, debug=False, fem=True):
    self.p       = parameters
    self.Image   = Image
    self.phantom = phantom
    self.debug   = debug
    self.fem     = fem

  # Ventricle and image meshes
  def ventricle_mesh(self):
    mesh = fem_ventricle_geometry(self.p["R_en"], self.p["tau"], self.p["mesh_resolution"])
    return mesh

  # Obtain image
  def get_image(self):
    if self.Image.technique is "Tagging":
      if isinstance(self.Image,Image):
        o_image = get_tagging_image(self.Image, self.phantom, self.p, self.debug)
      else:
        o_image = get_cspamm_image(self.Image, self.phantom, self.p, self.debug, self.fem)
    elif self.Image.technique is "DENSE":
      if isinstance(self.Image,Image):
        o_image = get_dense_image(self.Image, self.phantom, self.p, self.debug, self.fem)
      else:
        o_image = get_complementary_dense_image(self.Image, self.phantom, self.p, self.debug, self.fem)
    elif self.Image.technique is "PCSPAMM":
      o_image = get_PCSPAMM_image(self.Image, self.phantom, self.p, self.debug)
    elif self.Image.technique is "EXACT":
      o_image = get_exact_image(self.Image, self.phantom, self.p, self.debug)
    elif self.Image.technique is "SINE":
      o_image = get_sine_image(self.Image, self.phantom, self.p, self.debug)
    return o_image

# Function space of FE representation of image
def _ventricle_space(mesh, image):
  if image.technique in ['DENSE']:
    FE  = VectorElement(mesh.cell_type())
  if image.technique in ['Tagging','PCSPAMM','EXACT','SINE']:
    FE  = VectorElement(mesh.cell_type())
  V = FunctionSpace(mesh, FE)
  return V


#######################################
#   SINE Images
#######################################
# SINE image
def get_sine_image(image, phantom, parameters, debug, fem=False):
  """ Generate tagging images
  """
  # Time-steping parameters
  t     = parameters["t"]
  t_end = parameters["t_end"]
  n     = parameters["time_steps"]
  dt    = t_end/n

  # Output image
  o_image = np.zeros(np.append(image.array_resolution, [image.type_dim(), n+1]), dtype=complex)
  mask    = np.zeros(np.append(image.array_resolution, n+1), dtype=np.int16)

  # Sequence parameters
  T1    = image.T1                     # relaxation
  ke    = image.encoding_frequency     # Encoding frequency
  M0    = 1.0                          # Thermal equilibrium magnetization

  # Encoding frequency
  if image.taglines is None:
    ke = image.encoding_frequency
  else:
    FOV = image._FOV
    N   = image.taglines
    ke = [2.0*np.pi/(FOV[i]/N[i]) for i in range(len(N))]

  # Check encoding direction
  if image.encoding_direction is None:
    edir = np.array([i for i in range(len(ke))])
  else:
    edir = image.encoding_direction

  # Magnetization expression
  Mxy = lambda mask, M0, t, T1, tagline: np.multiply(M0*np.exp(-t/T1)*tagline,mask)

  # Spamm functions (TODO: add sanity check for non-encoded directions)
  if fem:
    V = _ventricle_space(phantom.mesh, image)
    x = V.dof_coordinates()
    spamm_fem = Function(V)
  else:
    [X,Y] = image._grid
    taglines = np.zeros(np.append(X.shape,2))
    spamm = lambda omega, X: np.cos(omega*X)

  # SPAMM modulation
  d = image.geometric_dimension()
  if fem:
    for k in range(len(ke)):
      spamm_fem.vector()[k::d] = np.cos(ke[k]*x[0::d,edir[k]])

  # Time stepping
  for i in range(n+1):

    if rank==0 and debug: print("- Time: {:.2f}".format(t))

    # Get displacements in the reference frame
    u = phantom.displacement(i)

    # Get coordinates in spatial config
    if not fem:
      x = interpolate.griddata(((X+u[...,0]).ravel(),(Y+u[...,1]).ravel()),X.ravel(),(X,Y))
      y = interpolate.griddata(((X+u[...,0]).ravel(),(Y+u[...,1]).ravel()),Y.ravel(),(X,Y))

    # Project to fem-image in the deformed frame
    if fem:
      taglines, m = _project2image_vector(u, spamm_fem, image, parameters["mesh_resolution"])
    else:
      taglines[...,0] = spamm(ke[0],x)
      taglines[...,1] = spamm(ke[1],y)
      taglines[np.isnan(taglines)] = 0

    # Normalize mask
    if fem:
      m = m >= 1.0
    else:
      r = np.sqrt(np.power(X,2)+np.power(Y,2))
      R_ep = parameters["R_ep"]
      R_en = parameters["R_en"]
      sigma = parameters["sigma"]
      mu = (R_ep - r)/(R_ep - R_en)
      mu = np.power(mu, sigma)
      mu = interpolate.griddata(((X+u[...,0]).ravel(),(Y+u[...,1]).ravel()),mu.ravel(),(X,Y))
      mu[np.isnan(mu)] = -1e-04
      mu[np.logical_or(mu>1,mu<0)] = np.nan
      m = ~np.isnan(mu)

    # Complex magnetization data
    for j in range(o_image.shape[-2]):
      o_image[...,j,i] =  Mxy(m, M0, t, T1, taglines[...,j])

    # Save mask
    mask[...,i] = m

    # Update time
    t += dt

  return o_image, mask


#######################################
#   SPAMM Images
#######################################
# SPAMM image
def get_tagging_image(image, phantom, parameters, debug=False):
  """ Generate tagging images
  """
  # Time-steping parameters
  t     = parameters["t"]
  dt    = parameters["dt"]
  t_end = parameters["t_end"]
  n     = parameters["time_steps"]

  # Output image
  o_image = np.zeros(np.append(image.array_resolution, [image.type_dim(), n]), dtype=complex)
  mask    = np.zeros(np.append(image.array_resolution, n), dtype=np.int16)

  # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # Flip angle
  beta  = image.encoding_angle          # Encoding angle
  ke    = image.encoding_frequency     # Encoding frequency
  M0    = 1.0                          # Thermal equilibrium magnetization
  M     = M0

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
    alpha = alpha*np.ones([n],dtype=np.float)

  # Encoding frequency
  if image.taglines is None:
    ke = image.encoding_frequency
  else:
    FOV = image._FOV
    N   = image.taglines
    ke = [2.0*np.pi/(FOV[i]/N[i]) for i in range(len(N))]

  # Check encoding direction
  if image.encoding_direction is None:
    edir = np.array([i for i in range(len(ke))])
  else:
    edir = image.encoding_direction

  # Check complementary acquisition
  if image.complementary:
    comp = -1.0
  else:
    comp = 1.0

  # Magnetization expression
  Mxy = lambda i, mask, M, M0, alpha, prod, beta, t, T1, tagline: mask*(
                                      M0*np.cos(beta)**2*np.exp(-t/T1) - \
                                      comp*M0*np.sin(beta)**2*np.exp(-t/T1)*tagline + \
                                      M0*(1 - np.exp(-t/T1)))*prod*np.sin(alpha)

  # Spamm functions (TODO: add sanity check for non-encoded directions)
  V = _ventricle_space(phantom.mesh, image)
  x = V.dof_coordinates()
  spamm_fem = Function(V)

  # SPAMM modulation
  prod = 1
  d = image.geometric_dimension()
  for k in range(len(ke)):
    spamm_fem.vector()[k::d] = np.cos(ke[k]*x[0::d,edir[k]])

  # Time stepping
  for i in range(n):

    # Update time
    t += dt

    if rank==0 and debug: print("- Time: {:.2f}".format(t))

    # Get displacements in the reference frame
    u = phantom.displacement(i)

    # Project to fem-image in the deformed frame
    taglines, m = _project2image_vector(u, spamm_fem, image, parameters["mesh_resolution"])

    # Normalize mask
    m = m >= 1.0

    # Complex magnetization data
    for j in range(o_image.shape[-2]):
      o_image[...,j,i] =  Mxy(i+1, m, M, M0, alpha[i], prod, beta, t, T1, taglines[...,j])

    # Save mask
    mask[...,i] = m

    # Flip angles product
    prod = prod*np.cos(alpha[i])

  return o_image, mask

# complementary SPAMM images
def get_cspamm_image(image, phantom, parameters, debug, fem):
  """ Generate tagging images
  """
  # Time-steping parameters
  t     = parameters["t"]
  t_end = parameters["t_end"]
  n     = parameters["time_steps"]
  dt    = t_end/n

  # Output image
  size = np.append(image.array_resolution, [image.type_dim(), n])
  image_0 = np.zeros(size, dtype=complex)
  image_1 = np.zeros(size, dtype=complex)
  mask    = np.zeros(np.append(image.array_resolution, n), dtype=np.int16)

  # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # Flip angle
  beta  = image.encoding_angle          # Encoding angle
  ke    = image.encoding_frequency     # Encoding frequency
  M0    = 1.0                          # Thermal equilibrium magnetization
  M     = M0

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
    alpha = alpha*np.ones([n],dtype=np.float)

  # Encoding frequency
  if image.taglines is None:
    ke = image.encoding_frequency
  else:
    FOV = image._FOV
    N   = image.taglines
    ke = [2.0*np.pi/(FOV[i]/N[i]) for i in range(len(N))]

  # Check encoding direction
  if image.encoding_direction is None:
    edir = np.array([i for i in range(len(ke))])
  else:
    edir = image.encoding_direction

  # Magnetization expression
  Mxy0 = lambda mask, M, M0, alpha, prod, beta, t, T1, tagline, phi: mask*(
                                      M0*np.cos(beta)**2*np.exp(-t/T1) - \
                                      M0*np.sin(beta)**2*np.exp(-t/T1)*tagline + \
                                      M0*(1 - np.exp(-t/T1)))*prod*np.sin(alpha)*np.exp(1j*phi)
  Mxy1 = lambda mask, M, M0, alpha, prod, beta, t, T1, tagline, phi: mask*(
                                      M0*np.cos(beta)**2*np.exp(-t/T1) + \
                                      M0*np.sin(beta)**2*np.exp(-t/T1)*tagline + \
                                      M0*(1 - np.exp(-t/T1)))*prod*np.sin(alpha)*np.exp(1j*phi)

  # Check k space bandwidth to avoid folding artifacts
  pxsz = np.array([2*np.pi/(4.5*ke[i]) for i in range(ke.size)])
  res  = np.floor(np.divide(image.FOV,pxsz)).astype(int)
  chck = [image.resolution[i] < res[i] for i in range(ke.size)]
  if np.any(chck):
    # Check if resolutions are even or odd
    for i in range(res.size):
      if (image.resolution[i] % 2 == 0) and (res[i] % 2 != 0):
        res[i] += 1
      elif (image.resolution[i] % 2 != 0) and (res[i] % 2 == 0):
        res[i] += 1

    # Create a new image object
    new_image = CSPAMMImage(FOV=image.FOV,
            resolution=res,
            encoding_frequency=ke,
            T1=T1,
            flip_angle=alpha,
            off_resonance=image.off_resonance,
            interpolation=image.interpolation)
    new_image._modified_resolution = True
    new_image._original_resolution = image.resolution
    new_image._original_grid = image._grid
    new_image._original_array_resolution = image.array_resolution
    new_image._original_voxel_size = image.voxel_size()
    input_image = new_image
  else:
    input_image = image

  # Spamm functions (TODO: add sanity check for non-encoded directions)
  if fem:
    V = _ventricle_space(phantom.mesh, image)
    x = V.dof_coordinates()
    spamm_fem = Function(V)
  else:
    [X,Y] = image._grid
    taglines = np.zeros(np.append(X.shape,2))
    spamm = lambda omega, X: np.cos(omega*X)

  # Off resonance
  d = image.geometric_dimension()
  if not image.off_resonance:
    phi = np.zeros(input_image.array_resolution)
    phi_fem = np.zeros(x[0::d,0].shape)
  else:
    phi = image.off_resonance(input_image._grid[0],input_image._grid[1])
    phi_fem = image.off_resonance(x[0::d,0],x[0::d,1])

  # SPAMM modulation
  if fem:
    for k in range(len(ke)):
      spamm_fem.vector()[k::d] = np.cos(ke[k]*x[0::d,edir[k]])
      # spamm_fem.vector()[k::d] = np.cos(ke[k]*x[0::d,edir[k]]+phi_fem)

  # T1 map
  if not isinstance(image.T1,float):
    T1_fem = Function(V)
    T1_fem.vector()[0::d] = image.T1(x[0::d,0],x[0::d,1])
  else:
    T1 = image.T1*np.ones(input_image.resolution)

  # Hamming filter to reduce Gibbs ringing artifacts
  H = signal.hamming(image.array_resolution[0])
  H = np.outer(H,H)

  # Time stepping
  prod = 1.0
  for i in range(n):

    # Update time
    t += dt

    if rank==0 and debug: print("- Time: {:.2f}".format(t))

    # Get displacements in the reference frame
    u = phantom.displacement(i)

    # Get coordinates in spatial config
    if not fem:
      x = interpolate.griddata(((X+u[...,0]).ravel(),(Y+u[...,1]).ravel()),X.ravel(),(X,Y))
      y = interpolate.griddata(((X+u[...,0]).ravel(),(Y+u[...,1]).ravel()),Y.ravel(),(X,Y))

    # Project to fem-image in the deformed frame
    if fem:
      taglines, m, original_m = _project2image_vector(u, spamm_fem, input_image, parameters["mesh_resolution"])
      if not isinstance(image.T1,float):
        T1, m, original_m = _project2image_vector(u, T1_fem, input_image, parameters["mesh_resolution"])
    else:
      taglines[...,0] = spamm(ke[0],x)
      taglines[...,1] = spamm(ke[1],y)
      taglines[np.isnan(taglines)] = 0

    # Save mask
    if np.any(chck):
      _mask = original_m
    else:
      _mask = m
    mask[...,i] = _mask

    # Normalize mask
    if fem:
      m = m >= 1.0
    else:
      r = np.sqrt(np.power(X,2)+np.power(Y,2))
      R_ep = parameters["R_ep"]
      R_en = parameters["R_en"]
      sigma = parameters["sigma"]
      mu = (R_ep - r)/(R_ep - R_en)
      mu = np.power(mu, sigma)
      mu = interpolate.griddata(((X+u[...,0]).ravel(),(Y+u[...,1]).ravel()),mu.ravel(),(X,Y))
      mu[np.isnan(mu)] = -1e-04
      mu[np.logical_or(mu>1,mu<0)] = np.nan
      m = ~np.isnan(mu)

    # Complex magnetization data
    for j in range(image_0.shape[-2]):

      # Magnetization expressions
      tmp0 =  Mxy0(m, M, M0, alpha[i], prod, beta, t, T1, taglines[...,j], phi)
      tmp1 =  Mxy1(m, M, M0, alpha[i], prod, beta, t, T1, taglines[...,j], phi)

      # Check if images should be cropped
      if np.any(chck):

        # Images size
        S = new_image.array_resolution
        s = image.array_resolution

        # fig, ax = plt.subplots(1,2)
        # im0 = ax[0].imshow(np.abs(FFT(tmp0)))
        # im1 = ax[1].imshow(H*np.abs(FFT(tmp0)[int(0.5*(S[0]-s[0])):int(0.5*(S[0]-s[0])+s[0]):1,
        #                                     int(0.5*(S[1]-s[1])):int(0.5*(S[1]-s[1])+s[1]):1]))
        # plt.show()


        # kspace cropping
        image_0[...,j,i] = iFFT(H*FFT(tmp0)[int(0.5*(S[0]-s[0])):int(0.5*(S[0]-s[0])+s[0]):1,
                                          int(0.5*(S[1]-s[1])):int(0.5*(S[1]-s[1])+s[1]):1])
        image_1[...,j,i] = iFFT(H*FFT(tmp1)[int(0.5*(S[0]-s[0])):int(0.5*(S[0]-s[0])+s[0]):1,
                                          int(0.5*(S[1]-s[1])):int(0.5*(S[1]-s[1])+s[1]):1])

      else:

        image_0[...,j,i] = tmp0
        image_1[...,j,i] = tmp1

    # Flip angles product
    prod = prod*np.cos(alpha[i])

  return image_0, image_1, mask


# Exact image
def get_exact_image(image, phantom, parameters, debug=False):
  """ Generate tagging images
  """
  # Time-steping parameters
  t     = parameters["t"]
  t_end = parameters["t_end"]
  n     = parameters["time_steps"]
  dt    = t_end/n

  # Output image
  size = np.append(image.array_resolution, [image.type_dim(), n+1])
  image_0 = np.zeros(size, dtype=np.float)
  mask    = np.zeros(np.append(image.array_resolution, n+1), dtype=np.int16)

  # Time stepping
  for i in range(n+1):

    if rank==0 and debug: print("- Time: {:.2f}".format(t))

    # Get displacements in the reference frame
    u = phantom.displacement(i)

    # Project to fem-image in the deformed frame
    taglines, m = _project2image_vector(u, u, image, parameters["mesh_resolution"],deformed=False)

    # Complex magnetization data
    for j in range(image_0.shape[-2]):
      image_0[...,j,i] = taglines[...,j]

    # Save mask
    mask[...,i] = m

    # Update time
    t += dt

  return image_0, mask


#######################################
#   PC-SPAMM Images
#######################################
# PC-SPAMM images
def get_PCSPAMM_image(image, phantom, parameters, debug=False):
  """ Generate tagging images
  """
  # Time-steping parameters
  t     = parameters["t"]
  dt    = parameters["dt"]
  t_end = parameters["t_end"]
  n     = parameters["time_steps"]

  # Output image
  size = np.append(image.array_resolution, [image.type_dim(), n])
  image_0 = np.zeros(size, dtype=complex)
  image_1 = np.zeros(size, dtype=complex)
  mask    = np.zeros(np.append(image.array_resolution, n), dtype=np.int16)

  # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # Flip angle
  beta  = image.encoding_angle          # Encoding angle
  ke    = image.encoding_frequency     # Encoding frequency
  M0    = 1.0                          # Thermal equilibrium magnetization
  M     = M0

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
    alpha = alpha*np.ones([n],dtype=np.float)

  # Encoding frequency
  if image.taglines is None:
    ke = image.encoding_frequency
    kv = image.vel_encoding_frequency
  else:
    FOV = image._FOV
    N   = image.taglines
    ke = [2.0*np.pi/(FOV[i]/N[i]) for i in range(len(N))]

  # Check encoding direction
  if image.encoding_direction is None:
    edir = np.array([i for i in range(len(ke))])
  else:
    edir = image.encoding_direction

  # Magnetization expression
  Mxy0 = lambda i, mask, M, M0, alpha, prod, beta, t, T1, tagline, phase_velocity: mask*(
                                      M0*np.cos(beta)**2*np.exp(-t/T1)*np.exp(+1j*phase_velocity) - \
                                      M0*np.sin(beta)**2*np.exp(-t/T1)*tagline*np.exp(+1j*phase_velocity) + \
                                      M0*(1 - np.exp(-t/T1))*np.exp(+1j*phase_velocity))*prod*np.sin(alpha)
  Mxy1 = lambda i, mask, M, M0, alpha, prod, beta, t, T1, tagline, phase_velocity: mask*(
                                      M0*np.cos(beta)**2*np.exp(-t/T1)*np.exp(-1j*phase_velocity) + \
                                      M0*np.sin(beta)**2*np.exp(-t/T1)*tagline*np.exp(-1j*phase_velocity) + \
                                      M0*(1 - np.exp(-t/T1))*np.exp(-1j*phase_velocity))*prod*np.sin(alpha)

  # Spamm functions (TODO: add sanity check for non-encoded directions)
  V = _ventricle_space(phantom.mesh, image)
  x = V.dof_coordinates()
  spamm_fem = Function(V)

  # SPAMM modulation
  d = image.geometric_dimension()
  for k in range(len(ke)):
    spamm_fem.vector()[k::d] = np.cos(ke[k]*x[0::d,edir[k]])

  # Velocity encoding function
  v_encoding = Function(V)

  # Time stepping
  prod = 1
  for i in range(n):

    # Update time
    t += dt

    if rank==0 and debug: print("- Time: {:.2f}".format(t))

    # Get displacements in the reference frame
    u, v = phantom.velocity(i)

    # Phase velocity in the reference frame
    for k in range(len(kv)):
      v_encoding.vector()[k::d] = kv[k]*v.vector()[k::d]

    # Project deformed taglines to images
    taglines, m = _project2image_vector(u, spamm_fem, image, parameters["mesh_resolution"])

    # Project deformed velocity to images
    phase_v, m = _project2image_vector(u, v_encoding, image, parameters["mesh_resolution"])

    # Complex magnetization data
    for j in range(image_0.shape[-2]):
      image_0[...,j,i] =  Mxy0(i, m, M, M0, alpha, prod, beta, t, T1, taglines[...,j], phase_v[...,j])
      image_1[...,j,i] =  Mxy1(i, m, M, M0, alpha, prod,  beta, t, T1, taglines[...,j], phase_v[...,j])

    # Save mask
    mask[...,i] = m

    # Flip angles product
    prod = prod*np.cos(alpha[i])

  return image_0, image_1, mask


#######################################
#   DENSE Images
#######################################
# Generate DENSE images
def get_dense_image(image, phantom, parameters, debug, fem):
  """ Generate DENSE images
  """
  # Time-steping parameters
  t     = parameters["t"]
  dt    = parameters["dt"]
  t_end = parameters["t_end"]
  n     = parameters["time_steps"]

  # Output image
  o_image = np.zeros(np.append(image.array_resolution, [image.type_dim(), n]), dtype=complex)
  mask    = np.zeros(np.append(image.array_resolution, n), dtype=np.int16)

 # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # flip angle
  ke    = image.encoding_frequency     # encoding frequency
  M0    = 1.0                          # thermal equilibrium magnetization
  M     = M0

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
    alpha = alpha*np.ones([n],dtype=np.float)

  # Check complementary acquisition
  if image.complementary:
    comp = -1.0
  else:
    comp = 1.0

  # Magnetization expression
  Mxy = lambda mask, M, M0, alpha, prod, t, T1, ke, X, u: mask*(comp*0.5*M*np.sin(alpha)*np.exp(-t/T1)*np.exp(-1j*ke*u*mask) +
                                                          M0*np.sin(alpha)*(1 - np.exp(-t/T1))*np.exp(-1j*ke*(X+u*mask)))*prod

  # Mxy = lambda mask, M, M0, alpha, t, T1, ke, X, u: mask*(comp*0.5*M*np.sin(alpha)*np.exp(-t/T1)*np.exp(-1j*ke*u*mask) +
  #                                                         M0*np.sin(alpha)*(1 - np.exp(-t/T1))*np.exp(-1j*ke*(X+u*mask)) +
  #                                                         comp*0.5*M*np.sin(alpha)*np.exp(-t/T1)*np.exp(-1j*ke*(2*X+u*mask)))

  # Time stepping
  prod = 1
  for i in range(n):

    # Update time
    t += dt

    if rank==0 and debug: print("- Time: {:.2f}".format(t))

    # Get displacements in the reference frame
    u = phantom.displacement(i)

    # Project to fem-image in the deformed frame
    u_image, m = _project2image_vector(u, u, image, parameters["mesh_resolution"])

    # Complex magnetization data
    for j in range(o_image.shape[-2]):
      o_image[...,j,i] =  Mxy(m, M, M0, alpha[i], prod, t, T1, ke[j], image._grid[j], u_image[...,j])

    # Save mask
    mask[...,i] = m

    # Flip angles product
    prod = prod*np.cos(alpha[i])

  return o_image, mask


# Generate complementary DENSE images
def get_complementary_dense_image(image, phantom, parameters, debug, fem):
  """ Generate DENSE images
  """
  # Time-steping parameters
  t     = parameters["t"]
  dt    = parameters["dt"]
  t_end = parameters["t_end"]
  n_t   = parameters["time_steps"]

  # Output image
  size = np.append(image._astute_resolution, [n_t])
  image_0 = np.zeros(size, dtype=complex)
  image_1 = np.zeros(size, dtype=complex)
  image_2 = np.zeros(size, dtype=complex)
  mask    = np.zeros(np.append(size[:-2], n_t), dtype=np.int16)

 # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # flip angle
  ke    = image.encoding_frequency     # encoding frequency
  M0    = 1.0                          # thermal equilibrium magnetization
  M     = M0

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
    alpha = alpha*np.ones([n_t],dtype=np.float)

  # Magnetization expressions
  Mxy0 = lambda mask, M, M0, alpha, prod, t, T1, ke, X, u, phi: mask*(+0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*u) +
                                                          0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*(2*X+u)) +
                                                          M0*(1 - np.exp(-t/T1))*np.exp(-1j*ke*(X+u)))*prod*np.sin(alpha)*np.exp(1j*phi)
  Mxy1 = lambda mask, M, M0, alpha, prod, t, T1, ke, X, u, phi: mask*(-0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*u) +
                                                          -0.5*M*np.exp(-t/T1)*np.exp(-1j*ke*(2*X+u)) +
                                                          M0*(1 - np.exp(-t/T1))*np.exp(-1j*ke*(X+u)))*prod*np.sin(alpha)*np.exp(1j*phi)
  Mxyin = lambda mask, M, M0, alpha, prod, t, T1, ke, phi: mask*(M0*(1 - np.exp(-t/T1)))*prod*np.sin(alpha)*np.exp(1j*phi)

  # Check k space bandwidth to avoid folding artifacts
  pxsz = np.array([2*np.pi/(image.kspace_factor*k) if k != 0
                   else image.voxel_size()[i] for i,k in enumerate(ke)])
  res  = np.floor(np.divide(image.FOV,pxsz)).astype(int)
  chck = [image.resolution[i] < res[i] for i in range(ke.size)]
  if np.any(chck):
    # Check if resolutions are even or odd
    for i in range(res.size):
      if (image.resolution[i] % 2 == 0) and (res[i] % 2 != 0):
        res[i] += 1
      elif (image.resolution[i] % 2 != 0) and (res[i] % 2 == 0):
        res[i] += 1

    # Create a new image object
    new_image = DENSEImage(FOV=image.FOV,
            resolution=res,
            encoding_frequency=ke,
            T1=T1,
            flip_angle=alpha,
            off_resonance=image.off_resonance,
            interpolation=image.interpolation)
    new_image._modified_resolution = True
    new_image._original_resolution = image.resolution
    new_image._original_grid = image._grid
    new_image._original_array_resolution = image.array_resolution
    new_image._original_voxel_size = image.voxel_size()
    input_image = new_image
  else:
    input_image = image

  # Off resonance
  if not image.off_resonance:
    phi = np.zeros(image.array_resolution)
  else:
    if np.any(chck):
      phi = image.off_resonance(new_image._grid[0],new_image._grid[1])
    else:
      phi = image.off_resonance(image._grid[0],image._grid[1])

  # Function space of the displacement field
  V = phantom.V

  # Determine if the image and phantom geometry are 2D or 3D
  di = input_image.type_dim()   # image geometric dimension
  dp = V.shape[0]               # phantom (fem) geometric dimension

  # Hamming filter to reduce Gibbs ringing artifacts
  Hf = [signal.hamming(image.array_resolution[i]) for i in range(di)]
  H = np.outer(Hf[0],Hf[1])

  # Spins positions
  x = V.dof_coordinates()[::dp]

  # Voxels centers
  c = input_image._grid                   # global voxel coordinates
  voxels, voxel_coords = scatter_image(c) # local indices and voxel coordinates
  nr_local_voxels = voxels.size           # Number of local voxels
  nr_voxels = c[0].size                   # Number of global voxels

  # Voxel width and image resolution
  width = input_image.voxel_size()
  resolution = input_image._astute_resolution[0:-1]

  # Connectivity (this is done just once)
  n = nr_local_voxels
  getConnectivity = globals()["getConnectivity{:d}".format(dp)]
  s2p = np.array(getConnectivity(x, voxel_coords, voxels,
                  width, nr_local_voxels, nr_voxels))  # pixel-to-spins map
  p2s = [[] for j in range(n)]
  [p2s[pixel].append(spin) for (spin, pixel) in enumerate(s2p)]

  # Spins positions with respect to its containing voxel center
  corners = np.array([c[i].flatten()[s2p]-0.5*width[i] for i in range(di)]).T
  x_rel = x[:,0:dp] - corners

  # Displacement image
  u_image = np.zeros(input_image._astute_resolution)

  # Time stepping
  prod = 1
  upre = np.zeros([x.shape[0],dp])
  for i in range(n_t):

    # Update time
    t += dt

    if rank==0 and debug: print("- Time: {:.2f}".format(t))

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(i)
    reshaped_u = u.vector().reshape((-1,dp))

    # Displacement in terms of pixels
    x_new = x_rel + reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    s2p[:] += (resolution[1]*pixel_u[:,1] + pixel_u[:,0]).astype(np.int64)

    # Update pixel-to-spins connectivity
    p2s = update_p2s(s2p, n)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Fill images
    for j in range(di):
        (I, m) = getImage(reshaped_u[:,j],p2s)
        u_image[...,j] = I.reshape(resolution)
    m = m.reshape(resolution)

    # Grid to evaluate magnetizations
    imgrid = input_image._grid

    # Iterates over slices
    S = input_image.array_resolution
    s = image.array_resolution
    for slice in range(resolution[2]):

      # Update mask
      mask[...,slice,i] = np.abs(iFFT(H*FFT(m)[int(0.5*(S[0]-s[0])):int(0.5*(S[0]-s[0])+s[0]):1,
                                        int(0.5*(S[1]-s[1])):int(0.5*(S[1]-s[1])+s[1]):1,
                                        slice]))

      # Complex magnetization data
      for j in range(image_0.shape[-2]):

        # Magnetization expressions
        tmp0 = Mxy0(m[...,slice], M, M0, alpha[i], prod, t, T1, ke[j],
                    imgrid[j][...,slice]-u_image[...,slice,j], u_image[...,slice,j],
                    phi[...,slice])
        tmp1 = Mxy1(m[...,slice], M, M0, alpha[i], prod, t, T1, ke[j],
                    imgrid[j][...,slice]-u_image[...,slice,j], u_image[...,slice,j],
                    phi[...,slice])
        tmp2 = Mxyin(m[...,slice], M, M0, alpha[i], prod, t, T1, ke[j],
                     phi[...,slice])

        # Check if images should be cropped
        if np.any(chck):

          # kspace cropping
          image_0[...,slice,j,i] = iFFT(H*FFT(tmp0)[int(0.5*(S[0]-s[0])):int(0.5*(S[0]-s[0])+s[0]):1,
                                            int(0.5*(S[1]-s[1])):int(0.5*(S[1]-s[1])+s[1]):1])
          image_1[...,slice,j,i] = iFFT(H*FFT(tmp1)[int(0.5*(S[0]-s[0])):int(0.5*(S[0]-s[0])+s[0]):1,
                                            int(0.5*(S[1]-s[1])):int(0.5*(S[1]-s[1])+s[1]):1])
          image_2[...,slice,j,i] = iFFT(H*FFT(tmp2)[int(0.5*(S[0]-s[0])):int(0.5*(S[0]-s[0])+s[0]):1,
                                            int(0.5*(S[1]-s[1])):int(0.5*(S[1]-s[1])+s[1]):1])

        else:

          image_0[...,slice,j,i] = tmp0
          image_1[...,slice,j,i] = tmp1
          image_2[...,slice,j,i] = tmp2

    # Flip angles product
    prod = prod*np.cos(alpha[i])

    # Copy previous displcement field
    upre = np.copy(reshaped_u)

  return image_0, image_1, image_2, mask


# Project vector fields onto images
def _project2image_vector(displacement, function, image, mesh_resolution, deformed=True):

  # Function space of tag-lines
  V = function.function_space()

  # Image vector dimension
  d = image.type_dim()

  # Move the ventricle according with u
  if deformed:
    V.mesh().move(displacement)

  # Number of slices
  number_of_slices = image._astute_resolution[2]

  # Output images
  o_image = np.zeros(image._astute_resolution,dtype=np.complex)
  if d is 2:
    mask_shape = image._astute_resolution[0:d+1]
  else:
    mask_shape = image._astute_resolution[0:d]

  # Projection scheme
  fem2image = image.projection_scheme()

  # Create local image data
  voxels, voxel_coords = scatter_image(image._grid)
  local_size = voxels.size
  if image._modified_resolution:
    _voxels, _voxel_coords = scatter_image(image._original_grid)
    _local_size = _voxels.size
    original_mask = image._modified_resolution
  else:
    original_mask = False

  # Constant input arguments
  dofs = V.vertex_to_dof_map()[::d]    # degrees of freedom
  x = V.dof_coordinates()[::d]         # updated ventricular coordinates
  nr_voxels = image._grid[0].size      # number of voxels (global)
  resolution = image.array_resolution  # image resolution
  if original_mask:
    _nr_voxels = image._original_grid[0].size
    _resolution = image._original_array_resolution

  # Connectivity (this should be done once)
  slice = 0
  p2s = getConnectivity(x, voxel_coords, voxels,
                  image.voxel_size(),
                  local_size, nr_voxels, slice)  # pixel-to-spins map
  s2p = -np.ones([x.shape[0],],dtype=np.int64)   # spin-to-pixel map
  n = len(p2s)
  for i in range(n):
      s2p[p2s[i]] = i

  # Fill image
  for slice in range(number_of_slices):


    TEST = 2
    if TEST == 1:
        #####################################
        # connectivity test
        #####################################


        # fig = plt.imshow(I,cmap=plt.get_cmap('gray'))
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
        # plt.show()

        #####################################

        # Get mask
        mask = 0
        weights = mask + 1e-10

        # Get original mask
        if original_mask:
          omask = getMask(x, _voxel_coords, _voxels,
                          image._original_voxel_size,
                          _local_size, _nr_voxels, slice)

          # Gather mask
          omask = gather_image(np.reshape(omask,_resolution))

        else:
          omask = []

        # Image
        for j in range(d):

          # Reduce images
          o_image[...,slice,j] = gather_image(np.reshape(I[j]/weights,resolution))

        # Gather mask
        mask = gather_image(np.reshape(mask,resolution))
    else:

        # Get mask
        if image.interpolation is 'mean':
          weights = getWeights(x, voxel_coords, voxels,
                          image.voxel_size(),
                          local_size, nr_voxels, slice)
          mask = (weights > 1e-06).astype(float)
        else:
          mask = getMask(x, voxel_coords, voxels,
                        image.voxel_size(),
                        local_size, nr_voxels, slice)
          weights = mask

        # Get original mask
        if original_mask:
          omask = getMask(x, _voxel_coords, _voxels,
                          image._original_voxel_size,
                          _local_size, _nr_voxels, slice)

          # Gather mask
          omask = gather_image(np.reshape(omask,_resolution))

        else:
          omask = []

        # Check dofs
        I = fem2image(mask, x, voxel_coords,
                     image.voxel_size(),
                     voxels, dofs, function.vector(),
                     local_size, nr_voxels, slice)
        # Image
        for j in range(d):

          # Reduce images
          o_image[...,slice,j] = gather_image(np.reshape(I[j]/weights,resolution))

        # Gather mask
        mask = gather_image(np.reshape(mask,resolution))


  # Reset mesh
  if deformed:
    displacement.vector()[:] *= -1.0
    V.mesh().move(displacement)
    displacement.vector()[:] *= -1.0

  return np.squeeze(o_image), mask, omask


# #######################################
# #   Projections
# #######################################
# # Project scalar fields onto images
# def _project2image_scalar(displacement, taglines, image, mesh_resolution):
#   # Function space of tag-lines
#   V = taglines.function_space()
#
#   # Image vector dimension
#   d = image.type_dim()
#
#   # Ventricle dofmap (dofs coordinates changes with time)
#   dofs = V.vertex_to_dof_map()
#
#   # Move the ventricle according with u
#   V.mesh().move(displacement)
#
#   # Updated ventricular dof coordinates
#   x = V.dof_coordinates()
#
#   # Number of slices
#   number_of_slices = image._astute_resolution[2]
#
#   # Output image
#   o_image = np.zeros(image._astute_resolution)
#   weights = np.zeros(o_image.shape)
#
#   # Projection scheme
#   fem2image = image._scheme
#
#   # Voxel and element volumes
#   voxel_vol = np.prod(image.voxel_size(), axis=0)
#   elem_vol  = np.pi*mesh_resolution**d
#   if d > 2: elem_vol = 4.0/3.0*elem_vol
#
#   # Estimated number of dofs in voxel
#   voxel_dofs = int(round(8.0*voxel_vol/elem_vol))
#   mask = np.zeros([voxel_dofs,], dtype=int)
#
#   # Create local data
#   local_dofs, local_coords, local_values = scatter_dofs(dofs, x, taglines.vector())
#
#   # Fill image
#   for i in range(number_of_slices):
#
#     # Check dofs
#     [spamm_image, lweights] = fem2image(mask, local_coords, image.sparse_grid, image.voxel_size(),
#                                        local_dofs, local_values, image.resolution, i)
#
#     # Gather results
#     spamm_image = gather_image(spamm_image)
#     lweights    = gather_image(lweights)
#
#     # Avoid dividing by zero
#     if rank is not 0:
#       lweights = 1
#
#     # Weighted image
#     spamm_image = np.divide(spamm_image, lweights)
#
#     # Fill image
#     o_image[...,i] = spamm_image
#     weights[...,i] = lweights
#
#   # Reset mesh
#   displacement.vector()[:] *= -1.0
#   V.mesh().move(displacement)
#   displacement.vector()[:] *= -1.0
#
#   return np.squeeze(o_image), np.squeeze(weights)
