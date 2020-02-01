from PyMRStrain.Function import Function
from PyMRStrain.Image import Image, DENSEImage, CSPAMMImage
from PyMRStrain.Helpers import order, m_dirs
from PyMRStrain.Magnetizations import DENSEMagnetizations
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import (gather_image, MPI_rank,
    MPI_print, MPI_size)
from PyMRStrain.MRImaging import acq_to_res
from PyMRStrain.PySpinBasedUtils import (update_s2p2, update_s2p3,
    check_kspace_bw, check_nb_slices)
from Connectivity import (getConnectivity2, getConnectivity3,
    update_p2s)
from ImageBuilding import (get_images, CSPAMM_magnetizations,
    DENSE_magnetizations)
import matplotlib.pyplot as plt
import numpy as np
import time

###################
# Common Generator
###################
class Generator:
  def __init__(self, parameters, Image, epi, phantom=None, debug=False, fem=True):
    self.p = parameters
    self.Image = Image
    self.phantom = phantom
    self.debug = debug
    self.fem = fem
    self.epi = epi

  # Obtain image
  def get_image(self):
    if self.Image.technique is "CSPAMM":
      o_image = get_cspamm_image(self.Image, self.epi, self.phantom, self.p, self.debug, self.fem)
    elif self.Image.technique is "DENSE":
      if isinstance(self.Image,Image):
        o_image = get_dense_image(self.Image, self.phantom, self.p, self.debug, self.fem)
      else:
        o_image = get_complementary_dense_image(self.Image, self.epi, self.phantom, self.p, self.debug)
    elif self.Image.technique is "PCSPAMM":
      o_image = get_PCSPAMM_image(self.Image, self.phantom, self.p, self.debug)
    elif self.Image.technique is "EXACT":
      o_image = get_exact_image(self.Image, self.phantom, self.p, self.debug)
    elif self.Image.technique is "SINE":
      o_image = get_sine_image(self.Image, self.phantom, self.p, self.debug)
    return o_image


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
  o_image = np.zeros(np.append(image.resolution, [image.type_dim(), n+1]), dtype=complex)
  mask    = np.zeros(np.append(image.resolution, n+1), dtype=np.int16)

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
    [X,Y] = image.grid
    taglines = np.zeros(np.append(X.shape,2))
    spamm = lambda omega, X: np.cos(omega*X)

  # SPAMM modulation
  d = image.geometric_dimension()
  if fem:
    for k in range(len(ke)):
      spamm_fem.vector()[k::d] = np.cos(ke[k]*x[0::d,edir[k]])

  # Time stepping
  for i in range(n+1):

    if debug: MPI_print("- Time: {:.2f}".format(t))

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


# complementary SPAMM images
def get_cspamm_image(image, epi, phantom, parameters, debug, fem):
  """ Generate tagging images
  """
  # Time-steping parameters
  t     = parameters["t"]
  dt    = parameters["dt"]
  t_end = parameters["t_end"]
  n_t   = parameters["time_steps"]

 # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # flip angle
  beta  = image.encoding_angle         # tip angle
  ke    = image.encoding_frequency     # encoding frequency
  M0    = 1.0                          # thermal equilibrium magnetization
  M     = M0

  # Determine if the image and phantom geometry are 2D or 3D
  di = image.type_dim()                      # image geometric dimension
  dp = phantom.x.shape[-1]                   # phantom (fem) geometric dimension
  dk = np.sum(int(k/k) for k in ke if k!= 0) # Number of encoding directions

  # Output image
  size = np.append(image.resolution, [dk, n_t])
  image_0 = np.zeros(size, dtype=np.complex64)
  image_1 = np.zeros(size, dtype=np.complex64)
  image_2 = np.zeros(size, dtype=np.complex64)
  mask    = np.zeros(np.append(image.resolution, n_t), dtype=np.float32)

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
      alpha = alpha*np.ones([n_t],dtype=np.float)

  # Spins positions
  x = phantom.x

  # Check k space bandwidth to avoid folding artifacts
  res, incr_bw, D = check_kspace_bw(image, x)

  # Off resonance
  if not image.off_resonance:
    phi = np.zeros(image.resolution)
  else:
    phi = image.off_resonance(D["grid"][0],D["grid"][1])

  # Grid, voxel width, image resolution and number of voxels
  Xf = D['grid']
  width = D['voxel_size']
  resolution = D['resolution']
  nr_voxels = Xf[0].size

  # Check if the number of slices needs to be increased
  # for the generation of the connectivity when generating
  # Slice-Following (SF) images
  if image.slice_following:
      Xf, SL = check_nb_slices(Xf, x, width, res)

  # Connectivity (this is done just once)
  voxel_coords = [X.flatten('F') for X in Xf]
  get_connectivity = globals()["getConnectivity{:d}".format(dp)]
  (s2p, excited_spins) = get_connectivity(x, voxel_coords, width)
  s2p = np.array(s2p)

  # Spins positions with respect to its containing voxel center
  # Obs: the option -order='F'- is included to make the grid of the
  # Z coordinate at the end of the flattened array
  corners = np.array([Xf[j].flatten('F')[s2p]-0.5*width[j] for j in range(di)]).T
  x_rel = x[:, 0:dp] - corners

  # List of spins inside the excited slice
  if image.slice_following:
      voxel_coords  = [X.flatten('F') for X in D['grid']] # reset voxel coords

  # Magnetization images and spins magnetizations
  m0_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)
  m1_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)
  m2_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)

  # Grid to evaluate magnetizations
  X = D['grid']

  # Resolutions and cropping ranges
  fac = image.oversampling_factor
  S = resolution
  s = image.resolution
  r = [int(0.5*(S[0]-fac*s[0])), int(0.5*(S[0]-fac*s[0])+fac*s[0])]
  c = [int(0.5*(S[1]-fac*s[1])), int(0.5*(S[1]-fac*s[1])+fac*s[1])]
  dr = [1, fac]
  dc = [fac, 1]

  # Spins inside the ventricle
  inside = phantom.spins.regions[:,-1]
  if image.slice_following:
      sf = (x[:,2] < image.center[2] + 0.5*image.slice_thickness)
      sf *= (x[:,2] > image.center[2] - 0.5*image.slice_thickness)
      # SP = slice_profile(x[:,2], image.center[2], image.slice_thickness)

  # Time stepping
  prod = 1
  upre = np.zeros([x.shape[0], dp])
  for i in range(n_t):

    # Update time
    t += dt

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(i)
    reshaped_u = u.vector()

    # Displacement in terms of pixels
    x_new = x_rel + reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    globals()["update_s2p{:d}".format(dp)](s2p, pixel_u, resolution)

    # Update pixel-to-spins connectivity
    if image.slice_following:
      p2s = update_p2s(s2p-SL, excited_spins, nr_voxels)
    else:
      p2s = update_p2s(s2p, excited_spins, nr_voxels)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Updated spins positions
    x_upd = x + reshaped_u

    # Copy previous displcement field
    upre = np.copy(reshaped_u)

    # Get magnetization on each spin
    (m0, m1, min) = CSPAMM_magnetizations(M, M0, alpha[i], beta, prod, t, T1,
                        ke[0:dk], x[:,0:dk])
    m0[~inside,:] = 0
    m1[~inside,:] = 0
    if image.slice_following:
        m0[~sf,:] = 0
        m1[~sf,:] = 0
        # for k in range(dk):
        #     m0[:,k] *= SP
        #     m1[:,k] *= SP
    mags = [m0, m1, m0]

    # # Debug
    # if MPI_rank==0:
    #     from PyMRStrain.IO import write_vtk
    #     from PyMRStrain.Math import wrap
    #     s2p_fun = Function(u.spins, dim=1)
    #     s2p_fun.vector()[:] = 0
    #     slicef = Function(u.spins, dim=2)
    #     slicef.vector()[:] = 0
    #     rot = Function(u.spins, dim=1)
    #     theta = np.arctan(x[:,1]/x[:,0])
    #     rot.vector()[:] = wrap(theta.reshape((-1,1)), np.pi/8)
    #     if image.slice_following:
    #       slicef.vector()[sf] = 1
    #       s2p_fun.vector()[excited_spins] = (s2p-SL).reshape((-1,1))
    #       write_vtk([u,s2p_fun,slicef,rot], path='output/uu_SF_{:04d}.vtu'.format(i), name=['u','s2p','ex_slice','rot'])
    #     else:
    #       slicef.vector()[:] = 1
    #       s2p_fun.vector()[excited_spins] = s2p.reshape((-1,1))
    #       write_vtk([u,s2p_fun,slicef,rot], path='output/uu_{:04d}.vtu'.format(i), name=['u','s2p','ex_slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Number of spins inside a voxel: {:.0f}'.format(i, MPI_size*m.max()))
    else:
        MPI_print('Time step {:d}.'.format(i))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))
    m = gather_image(m.reshape(resolution, order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Update mask
      # mask[...,slice,i] = np.abs(ktoi(H*itok(m[...,slice])[r[0]:r[1]:fac, c[0]:c[1]:1]))

      # Complex magnetization data
      for j in range(image_0.shape[-2]):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,j]
        tmp1 = m1_image[...,slice,j]
        tmp2 = m2_image[...,slice,j]

        # Uncorrected kspaces
        k0 = itok(tmp0)[r[0]:r[1]:dr[j], c[0]:c[1]:dc[j]]
        k1 = itok(tmp1)[r[0]:r[1]:dr[j], c[0]:c[1]:dc[j]]
        k2 = itok(tmp2)[r[0]:r[1]:dr[j], c[0]:c[1]:dc[j]]

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[j][1]]/image.phase_profiles
        k0 = acq_to_res(k0, image.acq_matrix[m_dirs[j]], k0.shape,
                delta_ph, epi=epi, dir=m_dirs[j], oversampling=fac)
        k1 = acq_to_res(k1, image.acq_matrix[m_dirs[j]], k1.shape,
                delta_ph, epi=epi, dir=m_dirs[j], oversampling=fac)
        k2 = acq_to_res(k2, image.acq_matrix[m_dirs[j]], k2.shape,
                delta_ph, epi=epi, dir=m_dirs[j], oversampling=fac)

        # kspace cropping
        image_0[...,slice,j,i] = ktoi(k0)
        image_1[...,slice,j,i] = ktoi(k1)
        image_2[...,slice,j,i] = ktoi(k2)

    # Flip angles product
    prod = prod*np.cos(alpha[i])

  return image_0, image_1, image_2, mask



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
  size = np.append(image.resolution, [image.type_dim(), n+1])
  image_0 = np.zeros(size, dtype=np.float)
  mask    = np.zeros(np.append(image.resolution, n+1), dtype=np.int16)

  # Time stepping
  for i in range(n+1):

    if debug: MPI_print("- Time: {:.2f}".format(t))

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
  size = np.append(image.resolution, [image.type_dim(), n])
  image_0 = np.zeros(size, dtype=complex)
  image_1 = np.zeros(size, dtype=complex)
  mask    = np.zeros(np.append(image.resolution, n), dtype=np.int16)

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

    if debug: MPI_print("- Time: {:.2f}".format(t))

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


# Generate complementary DENSE images
def get_complementary_dense_image(image, epi, phantom, parameters, debug):
  """ Generate DENSE images
  """
  # Time-steping parameters
  t     = parameters["t"]
  dt    = parameters["dt"]
  t_end = parameters["t_end"]
  n_t   = parameters["time_steps"]

 # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # flip angle
  ke    = image.encoding_frequency     # encoding frequency
  M0    = 1.0                          # thermal equilibrium magnetization
  M     = M0

  # Determine if the image and phantom geometry are 2D or 3D
  di = image.type_dim()                      # image geometric dimension
  dp = phantom.x.shape[-1]                   # phantom (fem) geometric dimension
  dk = np.sum(int(k/k) for k in ke if k!= 0) # Number of encoding directions

  # Output image
  size = np.append(image.resolution, [dk, n_t])
  image_0 = np.zeros(size, dtype=np.complex64)
  image_1 = np.zeros(size, dtype=np.complex64)
  image_2 = np.zeros(size, dtype=np.complex64)
  mask    = np.zeros(np.append(image.resolution, n_t), dtype=np.float32)

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
      alpha = alpha*np.ones([n_t],dtype=np.float)

  # Spins positions
  x = phantom.x

  # Check k space bandwidth to avoid folding artifacts
  res, incr_bw, D = check_kspace_bw(image, x)

  # Off resonance
  if not image.off_resonance:
    phi = np.zeros(image.resolution)
  else:
    phi = image.off_resonance(D["grid"][0],D["grid"][1])

  # Grid, voxel width, image resolution and number of voxels
  Xf = D['grid']
  width = D['voxel_size']
  resolution = D['resolution']
  nr_voxels = Xf[0].size

  # Check if the number of slices needs to be increased
  # for the generation of the connectivity when generating
  # Slice-Following (SF) images
  if image.slice_following:
      Xf, SL = check_nb_slices(Xf, x, width, res)

  # Connectivity (this is done just once)
  voxel_coords = [X.flatten('F') for X in Xf]
  get_connectivity = globals()["getConnectivity{:d}".format(dp)]
  (s2p, excited_spins) = get_connectivity(x, voxel_coords, width)
  s2p = np.array(s2p)

  # Spins positions with respect to its containing voxel center
  # Obs: the option -order='F'- is included to make the grid of the
  # Z coordinate at the end of the flattened array
  corners = np.array([Xf[j].flatten('F')[s2p]-0.5*width[j] for j in range(di)]).T
  x_rel = x[:, 0:dp] - corners

  # List of spins inside the excited slice
  if image.slice_following:
      voxel_coords  = [X.flatten('F') for X in D['grid']] # reset voxel coords

  # Magnetization images and spins magnetizations
  m0_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)
  m1_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)
  m2_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)

  # Grid to evaluate magnetizations
  X = D['grid']

  # Resolutions and cropping ranges
  fac = image.oversampling_factor
  S = resolution
  s = image.resolution
  r = [int(0.5*(S[0]-fac*s[0])), int(0.5*(S[0]-fac*s[0])+fac*s[0])]
  c = [int(0.5*(S[1]-fac*s[1])), int(0.5*(S[1]-fac*s[1])+fac*s[1])]
  dr = [1, fac]
  dc = [fac, 1]

  # Spins inside the ventricle
  inside = phantom.spins.regions[:,-1]
  if image.slice_following:
      sf = (x[:,2] < image.center[2] + 0.5*image.slice_thickness)
      sf *= (x[:,2] > image.center[2] - 0.5*image.slice_thickness)
      # SP = slice_profile(x[:,2], image.center[2], image.slice_thickness)

  # Time stepping
  prod = 1
  upre = np.zeros([x.shape[0], dp])
  for i in range(n_t):

    # Update time
    t += dt

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(i)
    reshaped_u = u.vector()

    # Displacement in terms of pixels
    x_new = x_rel + reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    globals()["update_s2p{:d}".format(dp)](s2p, pixel_u, resolution)

    # Update pixel-to-spins connectivity
    if image.slice_following:
      p2s = update_p2s(s2p-SL, excited_spins, nr_voxels)
    else:
      p2s = update_p2s(s2p, excited_spins, nr_voxels)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Updated spins positions
    x_upd = x + reshaped_u

    # Copy previous displcement field
    upre = np.copy(reshaped_u)

    # Get magnetization on each spin
    (m0, m1, min) = DENSE_magnetizations(M, M0, alpha[i], prod, t, T1,
                        ke[0:dk], x[:,0:dk], reshaped_u[:,0:dk])
    m0[~inside,:] = 0
    m1[~inside,:] = 0
    if image.slice_following:
        m0[~sf,:] = 0
        m1[~sf,:] = 0
        # for k in range(dk):
        #     m0[:,k] *= SP
        #     m1[:,k] *= SP
    mags = [m0, m1, m0]

    # # Debug
    # if MPI_rank==0:
    #     from PyMRStrain.IO import write_vtk
    #     from PyMRStrain.Math import wrap
    #     s2p_fun = Function(u.spins, dim=1)
    #     s2p_fun.vector()[:] = 0
    #     slicef = Function(u.spins, dim=2)
    #     slicef.vector()[:] = 0
    #     rot = Function(u.spins, dim=1)
    #     theta = np.arctan(x[:,1]/x[:,0])
    #     rot.vector()[:] = wrap(theta.reshape((-1,1)), np.pi/8)
    #     if image.slice_following:
    #       slicef.vector()[sf] = 1
    #       s2p_fun.vector()[excited_spins] = (s2p-SL).reshape((-1,1))
    #       write_vtk([u,s2p_fun,slicef,rot], path='output/uu_SF_{:04d}.vtu'.format(i), name=['u','s2p','ex_slice','rot'])
    #     else:
    #       slicef.vector()[:] = 1
    #       s2p_fun.vector()[excited_spins] = s2p.reshape((-1,1))
    #       write_vtk([u,s2p_fun,slicef,rot], path='output/uu_{:04d}.vtu'.format(i), name=['u','s2p','ex_slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Number of spins inside a voxel: {:.0f}'.format(i, MPI_size*m.max()))
    else:
        MPI_print('Time step {:d}.'.format(i))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))
    m = gather_image(m.reshape(resolution, order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Update mask
      # mask[...,slice,i] = np.abs(ktoi(H*itok(m[...,slice])[r[0]:r[1]:fac, c[0]:c[1]:1]))

      # Complex magnetization data
      for j in range(image_0.shape[-2]):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,j]
        tmp1 = m1_image[...,slice,j]
        tmp2 = m2_image[...,slice,j]

        # # Check if images should be cropped
        # if incr_bw:

        # Uncorrected kspaces
        k0 = itok(tmp0)[r[0]:r[1]:dr[j], c[0]:c[1]:dc[j]]
        k1 = itok(tmp1)[r[0]:r[1]:dr[j], c[0]:c[1]:dc[j]]
        k2 = itok(tmp2)[r[0]:r[1]:dr[j], c[0]:c[1]:dc[j]]

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[j][1]]/image.phase_profiles
        k0 = acq_to_res(k0, image.acq_matrix[m_dirs[j]], k0.shape,
                delta_ph, epi=epi, dir=m_dirs[j], oversampling=fac)
        k1 = acq_to_res(k1, image.acq_matrix[m_dirs[j]], k1.shape,
                delta_ph, epi=epi, dir=m_dirs[j], oversampling=fac)
        k2 = acq_to_res(k2, image.acq_matrix[m_dirs[j]], k2.shape,
                delta_ph, epi=epi, dir=m_dirs[j], oversampling=fac)

        # kspace cropping
        image_0[...,slice,j,i] = ktoi(k0)
        image_1[...,slice,j,i] = ktoi(k1)
        image_2[...,slice,j,i] = ktoi(k2)

        # else:
        #
        #   image_0[...,slice,j,i] = tmp0
        #   image_1[...,slice,j,i] = tmp1
        #   image_2[...,slice,j,i] = tmp2

    # Flip angles product
    prod = prod*np.cos(alpha[i])

  return image_0, image_1, image_2, mask
