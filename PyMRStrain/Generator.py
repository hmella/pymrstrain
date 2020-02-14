import time

import matplotlib.pyplot as plt
import numpy as np
from Connectivity import getConnectivity2, getConnectivity3, update_p2s
from ImageBuilding import (CSPAMM_magnetizations, DENSE_magnetizations,
                           get_images)
from PyMRStrain.Helpers import m_dirs, order, cropping_ranges
from PyMRStrain.KSpace import kspace
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import MPI_print, MPI_rank, MPI_size, gather_image
from PyMRStrain.MRImaging import acq_to_res
from PyMRStrain.PySpinBasedUtils import (check_kspace_bw, check_nb_slices,
                                         update_s2p2, update_s2p3)
from PyMRStrain.Spins import Function


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


# Complementary SPAMM images
def get_cspamm_image(image, epi, phantom, parameters, debug):
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

  # Output kspaces
  k_nsa_1 = kspace(size, image.acq_matrix, image.oversampling_factor, epi)
  k_nsa_2 = kspace(size, image.acq_matrix, image.oversampling_factor, epi)
  k_in = kspace(size, image.acq_matrix, image.oversampling_factor, epi)

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
  # if image.slice_following:
  #     Xf, SL = check_nb_slices(Xf, x, width, res)

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
  # if image.slice_following:
  #     voxel_coords  = [X.flatten('F') for X in D['grid']] # reset voxel coords

  # Magnetization images and spins magnetizations
  m0_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)
  m1_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)
  m2_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)

  # Grid to evaluate magnetizations
  X = D['grid']

  # Resolutions and cropping ranges
  ovrs_fac = image.oversampling_factor
  r, c, dr, dc = cropping_ranges(image.resolution, resolution, ovrs_fac)

  # Spins inside the ventricle
  inside = phantom.spins.regions[:,-1]
  exc_slice  = (x[:,2] < image.center[2] + 0.5*image.slice_thickness)
  exc_slice *= (x[:,2] > image.center[2] - 0.5*image.slice_thickness)

  # Time stepping
  prod = 1
  upre = np.zeros([x.shape[0], dp])
  for time_step in range(n_t):

    # Update time
    t += dt

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(time_step)
    reshaped_u = u.vector()

    # Displacement in terms of pixels
    x_new = x_rel + reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    globals()["update_s2p{:d}".format(dp)](s2p, pixel_u, resolution)

    # Update pixel-to-spins connectivity
    # if image.slice_following:
    #   p2s = update_p2s(s2p-SL, excited_spins, nr_voxels)
    # else:
      # p2s = update_p2s(s2p, excited_spins, nr_voxels)
    p2s = update_p2s(s2p, excited_spins, nr_voxels)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Updated spins positions
    x_upd = x + reshaped_u

    # Copy previous displcement field
    upre = np.copy(reshaped_u)

    # Get magnetization on each spin
    (m0, m1, m_in) = CSPAMM_magnetizations(M, M0, alpha[time_step], beta, prod, t, T1,
                        ke[0:dk], x[:,0:dk])
    m0[(~inside + ~exc_slice),:] = 0
    m1[(~inside + ~exc_slice),:] = 0
    mags = [m0, m1, m_in]

    # # Debug
    # if MPI_rank==0:
    #     from PyMRStrain.IO import write_vtk
    #     from PyMRStrain.Math import wrap
    #     S2P = Function(u.spins, dim=1)  # spins-to-pixel connectivity
    #     EXC = Function(u.spins, dim=1)  # excited slice (spins)
    #     rot = Function(u.spins, dim=1)
    #     theta = np.arctan(x[:,1]/x[:,0])
    #     rot.vector()[:] = wrap(theta.reshape((-1,1)), np.pi/8)
    #     EXC.vector()[exc_slice] = 1
    #     if image.slice_following:
    #       S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #       write_vtk([u,S2P,EXC,rot], path='output/SF_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])
    #     else:
    #       S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #       write_vtk([u,S2P,EXC,rot], path='output/Normal_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Number of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*m.max()))
    else:
        MPI_print('Time step {:d}.'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))
    m = gather_image(m.reshape(resolution, order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Update mask
      # mask[...,slice,i] = np.abs(ktoi(H*itok(m[...,slice])[r[0]:r[1]:fac, c[0]:c[1]:1]))

      # Complex magnetization data
      for enc_dir in range(image_0.shape[-2]):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,enc_dir]
        tmp1 = m1_image[...,slice,enc_dir]
        tmp2 = m2_image[...,slice,enc_dir]

        # Uncorrected kspaces
        k0 = itok(tmp0)[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]
        k1 = itok(tmp1)[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]
        k2 = itok(tmp2)[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[enc_dir][1]]/image.phase_profiles
        k_nsa_1.gen_to_acq(k0, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_nsa_2.gen_to_acq(k1, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_in.gen_to_acq(k2, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)

        # kspace cropping
        image_0[...,slice,enc_dir,time_step] = ktoi(k_nsa_1.k[...,slice,enc_dir,time_step])
        image_1[...,slice,enc_dir,time_step] = ktoi(k_nsa_2.k[...,slice,enc_dir,time_step])
        image_2[...,slice,enc_dir,time_step] = ktoi(k_in.k[...,slice,enc_dir,time_step])

    # Flip angles product
    prod = prod*np.cos(alpha[time_step])

  return k_nsa_1, k_nsa_2, k_in, mask



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


# Complementary DENSE images
def get_cdense_image(image, epi, phantom, parameters, debug):
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

  # Output kspaces
  k_nsa_1 = kspace(size, image.acq_matrix, image.oversampling_factor, epi)
  k_nsa_2 = kspace(size, image.acq_matrix, image.oversampling_factor, epi)
  k_in = kspace(size, image.acq_matrix, image.oversampling_factor, epi)

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
  # if image.slice_following:
  #     Xf, SL = check_nb_slices(Xf, x, width, res)

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
  # if image.slice_following:
  #     voxel_coords  = [X.flatten('F') for X in D['grid']] # reset voxel coords

  # Magnetization images and spins magnetizations
  m0_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)
  m1_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)
  m2_image = np.zeros(np.append(resolution, dk), dtype=np.complex64)

  # Grid to evaluate magnetizations
  X = D['grid']

  # Resolutions and cropping ranges
  ovrs_fac = image.oversampling_factor
  r, c, dr, dc = cropping_ranges(image.resolution, resolution, ovrs_fac)

  # Spins inside the ventricle
  inside = phantom.spins.regions[:,-1]
  exc_slice  = (x[:,2] < image.center[2] + 0.5*image.slice_thickness)
  exc_slice *= (x[:,2] > image.center[2] - 0.5*image.slice_thickness)

  # Time stepping
  prod = 1
  upre = np.zeros([x.shape[0], dp])
  for time_step in range(n_t):

    # Update time
    t += dt

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(time_step)
    reshaped_u = u.vector()

    # Displacement in terms of pixels
    x_new = x_rel + reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    globals()["update_s2p{:d}".format(dp)](s2p, pixel_u, resolution)

    # Update pixel-to-spins connectivity
    # if image.slice_following:
    #   p2s = update_p2s(s2p-SL, excited_spins, nr_voxels)
    # else:
    #   p2s = update_p2s(s2p, excited_spins, nr_voxels)
    p2s = update_p2s(s2p, excited_spins, nr_voxels)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Updated spins positions
    x_upd = x + reshaped_u

    # Copy previous displcement field
    upre = np.copy(reshaped_u)

    # Get magnetization on each spin
    (m0, m1, m_in) = DENSE_magnetizations(M, M0, alpha[time_step], prod, t, T1,
                        ke[0:dk], x[:,0:dk], reshaped_u[:,0:dk])
    m0[(~inside + ~exc_slice),:] = 0
    m1[(~inside + ~exc_slice),:] = 0
    mags = [m0, m1, m_in]

    # # Debug
    # if MPI_rank==0:
    #     from PyMRStrain.IO import write_vtk
    #     from PyMRStrain.Math import wrap
    #     S2P = Function(u.spins, dim=1)  # spins-to-pixel connectivity
    #     EXC = Function(u.spins, dim=1)  # excited slice (spins)
    #     rot = Function(u.spins, dim=1)
    #     theta = np.arctan(x[:,1]/x[:,0])
    #     rot.vector()[:] = wrap(theta.reshape((-1,1)), np.pi/8)
    #     EXC.vector()[exc_slice] = 1
    #     if image.slice_following:
    #       S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #       write_vtk([u,S2P,EXC,rot], path='output/SF_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])
    #     else:
    #       S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #       write_vtk([u,S2P,EXC,rot], path='output/Normal_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Number of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*m.max()))
    else:
        MPI_print('Time step {:d}.'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))
    m = gather_image(m.reshape(resolution, order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Update mask
      # mask[...,slice,i] = np.abs(ktoi(H*itok(m[...,slice])[r[0]:r[1]:fac, c[0]:c[1]:1]))

      # Complex magnetization data
      for enc_dir in range(image_0.shape[-2]):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,enc_dir]
        tmp1 = m1_image[...,slice,enc_dir]
        tmp2 = m2_image[...,slice,enc_dir]

        # Uncorrected kspaces
        k0 = itok(tmp0)[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]
        k1 = itok(tmp1)[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]
        k2 = itok(tmp2)[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[enc_dir][1]]/image.phase_profiles
        k_nsa_1.gen_to_acq(k0, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_nsa_2.gen_to_acq(k1, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_in.gen_to_acq(k2, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)

        # kspace cropping
        image_0[...,slice,enc_dir,time_step] = ktoi(k_nsa_1.k[...,slice,enc_dir,time_step])
        image_1[...,slice,enc_dir,time_step] = ktoi(k_nsa_2.k[...,slice,enc_dir,time_step])
        image_2[...,slice,enc_dir,time_step] = ktoi(k_in.k[...,slice,enc_dir,time_step])

    # Flip angles product
    prod = prod*np.cos(alpha[time_step])

  return k_nsa_1, k_nsa_2, k_in, mask
