import time

import matplotlib.pyplot as plt
import numpy as np
from Connectivity import getConnectivity3, update_p2s
from ImageBuilding import (CSPAMM_magnetizations, DENSE_magnetizations,
                           EXACT_magnetizations, SINE_magnetizations,
                           get_images)
from PyMRStrain.Helpers import cropping_ranges, m_dirs, order
from PyMRStrain.KSpace import kspace
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import MPI_print, MPI_rank, MPI_size, gather_image
from PyMRStrain.MRImaging import acq_to_res
from PyMRStrain.PySpinBasedUtils import (check_kspace_bw, check_nb_slices,
                                         update_s2p)
from PyMRStrain.Spins import Function


# Complementary SPAMM images
def get_cspamm_image(image, epi, phantom, parameters, debug=False):
  """ Generate tagging images
  """
  # Time-steping parameters
  t     = parameters.t
  dt    = parameters.dt
  n_t   = parameters.time_steps

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
  mask    = np.zeros(np.append(image.resolution, n_t), dtype=np.float32)

  # Output kspaces
  k_nsa_1 = kspace(size, image, epi)
  k_nsa_2 = kspace(size, image, epi)
  k_in = kspace(size, image, epi)

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
  (s2p, excited_spins) = getConnectivity3(x, voxel_coords, width)
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
    update_s2p(s2p, pixel_u, resolution)

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
    mags = [m0, m1, m_in]
    for i in range(len(mags)):
        mags[i][(~inside + ~exc_slice),:] = 0

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
    #     S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #     write_vtk([u,S2P,EXC,rot], path='output/Normal_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Average nb. of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*0.5*(m.max()-m.min())))
    else:
        MPI_print('Time step {:d}'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))
    m = gather_image(m.reshape(resolution, order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Update mask
      # mask[...,slice,i] = np.abs(ktoi(itok(m[...,slice])[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]))

      # Complex magnetization data
      for enc_dir in range(dk):

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

    # Flip angles product
    prod = prod*np.cos(alpha[time_step])

  return k_nsa_1, k_nsa_2, k_in, mask


# Complementary DENSE images
def get_cdense_image(image, epi, phantom, parameters, debug=False):
  """ Generate DENSE images
  """
  # Time-steping parameters
  t     = parameters.t
  dt    = parameters.dt
  n_t   = parameters.time_steps

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
  mask    = np.zeros(np.append(image.resolution, n_t), dtype=np.float32)

  # Output kspaces
  k_nsa_1 = kspace(size, image, epi)
  k_nsa_2 = kspace(size, image, epi)
  k_in = kspace(size, image, epi)

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
  (s2p, excited_spins) = getConnectivity3(x, voxel_coords, width)
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
    update_s2p(s2p, pixel_u, resolution)

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
    mags = DENSE_magnetizations(M, M0, alpha[time_step], prod, t, T1,
                        ke[0:dk], x[:,0:dk], reshaped_u[:,0:dk])
    # m0[(~inside + ~exc_slice),:] = 0
    # m1[(~inside + ~exc_slice),:] = 0
    # mags = [m0, m1, m_in]
    for i in range(len(mags)):
        mags[i][(~inside + ~exc_slice),:] = 0

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
        MPI_print('Time step {:d}. Average nb. of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*0.5*(m.max()-m.min())))
    else:
        MPI_print('Time step {:d}'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))
    m = gather_image(m.reshape(resolution, order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Update mask
      # mask[...,slice,i] = np.abs(ktoi(H*itok(m[...,slice])[r[0]:r[1]:fac, c[0]:c[1]:1]))

      # Complex magnetization data
      for enc_dir in range(dk):

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

    # Flip angles product
    prod = prod*np.cos(alpha[time_step])

  return k_nsa_1, k_nsa_2, k_in, mask


# Exact image
def get_exact_image(image, epi, phantom, parameters, debug=False):
  """ Generate EXACT images
  """
  # Time-steping parameters
  t     = parameters.t
  dt    = parameters.dt
  n_t   = parameters.time_steps

 # Sequence parameters
  ke    = image.encoding_frequency     # encoding frequency
  M0    = 1.0                          # thermal equilibrium magnetization

  # Determine if the image and phantom geometry are 2D or 3D
  di = image.type_dim()                      # image geometric dimension
  dp = phantom.x.shape[-1]                   # phantom (fem) geometric dimension
  dk = np.sum(int(k/k) for k in ke if k!= 0) # Number of encoding directions

  # Output image
  size = np.append(image.resolution, [dk, n_t])
  mask    = np.zeros(np.append(image.resolution, n_t), dtype=np.float32)

  # Output kspaces
  k_nsa_1 = kspace(size, image, epi)

  # Spins positions
  x = phantom.x

  # Check k space bandwidth to avoid folding artifacts
  res, incr_bw, D = check_kspace_bw(image, x)

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
  (s2p, excited_spins) = getConnectivity3(x, voxel_coords, width)
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
  for time_step in range(n_t):

    # Update time
    t += dt

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(time_step)
    reshaped_u = u.vector()

    # Displacement in terms of pixels
    x_new = x_rel #+ reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    update_s2p(s2p, pixel_u, resolution)

    # Update pixel-to-spins connectivity
    # if image.slice_following:
    #   p2s = update_p2s(s2p-SL, excited_spins, nr_voxels)
    # else:
    #   p2s = update_p2s(s2p, excited_spins, nr_voxels)
    p2s = update_p2s(s2p, excited_spins, nr_voxels)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Updated spins positions
    x_upd = x

    # Get magnetization on each spin
    mag = EXACT_magnetizations(ke[0:dk], reshaped_u[:,0:dk])
    mags = [mag]
    for i in range(len(mags)):
        mags[i][(~inside + ~exc_slice),:] = 0

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
    #     S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #     write_vtk([u,S2P,EXC,rot], path='output/Normal_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Average nb. of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*0.5*(m.max()-m.min())))
    else:
        MPI_print('Time step {:d}'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m = gather_image(m.reshape(resolution, order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Update mask
      # mask[...,slice,i] = np.abs(ktoi(H*itok(m[...,slice])[r[0]:r[1]:fac, c[0]:c[1]:1]))

      # Complex magnetization data
      for enc_dir in range(dk):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,enc_dir]

        # Uncorrected kspaces
        k0 = itok(tmp0)[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[enc_dir][1]]/image.phase_profiles
        k_nsa_1.gen_to_acq(k0, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)

  return k_nsa_1, mask


# Sine image
def get_sine_image(image, epi, phantom, parameters, debug=False):
  """ Generate EXACT images
  """
  # Time-steping parameters
  t     = parameters.t
  dt    = parameters.dt
  n_t   = parameters.time_steps

 # Sequence parameters
  T1    = image.T1                     # relaxation
  ke    = image.encoding_frequency     # encoding frequency

  # Determine if the image and phantom geometry are 2D or 3D
  di = image.type_dim()                      # image geometric dimension
  dp = phantom.x.shape[-1]                   # phantom (fem) geometric dimension
  dk = np.sum(int(k/k) for k in ke if k!= 0) # Number of encoding directions

  # Output image
  size = np.append(image.resolution, [dk, n_t])
  mask = np.zeros(np.append(image.resolution, n_t), dtype=np.float32)

  # Output kspaces
  k_nsa_1 = kspace(size, image, epi)
  k_mask  = kspace(size, image, epi)

  # Spins positions
  x = phantom.x

  # Check k space bandwidth to avoid folding artifacts
  res, incr_bw, D = check_kspace_bw(image, x)

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
  (s2p, excited_spins) = getConnectivity3(x, voxel_coords, width)
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
    update_s2p(s2p, pixel_u, resolution)

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
    mags = SINE_magnetizations(t, T1, ke[0:dk], x[:,0:dk])
    for i in range(len(mags)):
        mags[i][(~inside + ~exc_slice),:] = 0

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
    #     S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #     write_vtk([u,S2P,EXC,rot], path='output/Normal_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Average nb. of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*0.5*(m.max()-m.min())))
    else:
        MPI_print('Time step {:d}'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m = gather_image(I[1].reshape(m0_image.shape, order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Complex magnetization data
      for enc_dir in range(dk):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,enc_dir]
        tmp1 = m[...,slice,enc_dir]

        # Uncorrected kspaces
        k0 = itok(tmp0)[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]
        k1 = itok(tmp1)[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[enc_dir][1]]/image.phase_profiles
        k_nsa_1.gen_to_acq(k0, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_mask.gen_to_acq(k1, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)

  return k_nsa_1, k_mask


# PC-SPAMM images
def get_PCSPAMM_image(image, phantom, parameters, debug=False):
  """ Generate tagging images
  """
  return True, True, True
