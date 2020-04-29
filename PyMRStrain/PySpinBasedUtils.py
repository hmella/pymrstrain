import numpy as np
from PyMRStrain.Helpers import iseven, isodd, round_to_even
from PyMRStrain.MPIUtilities import MPI_print


# Three dimensional images with number of slices greater than 1
def update_s2p(s2p, pixel_u, resolution):
    s2p += (resolution[0]*pixel_u[:,1]             # jump betwen rows
        + resolution[0]*resolution[1]*pixel_u[:,2] # jump betwen slices
        + pixel_u[:,0]).astype(np.int64)           # jump between columns


# Check if the kspace bandwidth needs to be modified
# to avoid artifacts during the generation and to
# reproduce the behavior of the scanner
def check_kspace_bw(image, x):

  # Encoding frequency, voxelsize, FOV, kspace factor,
  # and oversampling_factor
  ke = image.encoding_frequency
  kf = image.kspace_factor
  vs = image.voxel_size()
  FOV = np.copy(image.FOV)
  ofac = image.oversampling_factor

  # Lower bound for the new encoding frequencies
  kl = 2*np.pi/vs

  # Virtual voxel size, estimated to satisfy the frequency requirements
  # in the kspace to avoid artifacts (i.e. folding) in the image domain
  pxsz = np.array([2*np.pi/(kf*freq) if (freq != 0 and kf*freq > kl[i])
                   else vs[i] for (i,freq) in enumerate(ke)])

  # Field of view in the measurement and phase directions including the
  # oversampling factor
  FOV[:2]  = ofac*FOV[:2]

  # Virtual resolution for the generation process
  virtual_resolution = np.divide(FOV, pxsz).astype(np.int64)
  if ofac != 1:
      virtual_resolution[:2] = round_to_even(virtual_resolution[:2])
  incr_bw = np.any([image.resolution[i] < virtual_resolution[i] for i in range(ke.size)])  

  # If the resolution needs modification then check if the new
  # one has even or odd components to make the cropping process
  # easier
  if incr_bw:
      # Check if resolutions are even or odd
      for i in range(virtual_resolution.size):
          if ke[i] != 0:
              tail = (virtual_resolution[i]-ofac*image.resolution[i])/2
              if iseven(image.resolution[i]) and iseven(tail):
                  print(1)
                  virtual_resolution[i] += 1
              if isodd(image.resolution[i]) and isodd(tail):
                  print(2)
                  virtual_resolution[i] += 1

      # Create a new image object with the new FOV and resolution
      new_image = image.__class__(FOV=FOV,
              center=image.center,
              resolution=virtual_resolution,
              encoding_frequency=ke,
              T1=image.T1,
              off_resonance=image.off_resonance)
  else:
      new_image = image

  # Output dict
  D = {"voxel_size": new_image.voxel_size(),
       "grid":       new_image.grid,
       "resolution":  new_image.resolution,
       "FOV": new_image.FOV}
  del new_image

  # Debug printings
  MPI_print(' Acq. matrix: ({:d},{:d})'.format(image.acq_matrix[0],
      image.acq_matrix[1]))
  MPI_print(' Generation matrix: ({:d},{:d})'.format(virtual_resolution[0],
      virtual_resolution[1]))

  return virtual_resolution, incr_bw, D


#
def check_nb_slices(grid, x, vsz, res):

  # Flatten grid
  Xf = [p.flatten('F') for p in grid]

  # Min and max position in the slice direction
  Z = np.array([Xf[2].min(), Xf[2].max()])
  z = np.array([x[:,2].min(), x[:,2].max()])
  borders = np.array([-0.5*vsz[2], 0.5*vsz[2]])
  N = (z - (Z+borders))/vsz[2]
  sign = [-1, 1]
  N = [int(np.ceil(np.abs(N[i]))) if sign[i]*N[i] > 0 else 0 for i in range(len(N))]

  # Number of additional slices
  r_slice = res[0]*res[1]
  for i in range(len(N)):
    for j in range(N[i]):
      for k in range(3):
        if k < 2:
          Xf[k] = np.append(Xf[k], Xf[k][0:r_slice], axis=0)
        else:
          Xf[k] = np.insert(Xf[k], i*Xf[k].size, Z[i]*np.ones([r_slice,]) + (j+1)*vsz[k]*(-1)**(i+1), axis=0)

  # Slice location factor and number of voxels
  SL = r_slice*N[0]

  # Reshape output
  resolution = [res[0], res[1], res[2]+sum(N)]
  Xf = [p.reshape(resolution, order='F') for p in Xf]

  return Xf, SL
