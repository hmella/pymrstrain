import numpy as np
from PyMRStrain.Image import *


# Two dimensional meshgrids
def update_s2p2(s2p, pixel_u, resolution):
    s2p[:] += (resolution[1]*pixel_u[:,1] + pixel_u[:,0]).astype(np.int64)


# Three dimensional images with number of slices greater than 1
def update_s2p3(s2p, pixel_u, resolution):
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
  FOV = image.FOV
  oversampling = image.oversampling_factor


  # Field of view in the measurement and phase
  # direction fullfilling the acquisition matrix
  # size
  # FOV_m  = 1.0/((1.0/vs[0])/(oversampling*image.resolution[0]))
  # pxsz_m = FOV_m/(oversampling*image.resolution[0])
  # vs[0], FOV[0] = pxsz_m, FOV_m
  FOV_m  = 1.0/((1.0/vs[0:2])/(oversampling*image.resolution[0:2]))
  pxsz_m = FOV_m/(oversampling*image.resolution[0:2])
  vs[0:2], FOV[0:2] = pxsz_m, FOV_m

  # Lower bound for the new encoding frequencies
  kl = 2*np.pi/vs

  # Modified pixel size
  pxsz = np.array([2*np.pi/(kf*freq) if (freq != 0 and kf*freq > kl[i])
                   else vs[i] for (i,freq) in enumerate(ke)])

  # Modified image resolution
  res = np.floor(np.divide(image.FOV,pxsz)).astype(np.int64)
  incr_bw = np.any([image.resolution[i] < res[i] for i in range(ke.size)])

  print(res, FOV)

  # If the resolution needs modification then check if the new
  # one has even or odd components to make the cropping process
  # easier
  if incr_bw:
      # Check if resolutions are even or odd
      for i in range(res.size):
          if (image.resolution[i] % 2 == 0) and (res[i] % 2 != 0):
              res[i] += 1
          elif (image.resolution[i] % 2 != 0) and (res[i] % 2 == 0):
              res[i] += 1

      # Create a new image object
      new_image = DENSEImage(FOV=image.FOV,
              center=image.center,
              resolution=res,
              encoding_frequency=ke,
              T1=image.T1,
              off_resonance=image.off_resonance)
  else:
      new_image = image

  # Output dict
  D = {"voxel_size": image.FOV/res,
       "grid":       new_image.grid,
       "resolution":  new_image.resolution}
  del new_image

  return res, incr_bw, D


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
