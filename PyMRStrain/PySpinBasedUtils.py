import numpy as np
from PyMRStrain.Image import *


# Two dimensional meshgrids
def update_s2p2(s2p, pixel_u, resolution):
    s2p[:] += (resolution[1]*pixel_u[:,1] + pixel_u[:,0]).astype(np.int64)
    # return s2p


# Three dimensional images with number of slices greater than 1
def update_s2p3(s2p, pixel_u, resolution, belong):
    # for i in range(len(s2p)):
    #     if s2p[i] >= 0 and s2p[i] < nr_pixels:
    #         s2p[i] += (resolution[1]*pixel_u[i,0]             # jump betwen rows
    #                + resolution[1]*resolution[0]*pixel_u[i,2] # jump betwen slices
    #                + pixel_u[i,1]).astype(np.int64)           # jump between columns
    s2p[belong] += (resolution[1]*pixel_u[belong,0]             # jump betwen rows
           + resolution[1]*resolution[0]*pixel_u[belong,2] # jump betwen slices
           + pixel_u[belong,1]).astype(np.int64)           # jump between columns


# Check if the kspace bandwidth needs to be modified
# to avoid artifacts during the generation and to
# reproduce the behavior of the scanner
def check_kspace_bw(image, x):

  # Encoding frequency
  ke = image.encoding_frequency

  # Modified pixel size
  pxsz = np.array([2*np.pi/(image.kspace_factor*k) if k != 0
                   else image.voxel_size()[i] for i,k in enumerate(ke)])

  # Modified image resolution
  res = np.floor(np.divide(image.FOV,pxsz)).astype(np.int64)
  incr_bw = np.any([image.resolution[i] < res[i] for i in range(ke.size)])

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
              resolution=res,
              encoding_frequency=ke,
              T1=image.T1,
              off_resonance=image.off_resonance,
              interpolation=image.interpolation)
      new_image._modified_resolution = True
      new_image._original_resolution = image.resolution
      new_image._original_grid = image._grid
      new_image._original_array_resolution = image.array_resolution
      new_image._original_voxel_size = image.voxel_size()
  else:
      new_image = image

  # Output dict
  D = {"voxel_size": image.FOV/res,
       "grid":       new_image._grid,
       "astute_resolution": new_image._astute_resolution,
       "array_resolution":  new_image.array_resolution,
       "inc_grid_slice_dir": []}
  del new_image

  return res, incr_bw, D

#
def check_nb_slices(grid, x, voxel_size, res):

  # Flatten grid
  Xf = [x.flatten('F') for x in grid]

  # Min and max position in the slice direction
  Z = np.array([Xf[2].min(), Xf[2].max()])
  z = np.array([x[:,2].min(), x[:,2].max()])
  N = (z - Z)/voxel_size[2]
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
          Xf[k] = np.insert(Xf[k], i*Xf[k].size, Z[i]*np.ones([r_slice,]) + (j+1)*voxel_size[k]*(-1)**(i+1), axis=0)

  # Slice location factor and number of voxels
  SL = r_slice*N[0]
  nr_voxels = Xf[0].size

  # Reshape output
  resolution = [res[0], res[1], res[2]+sum(N)]
  Xf = [x.reshape(resolution, order='F') for x in Xf]

  return Xf, SL
