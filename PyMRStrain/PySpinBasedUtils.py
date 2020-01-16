import numpy as np
from PyMRStrain.Image import *

# Check if the kspace bandwidth needs to be modified
# to avoid artifacts during the generation and to
# reproduce the behavior of the scanner
def check_kspace_bw(image):

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
       "array_resolution":  new_image.array_resolution}
  del new_image

  return res, incr_bw, D

# Two dimensional meshgrids
def update_s2p2(s2p, pixel_u, resolution):
    s2p[:] += (resolution[1]*pixel_u[:,1] + pixel_u[:,0]).astype(np.int64)
    # return s2p

# Three dimensional images with number of slices greater than 1
def update_s2p3(s2p, pixel_u, resolution):
        s2p[:] += (resolution[1]*pixel_u[:,0]             # jump betwen rows
               + resolution[1]*resolution[0]*pixel_u[:,2] # jump betwen slices
               + pixel_u[:,1]).astype(np.int64)           # jump between columns
