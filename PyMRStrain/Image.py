from ImageUtilities import *
import numpy as np

###################
# Base Image object
###################
class ImageBase(object):
  def __init__(self,
               FOV=np.array([1.0,1.0,0.04]),
               resolution=np.array([30, 30, 1]),
               center=np.array([0.0, 0.0, 0.0]),
               encoding_direction=[0, 1],
               flip_angle=0.5*np.pi,
               T1=1.0, T2=0.5,
               M0=1.0, M=1.0,
               off_resonance=[],
               interpolation='mean',
               kspace_factor=6.5):
    self.FOV = FOV
    self.resolution = resolution
    self.array_resolution = self._array_resolution()
    self.center = center
    self.encoding_direction = encoding_direction
    self.flip_angle = flip_angle
    self.T1 = T1
    self.T2 = T2
    self.M0 = M0
    self.M  = M
    self._grid = self.grid()
    self._astute_resolution = self.astute_resolution()
    self.off_resonance = off_resonance
    self.interpolation = interpolation
    self._modified_resolution = False
    self.kspace_factor = kspace_factor

  # Geometric dimension
  def geometric_dimension(self):
    return self.resolution.size

  # Pixel size
  def voxel_size(self):
    return self.FOV/(self.resolution-1)

  # Practical resolution
  # OBS: resolution of the image is intended to be the number of x-coordinates
  #      times the number of y-coordinates ([#X,#Y]), i.e., is not the same as
  #      the array resolution. Therefore, the array resolution must be the
  #      resolution of the image flipped. This was considere by introducing the
  #      variable 'array_resolution'
  def _array_resolution(self):
    r = self.resolution
    if self.resolution.size < 3:
      arr_resolution = r[::-1]
    else:
      arr_resolution = np.array([r[1],r[0],r[2]],dtype=int)
    return arr_resolution

  def astute_resolution(self):
    if self.resolution.size < 3:
      number_of_slices  = 1
      r = np.append(self.array_resolution, [number_of_slices, self.type_dim()])
    else:
      r = np.append(self.array_resolution, self.type_dim())
    return r.astype(int)

  # Field of view
  def field_of_view(self):
    return self.FOV

  # Image grids
  def grid(self, sparse=False):
    # Resolution
    resolution = self.resolution

    # Image dimension
    d = resolution.size

    # np.meshgrid generation
    X = [np.linspace(-0.5*self.FOV[i], 0.5*self.FOV[i], resolution[i]) + self.center[i] for i in range(d)]

    if d == 3 and resolution[2] == 1:
      X[2] = np.array(self.center[2])
    elif d < 3:
      X.append(np.array(self.center[2]))
    grid = np.meshgrid(X[0], X[1], X[2], indexing='xy', sparse=sparse)

    return grid

  # Get sparse grid
  def _get_sparse_grid(self):
    # Squeeze coordinates
    sparse_grid = self.grid(sparse=True)
    return [np.squeeze(x) for x in sparse_grid]

  # Type of image dimension
  def type_dim(self):
    return self.geometric_dimension()

  # Voxel centers
  def voxel_midpoints(self):
    d  = len(self.resolution)
    Xc = np.zeros([self._grid[0].size, d])
    for i in range(d):
      Xc[:,i] = self._grid[i].flatten('F')

    return Xc

  # Projection scheme
  def projection_scheme(self):
    if self.resolution.size < 3:
      if self.interpolation is 'mean':
        scheme = fem2image_vector_mean
      elif self.interpolation is 'gaussian':
        scheme = fem2image_vector_gaussian
    elif self.resolution.size == 3:
      scheme = fem2image_vector_3d
    return scheme

###################
# DENSE Image
###################
class DENSEImage(ImageBase):
  def __init__(self, encoding_frequency=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    self.technique = 'DENSE'
    self._modified_resolution = False
    self._original_resolution = []
    self._original_grid = []
    self._original_array_resolution = []
    self._original_voxel_size = []
    super(DENSEImage, self).__init__(**kwargs)

###################
# EXACT Image
###################
class EXACTImage(ImageBase):
  def __init__(self, encoding_frequency=None,**kwargs):
    self.technique = 'EXACT'
    super(EXACTImage, self).__init__(**kwargs)

###################
# CSPAMM Image
###################
class CSPAMMImage(ImageBase):
  def __init__(self, encoding_frequency=None,
               encoding_angle=15*np.pi/180,
               taglines=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    self.encoding_angle = encoding_angle
    self.taglines = taglines
    self.technique = 'Tagging'
    super(CSPAMMImage, self).__init__(**kwargs)

###################
# SINE Image
###################
class SINEImage(ImageBase):
  def __init__(self, encoding_frequency=None,
               taglines=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    self.taglines = taglines
    self.technique = 'SINE'
    super(SINEImage, self).__init__(**kwargs)

###################
# PCSPAMM Image
###################
class PCSPAMMImage(ImageBase):
  def __init__(self, encoding_frequency=None,
               v_encoding_frequency=None,
               encoding_angle=15*np.pi/180,
               taglines=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    self.vel_encoding_frequency = v_encoding_frequency
    self.encoding_angle = encoding_angle
    self.taglines = taglines
    self.technique = 'PCSPAMM'
    super(PCSPAMMImage, self).__init__(**kwargs)

###################
# Common Image
###################
class Image(ImageBase):
  def __init__(self, type=None,
               encoding_frequency=None,
               encoding_angle=15*np.pi/180,
               complementary=False,
               venc=None,
               taglines=None, **kwargs):
    self.technique  = type
    self.encoding_frequency = encoding_frequency
    self.complementary = complementary
    self.encoding_angle = encoding_angle
    self.taglines   = taglines
    super(Image, self).__init__(**kwargs)
