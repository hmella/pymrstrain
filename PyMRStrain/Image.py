import numpy as np

###################
# Base Image object
###################
class ImageBase(object):
  def __init__(self,
               FOV=np.array([0.1,0.1,0.08]),
               resolution=np.array([30, 30, 1]),
               center=np.array([0.0, 0.0, 0.0]),
               encoding_direction=[0, 1],
               flip_angle=0.5*np.pi,
               T1=1.0, T2=0.5,
               M0=1.0, M=1.0,
               off_resonance=[],
               kspace_factor=6.5,
               slice_following = False,
               slice_thickness = [],
               oversampling_factor=2,
               phase_profiles=64):
    self.FOV = FOV
    self.resolution = resolution
    self.center = center
    self.encoding_direction = encoding_direction
    self.flip_angle = flip_angle
    self.T1 = T1
    self.T2 = T2
    self.M0 = M0
    self.M  = M
    self.grid = self.generate_grid()
    self.off_resonance = off_resonance
    self.kspace_factor = kspace_factor
    self.slice_following = slice_following
    if slice_thickness != []:
        self.slice_thickness = slice_thickness
    else:
        self.slice_thickness = self.FOV[-1]
    self.oversampling_factor = oversampling_factor
    self.phase_profiles = phase_profiles
    self.acq_matrix = np.array([oversampling_factor*resolution[0], phase_profiles])

  # Flip angles
  def flip_angle_t(self, alpha):
    if isinstance(alpha,float) or isinstance(alpha,int):
      alpha = alpha*np.ones([n_t],dtype=np.float)
    return alpha

  # Geometric dimension
  def geometric_dimension(self):
    return self.resolution.size

  # Pixel size
  def voxel_size(self):
    return self.FOV/self.resolution

  # Field of view
  def field_of_view(self):
    return self.FOV

  # Image grids
  def generate_grid(self, sparse=False):
    # Resolution
    resolution = self.resolution

    # Image dimension
    d = resolution.size

    # Pixel size
    pxsz = self.voxel_size()

    # np.meshgrid generation
    X = [pxsz[i]*np.linspace(0,resolution[i]-1,resolution[i]) + self.center[i] for i in range(d)]
    X = [X[i] - X[i].mean() if i<2 else X[i] for i in range(d)]

    if d == 2:
      grid = np.meshgrid(X[0], X[1], indexing='ij', sparse=sparse)
    elif d == 3:
      grid = np.meshgrid(X[0], X[1], X[2], indexing='ij', sparse=sparse)

    return grid

  # Type of image dimension
  def type_dim(self):
    return self.geometric_dimension()

  # Voxel centers
  def voxel_midpoints(self):
    d  = len(self.resolution)
    Xc = np.zeros([self._grid[0].size, d])
    for i in range(d):
      Xc[:,i] = self._grid[i].flatten('C')

    return Xc


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
    self.technique = 'CSPAMM'
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
