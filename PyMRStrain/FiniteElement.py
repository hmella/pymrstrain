from PyMRStrain.FEMGeometry import *

# Base class for finite elements
class FiniteElementBase(object):
  def __init__(self, kind, degree=1):
    self.kind = cells_dict[kind]
    self.degree = degree

  # Get element degree
  def get_degree(self):
    return self.degree

  # Geometric dimension
  def dimension(self):
    if self.kind in CELL_TYPES_2:
      dim = 2
    elif self.kind in CELL_TYPES_3:
      dim = 3
    return dim


# Finite element class
class FiniteElement(FiniteElementBase):
  def __init__(self, *args, **kwargs):
    super(FiniteElement, self).__init__(*args, **kwargs)

  # Shape of finite element
  def shape(self):
    return (self.dimension(),)

  # Number of dofs per cell
  def get_dofs_per_cell(self):
    return _get_dofs_per_cell(self.kind, self.shape)


# Vector element class
class VectorElement(FiniteElementBase):
  def __init__(self, *args, **kwargs):
    super(VectorElement, self).__init__(*args, **kwargs)

  def shape(self):
    return (self.dimension(),)

  # Number of dofs per cell
  def get_dofs_per_cell(self):
    return _get_dofs_per_cell(self.kind, self.shape)



# Number of dofs per cell
def _get_dofs_per_cell(cell_type, shape):
  if cell_type == cells_dict["triangle"]:
    n = TRIANGLE
  elif cell_type == cells_dict["quadrilateral"]:
    n = QUADRILATERAL
  elif cell_type == cells_dict["tetrahedron"]:
    n = TETRAHEDRON
  elif cell_type == cells_dict["tetrahedron10"]:
    n = TETRAHEDRON10
  elif cell_type == cells_dict["hexahedron"]:
    n = HEXAHEDRON
  return shape[0]*n
