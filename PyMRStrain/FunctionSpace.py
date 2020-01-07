import numpy as np

# Function space class
class FunctionSpace(object):
  def __init__(self, domain, element):
    self.domain  = domain
    self.element = element
    self.degree  = element.get_degree()
    self.shape   = element.shape()

    # Get element
  def get_element(self):
    return self.element

  # Mesh retribution
  def mesh(self):
    return self.domain
 
  # Return dof coordinates array
  def dof_coordinates(self):
    # Shapes
    [m, n] = self.domain.vertex_coordinates().shape
    l = self.shape[0]
    
    # dofs coordinates
    x = np.zeros((l*m, n))
    for i in range(l):
      x[i::l,:] = self.domain.vertex_coordinates()

    return x

  # Number of dofs
  def num_dofs(self):
    return self.dof_coordinates().shape[0]

  # Cells dofmap
  def cells_dofmap(self):
    # Shapes
    [m, n] = self.domain.cells_connectivity().shape
    l = self.shape[0]

    # vertex to dofmap
    v2d = self.vertex_to_dof_map().reshape((-1, self.shape[0])) 

    # Vertices in cells
    c2v = self.domain.cells_connectivity()
    
    # dofs coordinates
    dofmap = np.zeros((m, l*n), dtype=int)
    for i in range(m):
      dofmap[i,:] = [d for d in v2d[c2v[i,:]].flatten()]

    return dofmap

  # Vertex to dofmap
  def vertex_to_dof_map(self):
    # Dofs in vertices
    ndofs = self.num_dofs()
    dofs  = np.linspace(0, ndofs-1, ndofs, dtype=int)      
    return dofs

  # Function space shape
  def element_shape(self):
    return self.shape
