import numpy as np
from Fem import assemble, massAssemble, massAssembleInt
import meshio

if __name__ == '__main__':

  # Import mesh
  mesh = meshio.read('mesh/tube.msh')
  elems = mesh.cells_dict['tetra']
  nodes = mesh.points 
  print(elems)
  print(nodes)

  # Values to be assembled
  values = np.zeros([nodes.shape[0], 3])
  values[:] = 2.0

  # Call to assemble function
  integral = assemble(elems,nodes,values)
  print(integral)

  integral = massAssembleInt(elems,nodes,values)
  print(integral)