from mpi4py import MPI
import numpy as np

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def scatter_image(X):

  # Get image coordinates and reshape
  # X = image._grid
  X = [X[i].flatten() for i in range(len(X))] 

  # Voxels indices
  voxels = np.linspace(0,X[0].size-1,X[0].size,dtype=np.int)


  if rank==0:

    # Number of voxels
    arr_size = X[0].size/size
    sections = [int(arr_size*i) for i in range(1,size)]

    # Split arrays
    local_voxels = [[a] for a in np.split(voxels, sections, axis=0)]

  else:
    #Create variables on other cores
    local_voxels = None

  # Scatter local arrays to other cores
  local_voxels = comm.scatter(local_voxels, root=0)[0]
  local_coords = [X[i][local_voxels] for i in range(len(X))]

  # # Make dofs local
  # local_voxels  = local_voxels - local_voxels.min()

  return local_voxels, local_coords

def scatter_dofs(dofs, coordinates, values, geodim):
  ''' Scatter dofmap and coordinate dofs to local processes

  Input:
  -----------
    dofs:        numpy ndarray of shape [n, d] with d the dimension of the function space
    coordinates: numpy ndarray dofs coordinates

  Output:
  -----------
    local_dofs: distributed dofs along all processes
    local_coords: distributed coordinates along all processes
  '''
  if rank==0:

    # Number of dofs 
    arr_size = int(dofs.size/size)
    if arr_size % geodim is not 0:
      arr_size = arr_size - 1
    sections = [int(arr_size*i) for i in range(1,size)]

    # Split arrays
    local_dofs = [[a] for a in np.split(dofs, sections, axis=0)]

  else:
    #Create variables on other cores
    local_dofs = None

  # Scatter local arrays to other cores
  local_dofs   = comm.scatter(local_dofs, root=0)[0]
  local_coords = coordinates[local_dofs]
  local_values = values[local_dofs]

  # Make dofs local
  local_dofs  = local_dofs - local_dofs.min()

  return local_dofs, local_coords, local_values
  
  
def gather_image(image):

  # Empty image
  total = np.zeros_like(image)

  # Reduced image
  comm.Reduce([image, MPI.DOUBLE], [total, MPI.DOUBLE], op=MPI.SUM, root=0)
  
  return total
