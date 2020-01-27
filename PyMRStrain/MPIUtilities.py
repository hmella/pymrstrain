from mpi4py import MPI
import numpy as np

# MPI
MPI_comm = MPI.COMM_WORLD
MPI_size = MPI_comm.Get_size()
MPI_rank = MPI_comm.Get_rank()


def scatter_image(X):

  # Get image coordinates and reshape
  # X = image._grid
  X = [X[i].flatten() for i in range(len(X))]

  # Voxels indices
  voxels = np.linspace(0,X[0].size-1,X[0].size,dtype=np.int)


  if MPI_rank==0:

    # Number of voxels
    arr_size = X[0].size/MPI_size
    sections = [int(arr_size*i) for i in range(1,MPI_size)]

    # Split arrays
    local_voxels = [[a] for a in np.split(voxels, sections, axis=0)]

  else:
    #Create variables on other cores
    local_voxels = None

  # Scatter local arrays to other cores
  local_voxels = MPI_comm.scatter(local_voxels, root=0)[0]
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
  if MPI_rank==0:

    # Number of dofs
    arr_size = int(dofs.size/MPI_size)
    if arr_size % geodim is not 0:
      arr_size = arr_size - 1
    sections = [int(arr_size*i) for i in range(1,MPI_size)]

    # Split arrays
    local_dofs = [[a] for a in np.split(dofs, sections, axis=0)]

  else:
    #Create variables on other cores
    local_dofs = None

  # Scatter local arrays to other cores
  local_dofs   = MPI_comm.scatter(local_dofs, root=0)[0]
  local_coords = coordinates[local_dofs]
  local_values = values[local_dofs]

  # Make dofs local
  local_dofs  = local_dofs - local_dofs.min()

  return local_dofs, local_coords, local_values


def gather_image(image):

  # Check input array dtype
  if image.dtype == np.complex128:
      MPI_TYPE = MPI.C_DOUBLE_COMPLEX
  elif image.dtype == np.complex64:
      MPI_TYPE = MPI.C_FLOAT_COMPLEX
  elif image.dtype == np.float:
      MPI_TYPE = MPI.DOUBLE
  elif image.dtype == np.float32:
      MPI_TYPE = MPI.FLOAT
  elif image.dtype == np.int:
        MPI_TYPE = MPI.LONG
  elif image.dtype == np.int32:
      MPI_TYPE = MPI.INT

  # Empty image
  total = np.zeros_like(image)

  # Reduced image
  MPI_comm.Reduce([image, MPI_TYPE], [total, MPI_TYPE], op=MPI.SUM, root=0)

  return total


# Scatter spins across processess
def ScatterSpins(coordinates):
  ''' Scatter dofmap and coordinate spins to local processes

  Input:
  -----------
    spins:        numpy ndarray of shape [n, d] with d the dimension of the function space
    coordinates: numpy ndarray spins coordinates

  Output:
  -----------
    local_spins: distributed spins along all processes
    local_coords: distributed coordinates along all processes
  '''
  if MPI_rank==0:

    # Number of spins
    nr_spins = coordinates.shape[0]
    geo_dim  = coordinates.shape[1]

    # Spins indices
    spins = np.linspace(0, nr_spins-1, nr_spins, dtype=np.int64)

    # Number of spins
    arr_size = int(nr_spins/MPI_size)
    if arr_size % geo_dim is not 0:
      arr_size = arr_size - 1
    sections = [int(arr_size*i) for i in range(1,MPI_size)]

    # Split arrays
    local_spins = [[a] for a in np.split(spins, sections, axis=0)]

  else:
    #Create variables on other cores
    local_spins = None

  # Scatter local arrays to other cores
  local_spins   = MPI_comm.scatter(local_spins, root=0)[0]
  local_coords = coordinates[local_spins]

  # Make spins local
  local_spins  = local_spins# - local_spins.min()

  return local_coords, local_spins
