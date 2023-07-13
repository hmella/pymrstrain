import matplotlib.pyplot as plt
import meshio
import numpy as np
from Fem import massAssemble
from FlowToImage import FlowImage

from PyMRStrain.KSpaceTraj import Cartesian
from PyMRStrain.Math import itok, ktoi

if __name__ == '__main__':

  # Path to phantom data
  xdmffile = "phantom/xdmf/phantom.xdmf"

  # Read test
  with meshio.xdmf.TimeSeriesReader(xdmffile) as reader:
    # Import mesh
    nodes, elems = reader.read_points_cells()
    elems = elems[0].data

    # Read velocity data in each time step
    d, point_data, cell_data = reader.read_data(22)
    velocity = point_data['velocity']

  # Use (0,0,0) as the center of the mesh
  nodes[:,0] -= nodes[:,0].mean()
  nodes[:,1] -= nodes[:,1].mean()
  nodes[:,2] -= nodes[:,2].mean()

  # Rotate mesh to align the aorta with the Z-axis
  tx = -np.pi/2
  Rx = np.array([[1, 0, 0],
                 [0, np.cos(tx), -np.sin(tx)],
                 [0, np.sin(tx), np.cos(tx)]])
  nodes = (np.matmul(Rx, nodes.T)).T
  velocity = (np.matmul(Rx, velocity.T)).T

  # Convert everything to meters
  nodes /= 100
  velocity /= 100

  # Kspace trajectory
  FOV = np.array([0.15, 0.15], dtype=np.float64)
  res = np.array([64, 64], dtype=np.int64)
  traj = Cartesian(FOV=FOV, res=res, oversampling=2, lines_per_shot=9)

  # Get mass matrix
  M = massAssemble(elems,nodes)

  # Characteristic function
  lmbda = np.ones([nodes.shape[0], 1])
  lmbda[nodes[:,2] > 0.005] = 0
  lmbda[nodes[:,2] < -0.005] = 0

  # Generate 4D flow image
  K = FlowImage(M, traj.points, 1000.0*traj.times, velocity, nodes, 50.0, 0.01, lmbda)
  K[:,1::2] = K[::-1,1::2]
  I = ktoi(K[::2,::-1])

  fig, axs = plt.subplots(1, 3)
  axs[0].imshow(np.abs(K[::2,:]),origin="lower")
  axs[1].imshow(np.abs(I),origin="lower")
  axs[2].imshow(np.angle(I),origin="lower")
  plt.show()