import matplotlib.pyplot as plt
import meshio
import numpy as np
from Fem import massAssemble
from FlowToImage import FlowImage3D

from PyMRStrain.KSpaceTraj import Cartesian
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import gather_image, MPI_rank
from PyMRStrain.Plotter import multi_slice_viewer

if __name__ == '__main__':

  # Path to phantom data
  xdmffile = "phantom/xdmf/phantom.xdmf"

  # Read test
  with meshio.xdmf.TimeSeriesReader(xdmffile) as reader:
    # Import mesh
    nodes, elems = reader.read_points_cells()
    elems = elems[0].data

    # Read velocity data in each time step
    d, point_data, cell_data = reader.read_data(7)
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
  # if MPI_rank==0:
  #   traj.plot_trajectory()

  # Get mass matrix
  M = massAssemble(elems,nodes)

  # Kspace parameters for sampling in kz
  FOV_z = 0.30
  res_z = 60
  delta_kz = 1.0/FOV_z 
  BW_kz = 1.0/(FOV_z/res_z)
  kz = np.arange(-0.5*BW_kz,0.5*BW_kz,delta_kz)

  # Generate 4D flow image
  K = np.zeros([traj.ro_samples, traj.ph_samples, len(kz)], dtype=complex)
  for i in range(len(kz)):
    print(MPI_rank, "kz location {:d}".format(i))
    # Generate kspace
    K[:,traj.local_idx,i] = FlowImage3D(M, traj.points, 1000*traj.times, velocity, nodes, 250.0, 1.5, kz[i])

  # Gather results
  K = gather_image(K)

  # Fix dimensions
  K[:,1::2,:] = K[::-1,1::2,:]
  I = ktoi(K[::2,::-1,:],[0,1,2])

  # Export generated data
  if MPI_rank==0:
    np.save('kspace_test',K)

  # Show figure
  if MPI_rank==0:
    multi_slice_viewer(np.abs(K[::2,:,:]))
    multi_slice_viewer(np.abs(I[:,:,:]))
    multi_slice_viewer(np.angle(I[:,:,:]))
    plt.show()