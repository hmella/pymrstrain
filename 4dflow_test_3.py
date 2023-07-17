import matplotlib.pyplot as plt
import meshio
import numpy as np
from Fem import massAssemble
from FlowToImage import FlowImage3Dv3

from PyMRStrain.KSpaceTraj import Cartesian
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import gather_image, MPI_rank, MPI_comm
from PyMRStrain.Plotter import multi_slice_viewer

if __name__ == '__main__':

  # Path to phantom data
  xdmffile = "phantom/xdmf/phantom.xdmf"

  # Read test
  with meshio.xdmf.TimeSeriesReader(xdmffile) as reader:
    # Import mesh
    nodes, elems = reader.read_points_cells()
    elems = elems[0].data

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

    # Convert everything to meters
    nodes /= 100

    # Kspace trajectory
    FOV = np.array([0.15, 0.15], dtype=np.float64)
    res = np.array([120, 120], dtype=np.int64)
    traj = Cartesian(FOV=FOV, res=res, oversampling=2, lines_per_shot=9)
    # if MPI_rank==0:
    #   traj.plot_trajectory()

    # Kspace parameters for sampling in kz
    FOV_z    = 0.3
    res_z    = 120
    delta_kz = 1.0/FOV_z 
    BW_kz    = 1.0/(FOV_z/res_z)
    kz       = np.arange(-0.5*BW_kz, 0.5*BW_kz, delta_kz)

    # T2 relaxation and VENC
    T2   = 250   # ms
    VENC = 1.5   #1.5 # m/s

    # Assemble mass matrix for integrals (just once)
    M = massAssemble(elems,nodes)

    # kspace
    K = np.zeros([traj.ro_samples, traj.ph_samples, len(kz), reader.num_steps], dtype=complex)

    # Generate images
    for fr in range(reader.num_steps):

      # Read velocity data in each time step
      d, point_data, cell_data = reader.read_data(fr)
      velocity = point_data['velocity']

      # Rotate velocity
      velocity = (np.matmul(Rx, velocity.T)).T

      # Convert everything to meters
      velocity /= 100

      # Generate 4D flow image
      print(MPI_rank, "Generating frame {:d}".format(fr))
      K[traj.local_idx,:,:,fr] = FlowImage3Dv3(M, traj.points, kz, 1000*traj.times, velocity, nodes, T2, VENC)

      # Synchronize MPI processes
      MPI_comm.Barrier()

    # Gather results
    K = gather_image(K)

    # Fix dimensions
    K[:,1::2,:,:] = K[::-1,1::2,:,:]
    I = ktoi(K[::2,::-1,:,:],[0,1,2])

    # Export generated data
    if MPI_rank==0:
      np.save('kspace_test',K)

    # Show figure
    if MPI_rank==0:
      multi_slice_viewer(np.abs(K[::2,:,:,7]))
      multi_slice_viewer(np.abs(I[:,:,:,7]))
      multi_slice_viewer(np.angle(I[:,:,:,7]), caxis=[-np.pi, np.pi])
      plt.show()
