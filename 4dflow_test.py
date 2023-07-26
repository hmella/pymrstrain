import time

import matplotlib.pyplot as plt
import meshio
import numpy as np
from Fem import massAssemble
from FlowToImage import FlowImage3D

from PyMRStrain.KSpaceTraj import Cartesian
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import MPI_comm, MPI_rank, gather_image
from PyMRStrain.Plotter import multi_slice_viewer


def Rx(tx):
  return np.array([[1, 0, 0],
                    [0, np.cos(tx), -np.sin(tx)],
                    [0, np.sin(tx), np.cos(tx)]])

def Ry(ty):
  return np.array([[np.cos(ty), 0, np.sin(ty)],
                    [0, 1, 0],
                    [-np.sin(ty), 0, np.cos(ty)]])

def Rz(tz):
  return np.array([[np.cos(tz), -np.sin(tz), 0],
                    [np.sin(tz), np.cos(tz), 0],
                    [0, 0, 1]])

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
    ty = -np.pi/40
    tz = np.pi/4
    nodes = (Rz(tz)@Ry(ty)@Rx(tx)@nodes.T).T

    # Convert everything to meters
    nodes /= 100
    # print(nodes[:,0].min(),nodes[:,1].min(),nodes[:,2].min())
    # print(nodes[:,0].max(),nodes[:,1].max(),nodes[:,2].max())

    # Export mesh for debugging
    meshio.write_points_cells("debug/mesh.vtk", nodes, [("tetra", elems)])

    # Kspace parameters for sampling in kz
    FOV_z    = 0.2
    res_z    = 10#100
    delta_kz = 1.0/FOV_z 
    BW_kz    = 1.0/(FOV_z/res_z)
    kz       = np.arange(-0.5*BW_kz, 0.5*BW_kz, delta_kz)

    # T2 relaxation and VENC
    T2   = 250.0/1000.0   # s
    VENC = 1.5            # 1.5 # m/s

    # Kspace trajectory
    FOV = np.array([0.1325, 0.1], dtype=np.float64)
    res = np.array([10, 10], dtype=np.int64)#np.array([66, 50], dtype=np.int64)
    traj = Cartesian(FOV=FOV, res=res, oversampling=2, lines_per_shot=5, VENC=VENC)
    if MPI_rank==0: print("Echo time = {:.1f} ms".format(1000.0*traj.echo_time))

    # Assemble mass matrix for integrals (just once)
    M = massAssemble(elems,nodes)

    # kspace
    K = np.zeros([traj.ro_samples, traj.ph_samples, len(kz), 3, reader.num_steps], dtype=complex)

    # Generate images
    times = []
    for fr in range(reader.num_steps):

      # Read velocity data in each time step
      d, point_data, cell_data = reader.read_data(fr)
      velocity = point_data['velocity']

      # Rotate velocity
      velocity = (Rz(tz)@Ry(ty)@Rx(tx)@velocity.T).T

      # Convert everything to meters
      velocity /= 100

      # Generate 4D flow image
      if MPI_rank == 0: print("Generating frame {:d}".format(fr))
      t0 = time.time()
      K[traj.local_idx,:,:,:,fr] = FlowImage3D(MPI_rank, M, traj.points, kz, traj.times, velocity, nodes, T2, VENC)
      t1 = time.time()
      times.append(t1-t0)

      # Synchronize MPI processes
      MPI_comm.Barrier()

      # Copy kspace and export it for debugging
      K_copy = np.copy(K)
      K_copy = gather_image(K_copy)
      K_copy[:,1::2,...] = K_copy[::-1,1::2,...]
      if MPI_rank==0: np.save('kspace_test',K_copy)

      # Synchronize MPI processes
      MPI_comm.Barrier()

    # Show mean time that takes to generate each 3D volume
    print(np.array(times).mean())

    # Gather results
    K = gather_image(K)

    # Fix dimensions
    K[:,1::2,...] = K[::-1,1::2,...]
    I = ktoi(K[::2,::-1,...],[0,1,2])

    # Export generated data
    if MPI_rank==0:
      np.save('kspace_test',K)

    # # Show figure
    # if MPI_rank==0:
    #   multi_slice_viewer(np.abs(K[::2,:,:,:,7]))
    #   multi_slice_viewer(np.abs(I[:,:,:,:,7]))
    #   multi_slice_viewer(np.angle(I[:,:,:,:,7]), caxis=[-np.pi, np.pi])
    #   plt.show()
