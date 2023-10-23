import os
import time

import matplotlib.pyplot as plt
import meshio
import numpy as np
import yaml
from Fem import massAssemble
from FlowToImage import FlowImage3D
from PyMRStrain.KSpaceTraj import Cartesian
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import MPI_comm, MPI_rank, gather_image


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

  # Preview partial results
  preview = True

  # Import imaging parameters
  with open('PARAMETERS.yaml') as file:
    pars = yaml.load(file, Loader=yaml.FullLoader)

  # Imaging parameters
  FOV = np.array(pars["Imaging"]["FOV"])
  RES = np.array(pars["Imaging"]["RES"])
  T2star = pars["Imaging"]["T2star"]/1000.0
  VENCs = np.array(pars["Imaging"]["VENC"])
  OFAC = pars["Imaging"]["OVERSAMPLING"]

  # Formatting parameters
  tx = pars["Formatting"]["tx"]
  ty = pars["Formatting"]["ty"]
  tz = pars["Formatting"]["tz"]

  # Kspace parameters in the z-direction
  delta_kz = 1.0/FOV[2]
  BW_kz    = 1.0/(FOV[2]/RES[2])
  kz       = np.arange(-0.5*BW_kz, 0.5*BW_kz, delta_kz)

  # Hematocrits
  hematocrits = [45] #[35, 45, 50]

  # Iterate over hematocrits
  for Hcr in hematocrits:

    # Navier-Stokes simulation data to be used
    path_NS = "phantoms/Non-linear/HCR{:d}/xdmf/phantom.xdmf".format(Hcr)

    # Import mesh, translate it to the origin, rotate it, and scale to meters
    with meshio.xdmf.TimeSeriesReader(path_NS) as reader:
      nodes, elems = reader.read_points_cells()
      elems = elems[0].data
      nodes[:,0] -= 0.5*(nodes[:,0].max()+nodes[:,0].min()) # x-translation
      nodes[:,1] -= 0.5*(nodes[:,1].max()+nodes[:,1].min()) # y-translation
      nodes[:,2] -= 0.5*(nodes[:,2].max()+nodes[:,2].min()) # z-translation
      nodes = (Rz(tz)@Ry(ty)@Rx(tx)@nodes.T).T  # mesh rotation
      nodes /= 100  # mesh scaling
      meshio.write_points_cells("debug/mesh.vtk", nodes, [("tetra", elems)])
      Nfr = reader.num_steps # number of frames
      if MPI_rank == 0:
        bmin = np.min(nodes, axis=0)
        bmax = np.max(nodes, axis=0)
        print('Bounding box: ({:f},{:f},{:f}), ({:f},{:f},{:f})'.format(bmin[0],bmin[1],bmin[2],bmax[0],bmax[1],bmax[2]))

      # Assemble mass matrix for integrals (just once)
      M = massAssemble(elems,nodes)

      # Imaging sequences for the image generation
      sequences = ["FFE", "EPI"]

      # Image generation
      for seq in sequences:
        for VENC in VENCs:

          # Path to export the generated data
          export_path = "MRImages/HCR{:d}/{:s}_V{:.0f}".format(Hcr,seq,100.0*VENC)

          # Make sure the directory exist
          if not os.path.isdir("MRImages/HCR{:d}".format(Hcr)):
            if MPI_rank==0:
              os.makedirs("MRImages/HCR{:d}".format(Hcr), exist_ok=True)

          # Generate kspace trajectory
          lps = pars[seq]["LinesPerShot"]
          traj = Cartesian(FOV=FOV[:-1], res=RES[:-1], oversampling=OFAC, lines_per_shot=lps, VENC=VENC)

          # Print echo time
          if MPI_rank==0: print("Echo time = {:.1f} ms".format(1000.0*traj.echo_time))

          # kspace array
          K = np.zeros([traj.ro_samples, traj.ph_samples, len(kz), 3, Nfr], dtype=complex)

          # List to store how much is taking to generate one volume
          times = []

          # Iterate over cardiac phases
          for fr in range(Nfr):

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
            K[traj.local_idx,:,:,:,fr] = FlowImage3D(MPI_rank, M, traj.points, kz, traj.times, velocity, nodes, T2star, VENC)
            t1 = time.time()
            times.append(t1-t0)

            # Save kspace for debugging purposes
            if preview:
              K_copy = np.copy(K)
              K_copy = gather_image(K_copy)
              if MPI_rank==0:
                np.save(export_path, K_copy)

            # Synchronize MPI processes
            print(np.array(times).mean())
            MPI_comm.Barrier()

          # Show mean time that takes to generate each 3D volume
          print(np.array(times).mean())

          # Gather results
          K = gather_image(K)

          # Export generated data
          if MPI_rank==0:
            np.save(export_path, K)

          # # Fix dimensions
          # K[:,1::2,...] = K[::-1,1::2,...]
          # I = ktoi(K[::2,::-1,...],[0,1,2])