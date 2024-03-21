import os
import time

import matplotlib.pyplot as plt
import meshio
import numpy as np
import yaml
from Fem import massAssemble
from FlowToImage import FlowImage3D
from PyMRStrain.IO import scale_data
from PyMRStrain.KSpaceTraj import Cartesian
from PyMRStrain.Math import itok, ktoi, Rx, Ry, Rz
from PyMRStrain.MPIUtilities import MPI_comm, MPI_rank, gather_image


def FlowKspace(M, traj, velocity, nodes, gamma_x_delta_B0, T2star, VENC, profile):
  # Instead of rotating the kspace to obtain the desired MPS orientation, rotate tha phantom
  nodes = (traj.MPS_ori.T@nodes.T).T        # mesh rotation
  velocity = (traj.MPS_ori.T@velocity.T).T  # velocity rotation

  # Generate kspace locations
  kspace = FlowImage3D(MPI_rank, M, traj.points, traj.times, velocity, nodes, gamma_x_delta_B0, T2star, VENC, profile)

  return kspace


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

  # Hardware parameters
  G_sr  = pars["Hardware"]["G_sr"]
  G_max = pars["Hardware"]["G_max"]
  r_BW  = pars["Hardware"]["r_BW"]

  # Imaging orientation paramters
  M = pars["Formatting"]["M"]
  P = pars["Formatting"]["P"]
  S = pars["Formatting"]["S"]
  MPS_ori = np.array([M,P,S])

  # Simulation type
  simtypes = ['Linear','Non-linear']

  # Hematocrits
  hematocrits = [60] #[10, 35, 45,  60, 70]

  # Imaging sequences for the image generation
  sequences = ["FFE", "EPI"]

  # Iterate over CFD simulation types
  for simtype in simtypes:

    # Iterate over sequences
    for seq in sequences:

      # Iterate over hematocrits
      for Hcr in hematocrits:

        # Navier-Stokes simulation data to be used
        path_NS = "phantoms/{:s}/HCR{:d}/xdmf/phantom.xdmf".format(simtype, Hcr)

        # Import mesh, translate it to the origin, rotate it, and scale to meters
        with meshio.xdmf.TimeSeriesReader(path_NS) as reader:
          nodes, elems = reader.read_points_cells()
          elems = elems[0].data
          nodes[:,0] -= 0.5*(nodes[:,0].max()+nodes[:,0].min()) # x-translation
          nodes[:,1] -= 0.5*(nodes[:,1].max()+nodes[:,1].min()) # y-translation
          nodes[:,2] -= 0.5*(nodes[:,2].max()+nodes[:,2].min()) # z-translation
          nodes /= 100  # mesh scaling
          meshio.write_points_cells("debug/mesh.vtk", nodes, [("tetra", elems)])
          Nfr = reader.num_steps # number of frames
          if MPI_rank == 0:
            bmin = np.min(nodes, axis=0)
            bmax = np.max(nodes, axis=0)
            print('Bounding box: ({:f},{:f},{:f}), ({:f},{:f},{:f})'.format(bmin[0],bmin[1],bmin[2],bmax[0],bmax[1],bmax[2]))

          # Assemble mass matrix for integrals (just once)
          M = massAssemble(elems,nodes)

          # Slice profile
          gammabar = 1.0e+6*42.58 # Hz/T 
          G_z = 1.0e-3*30.0   # [T/m]
          delta_z = 0.002  # [m]
          delta_g = gammabar*G_z*delta_z
          z0 = 0.0
          profile = (np.abs(nodes[:,1] - z0) <= delta_z).astype(np.float32)

          # Fiel inhomogeneity
          delta_B0 = nodes[:,0] + nodes [:,1] + nodes[:,2]  # spatial distribution
          delta_B0 /= np.abs(delta_B0).max()  # normalization
          delta_B0 *= 1.5*1e-6  # scaling (1 ppm of 1.5T)        
          delta_B0 *= 1.0  # additional scaling (just for testing)
          gamma_x_delta_B0 = 2*np.pi*gammabar*delta_B0

          # Iterate over vencs
          for VENC in VENCs:

            # Path to export the generated data
            export_path = "MRImages/{:s}/HCR{:d}/{:s}_V{:.0f}".format(simtype,Hcr,seq,100.0*VENC)

            # Make sure the directory exist
            if not os.path.isdir("MRImages/{:s}/HCR{:d}".format(simtype,Hcr)):
              if MPI_rank==0:
                os.makedirs("MRImages/{:s}/HCR{:d}".format(simtype,Hcr), exist_ok=True)

            # Generate kspace trajectory
            lps = pars[seq]["LinesPerShot"]
            traj = Cartesian(FOV=FOV, res=RES, oversampling=OFAC, lines_per_shot=lps, VENC=VENC, MPS_ori=MPS_ori, receiver_bw=r_BW, Gr_max=G_max, Gr_sr=G_sr, plot_seq=False)

            # Print echo time
            if MPI_rank==0: print("Echo time = {:.1f} ms".format(1000.0*traj.echo_time))

            # kspace array
            ro_samples = traj.ro_samples
            ph_samples = traj.ph_samples
            slices = traj.slices
            K = np.zeros([ro_samples, ph_samples, slices, 3, Nfr], dtype=np.complex64)

            # List to store how much is taking to generate one volume
            times = []

            # Iterate over cardiac phases
            for fr in range(Nfr):

              # Read velocity data in each time step
              d, point_data, cell_data = reader.read_data(fr)
              velocity = point_data['velocity']

              # Convert everything to meters
              velocity /= 100

              # Generate 4D flow image
              if MPI_rank == 0: print("Generating frame {:d}".format(fr))
              t0 = time.time()
              K[traj.local_idx,:,:,:,fr] = FlowKspace(M, traj, velocity, nodes, gamma_x_delta_B0, T2star, VENC, profile)
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
              # K_scaled = scale_data(K, mag=False, real=True, imag=True, dtype=np.uint64)
              np.save(export_path, K)
