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
from PyMRStrain.Math import Rx, Ry, Rz, itok, ktoi
from PyMRStrain.MPIUtilities import MPI_comm, MPI_rank, gather_image
from PyMRStrain.Phantom import femPhantom


def FlowKspace(M, traj, phantom, gamma_x_delta_B0, T2star, VENC):
  # Get positions and velocity from phantom
  nodes = phantom.mesh['nodes']
  velocity = phantom.velocity

  # Translate phantom to obtain the desired slice location
  nodes += -traj.LOC

  # Instead of rotating the kspace to obtain the desired MPS orientation, rotate tha phantom
  nodes = (traj.MPS_ori.T@nodes.T).T        # mesh rotation
  velocity = (traj.MPS_ori.T@velocity.T).T  # velocity rotation

  meshio.write_points_cells("debug/mesh_rot.vtk", 100*nodes, [("tetra", phantom.mesh['elems'])])

  # Slice profile
  gammabar = 1.0e+6*42.58 # Hz/T 
  G_z = 1.0e-3*30.0       # [T/m]
  delta_z = FOV[2]        # [m]
  delta_g = gammabar*G_z*delta_z
  profile = (np.abs(nodes[:,2]) <= delta_z).astype(np.float32)

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
  # Imaging orientation paramters
  try:
    M_ori = pars["Formatting"]["M_ORI"]
    P_ori = pars["Formatting"]["P_ORI"]
    S_ori = pars["Formatting"]["S_ORI"]
    MPS_ori = np.array([M_ori, P_ori, S_ori])
  except:
    theta_x = np.deg2rad(pars["Formatting"]["theta_x"])
    theta_y = np.deg2rad(pars["Formatting"]["theta_y"])
    theta_z = np.deg2rad(pars["Formatting"]["theta_z"])
    R = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
    MPS_ori = R
    print(R)
  LOC = np.array(pars["Formatting"]["LOC"])

  # Simulation type
  simtypes = ['Non-linear','Linear']

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

        # Create FEM phantom object
        phantom = femPhantom(path=path_NS, vel_label='velocity', scale_factor=0.01)

        # Assemble mass matrix for integrals (just once)
        M = massAssemble(phantom.mesh['elems'], phantom.mesh['nodes'])

        # Fiel inhomogeneity
        x =  phantom.mesh['nodes']
        gammabar = 1.0e+6*42.58 # Hz/T 
        delta_B0 = x[:,0] + x[:,1] + x[:,2]  # spatial distribution
        delta_B0 /= np.abs(delta_B0).max()  # normalization
        delta_B0 *= 1.5*1e-6  # scaling (1 ppm of 1.5T)        
        delta_B0 *= 0.0  # additional scaling (just for testing)
        gamma_x_delta_B0 = 2*np.pi*gammabar*delta_B0

        # Iterate over vencs
        for VENC in VENCs:

          # Path to export the generated data
          # export_path = "MRImages/{:s}/HCR{:d}/{:s}_V{:.0f}".format(simtype,Hcr,seq,100.0*VENC)
          export_path = "test/test"          

          # Make sure the directory exist
          if not os.path.isdir("MRImages/{:s}/HCR{:d}".format(simtype,Hcr)):
            if MPI_rank==0:
              os.makedirs("MRImages/{:s}/HCR{:d}".format(simtype,Hcr), exist_ok=True)

          # Generate kspace trajectory
          lps = pars[seq]["LinesPerShot"]
          traj = Cartesian(FOV=FOV, res=RES, oversampling=OFAC, lines_per_shot=lps, VENC=VENC, MPS_ori=MPS_ori, LOC=LOC, receiver_bw=r_BW, Gr_max=G_max, Gr_sr=G_sr, plot_seq=False)

          # Print echo time
          if MPI_rank==0: print("Echo time = {:.1f} ms".format(1000.0*traj.echo_time))

          # kspace array
          ro_samples = traj.ro_samples
          ph_samples = traj.ph_samples
          slices = traj.slices
          K = np.zeros([ro_samples, ph_samples, slices, 3, phantom.Nfr], dtype=np.complex64)

          # List to store how much is taking to generate one volume
          times = []

          # Iterate over cardiac phases
          for fr in range(phantom.Nfr):

            # Read velocity data in frame fr
            phantom.read_data(fr)

            # Generate 4D flow image
            if MPI_rank == 0: print("Generating frame {:d}".format(fr))
            t0 = time.time()
            K[traj.local_idx,:,:,:,fr] = FlowKspace(M, traj, phantom, gamma_x_delta_B0, T2star, VENC)
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
