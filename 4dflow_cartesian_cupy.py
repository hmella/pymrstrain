import os
import time

import matplotlib.pyplot as plt
import meshio
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
import yaml
from Fem import massAssemble
from PyMRStrain.IO import scale_data
from PyMRStrain.KSpaceTraj import Cartesian
from PyMRStrain.Math import itok, ktoi


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

def FlowImage3D(M, kxy, kzz, t ,v, r0, T2, VENC):
  # Number of kspace lines/spokes/interleaves
  nb_lines = kxy[0].shape[1] # kxy[0].cols()
  
  # Number of measurements in the readout direction
  nb_meas = kxy[0].shape[0] # kxy[0].rows()

  # Number of measurements in the kz direction
  nb_kz = len(kzz)

  # Number of spins
  nb_spins = r0.shape[0]

  # Get the equivalent gradient needed to go from the center of the kspace
  # to each location
  kx = 2.0 * np.pi * kxy[0]
  ky = 2.0 * np.pi * kxy[1]
  kz = 2.0 * np.pi * kzz

  # Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
  r = cp.asarray( np.zeros([nb_spins, 3], dtype=np.float32) , dtype=cp.float32)

  # Kspace and Fourier exponential
  Mxy = 1.0e+3 * nb_spins * np.exp(1j * np.pi / VENC * v)
  fe_xy = cp.asarray( np.zeros([nb_spins, 1], dtype=np.float32) , dtype=cp.float32)
  fe = cp.asarray( np.zeros([nb_spins, 1], dtype=np.complex64) , dtype=cp.complex64)

  # kspace
  kspace = cp.asarray( np.zeros([nb_meas, nb_lines, nb_kz, 3], dtype=np.complex64) , dtype=cp.complex64)

  # T2* decay
  T2_decay = cp.exp(-t / T2)

  # Iterate over kspace measurements/kspace points
  for j in range(nb_lines):

    # Debugging
    print("  ky location ", j)

    # Iterate over slice kspace measurements
    for i in range(nb_meas):

      # Update blood position at time t(i,j)
      r[:,:] = r0 + v*t[i,j]

      # Fourier exponential
      fe_xy[:,0] = -(r[:,0] * kx[i,j] + r[:,1] * ky[i,j])

      for k in range(nb_kz):

        # Update Fourier exponential
        fe[:,0] = cp.exp(1j * (fe_xy[:,0] - r[:,2] * kz[k]))

        # Calculate k-space values, add T2* decay, and assign value to output array
        for l in range(3):
          kspace[i,j,k,l] = M.dot(Mxy[:,l]).dot(fe[:,0]) * T2_decay[i,j]

  return kspace


if __name__ == '__main__':

  stream = cp.cuda.stream.Stream(non_blocking=True)
  cp.show_config()

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

  # Formatting parameters
  tx = pars["Formatting"]["tx"]
  ty = pars["Formatting"]["ty"]
  tz = pars["Formatting"]["tz"]

  # Kspace parameters in the z-direction
  BW_kz    = 1.0/(FOV[2]/RES[2])
  delta_kz = BW_kz/(RES[2]-1)
  kz = cp.asarray(np.linspace(-0.5*BW_kz, 0.5*BW_kz, RES[2], dtype=np.float32))

  # Fix kspace shifts
  if RES[2] % 2 == 0:
    kz = kz - 0.5*delta_kz

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
          nodes = (Rz(tz)@Ry(ty)@Rx(tx)@nodes.T).T  # mesh rotation
          nodes /= 100  # mesh scaling
          # meshio.write_points_cells("debug/mesh.vtk", nodes, [("tetra", elems)])
          Nfr = reader.num_steps # number of frames
          bmin = np.min(nodes, axis=0)
          bmax = np.max(nodes, axis=0)
          print('Bounding box: ({:f},{:f},{:f}), ({:f},{:f},{:f})'.format(bmin[0],bmin[1],bmin[2],bmax[0],bmax[1],bmax[2]))

          # Assemble mass matrix for integrals (just once)
          M = cp_csr_matrix( massAssemble(elems,nodes) )
          nodes = cp.asarray( nodes, dtype=cp.float32 )

          # Iterate over vencs
          for VENC in VENCs:

            # Path to export the generated data
            export_path = "MRImages/{:s}/HCR{:d}/{:s}_V{:.0f}".format(simtype,Hcr,seq,100.0*VENC)

            # Make sure the directory exist
            if not os.path.isdir("MRImages/{:s}/HCR{:d}".format(simtype,Hcr)):
              os.makedirs("MRImages/{:s}/HCR{:d}".format(simtype,Hcr), exist_ok=True)

            # Generate kspace trajectory
            lps = pars[seq]["LinesPerShot"]
            traj = Cartesian(FOV=FOV[:-1], res=RES[:-1], oversampling=OFAC, lines_per_shot=lps, VENC=VENC, receiver_bw=r_BW, Gr_max=G_max, Gr_sr=G_sr, plot_seq=False)
            traj.points = (cp.asarray(traj.points[0], dtype=cp.float32), cp.asarray(traj.points[1], dtype=cp.float32))
            traj.times  = cp.asarray(traj.times, dtype=cp.float32)

            # Print echo time
            print("Echo time = {:.1f} ms".format(1000.0*traj.echo_time))

            # kspace array
            K = cp.asarray(np.zeros([traj.ro_samples, traj.ph_samples, len(kz), 3, Nfr], dtype=np.complex64), dtype=cp.complex64)
            print(type(K))

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
              print("Generating frame {:d}".format(fr))
              t0 = time.time()
              K[traj.local_idx,:,:,:,fr] = FlowImage3D(M, traj.points, kz, traj.times, cp.asarray( velocity, dtype=cp.float32 ), nodes, T2star, VENC)
              t1 = time.time()
              times.append(t1-t0)

              # Save kspace for debugging purposes
              if preview:
                K_copy = np.copy(K)
                # np.save(export_path, K_copy)

              # Synchronize MPI processes
              print(np.array(times).mean())

            # Show mean time that takes to generate each 3D volume
            print(np.array(times).mean())

            # # Export generated data
            # if MPI_rank==0:
            #   # K_scaled = scale_data(K, mag=False, real=True, imag=True, dtype=np.uint64)
            #   np.save(export_path, K)

  stream.synchronize()

