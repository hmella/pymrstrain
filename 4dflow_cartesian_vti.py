import os
from subprocess import run

import matplotlib.pyplot as plt
import meshio
import numpy as np
from pyevtk.hl import imageToVTK
from pyevtk.vtk import VtkGroup
import yaml
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Noise import add_cpx_noise
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

  # Import imaging parameters
  with open('PARAMETERS.yaml') as file:
    pars = yaml.load(file, Loader=yaml.FullLoader)

  # Imaging parameters
  FOV = np.array(pars["Imaging"]["FOV"])
  RES = np.array(pars["Imaging"]["RES"])
  dt = pars["Imaging"]["TIMESPACING"]

  # Formatting parameters
  tx = pars["Formatting"]["tx"]
  ty = pars["Formatting"]["ty"]
  tz = pars["Formatting"]["tz"]

  # Import generated data
  seq  = 'FFE'
  VENC = 250
  Hcr  = 45
  K = np.load('MRImages/HCR{:d}/{:s}_V{:d}.npy'.format(Hcr,seq,VENC))

  # Fix the direction of kspace lines measured in the opposite direction
  if seq == 'EPI':
    K[:,1::2,...] = K[::-1,1::2,...]

  # Apply the inverse Fourier transform to obtain the image
  I = ktoi(K[::2,::-1,...],[0,1,2])

  # Add noise
  I = add_cpx_noise(I, relative_std=0.025, mask=1)

  # Make sure the directory exist
  vti_path = "MRImages/HCR{:d}/vti".format(Hcr)
  if not os.path.isdir(vti_path):
    os.makedirs(vti_path, exist_ok=True)

  # Origin and pixel spacing of the generated image
  origin  = -0.5*FOV
  spacing = FOV/RES

  # Conver to tuple
  origin  = tuple(origin)
  spacing = tuple(spacing)

  # Create VtkGroup object to write PVD
  pvd = VtkGroup(vti_path+'/IM_{:s}_V{:d}'.format(seq,VENC))

  # Export vti files
  print("Exporting vti...")
  for fr in range(K.shape[-1]):
    print("    writing fame {:d}".format(fr))

    # Get velocity and magnitude
    v = (np.angle(I[:,:,:,0,fr]), np.angle(I[:,:,:,1,fr]), np.angle(I[:,:,:,2,fr]))
    m = (np.abs(I[:,:,:,0,fr]), np.abs(I[:,:,:,1,fr]), np.abs(I[:,:,:,2,fr]))

    # Estimate angiographic image
    angio = (m[0] + m[1] + m[2])/3*np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)/K.shape[-1]

    # Write VTI
    frame_path = vti_path+'/IM_{:s}_V{:d}_{:04d}'.format(seq,VENC,fr)
    imageToVTK(frame_path, cellData={'velocity': v, 'angiography': angio, 'magnitude': m}, origin=origin, spacing=spacing)

    # Add VTI files to pvd group
    pvd.addFile(filepath=frame_path+'.vti', sim_time=fr*dt)

  # Save PVD
  pvd.save()

  # Import mesh, translate it to the origin, rotate it, and scale to meters
  path_NS = "phantoms/Non-linear/HCR{:d}/xdmf/phantom.xdmf".format(Hcr)
  with meshio.xdmf.TimeSeriesReader(path_NS) as reader:
    nodes, elems = reader.read_points_cells()
    nodes[:,0] -= 0.5*(nodes[:,0].max()+nodes[:,0].min()) # x-translation
    nodes[:,1] -= 0.5*(nodes[:,1].max()+nodes[:,1].min()) # y-translation
    nodes[:,2] -= 0.5*(nodes[:,2].max()+nodes[:,2].min()) # z-translation
    nodes = (Rz(tz)@Ry(ty)@Rx(tx)@nodes.T).T  # mesh rotation
    nodes /= 100  # mesh scaling
    Nfr = reader.num_steps # number of frames

    # Path to export the generated data
    xdmf_path = "MRImages/HCR{:d}/xdmf".format(Hcr)

    # Make sure the directory exist
    if not os.path.isdir(xdmf_path):
      os.makedirs(xdmf_path, exist_ok=True)

    # Write data
    xdmffile_path = xdmf_path+"/phantom.xdmf".format(Hcr)
    print("Exporting XDMF file in the image domain...")
    with meshio.xdmf.TimeSeriesWriter(xdmffile_path) as writer:
      writer.write_points_cells(nodes, elems)

      # Iterate over cardiac phases
      for fr in range(Nfr):
        print("    writing frame {:d}".format(fr))

        # Read velocity and pressure data in each time step
        d, point_data, cell_data = reader.read_data(fr)
        velocity = point_data['velocity']
        pressure = point_data['pressure']

        # Rotate velocity
        velocity = (Rz(tz)@Ry(ty)@Rx(tx)@velocity.T).T

        # Convert everything to meters
        velocity /= 100

        # Export data in the registered frame
        writer.write_data(fr*dt, point_data={"velocity": velocity, "pressure": pressure})

    # Move generated HDF5 file to the right folder
    run(['mv','phantom.h5',xdmf_path])
