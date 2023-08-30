import os

import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyevtk
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

  # Formatting parameters
  tx = pars["Formatting"]["tx"]
  ty = pars["Formatting"]["ty"]
  tz = pars["Formatting"]["tz"]

  # Import generated data
  seq = 'FFE'
  Hcr = 35
  K = np.load('MRImages/HCR{:d}/{:s}.npy'.format(Hcr,seq))

  # Fix the direction of kspace lines measured in the opposite direction
  if seq == 'EPI':
    K[:,1::2,...] = K[::-1,1::2,...]

  # Apply the inverse Fourier transform to obtain the image
  I = ktoi(K[::2,::-1,...],[0,1,2])

  # Add noise 
  I = add_cpx_noise(I, relative_std=0.025, mask=1)

  # Make sure the directory exist
  if not os.path.isdir("MRImages/HCR{:d}/vti".format(Hcr)):
    os.makedirs("MRImages/HCR{:d}/vti".format(Hcr), exist_ok=True)

  # Origin and pixel spacing of the generated image
  origin  = -0.5*FOV
  spacing = FOV/RES

  # Conver to tuple
  origin  = tuple(origin)
  spacing = tuple(spacing)

  # Export vti files
  for fr in range(K.shape[-1]):
    print(fr)
    for i in range(3):
      pyevtk.hl.imageToVTK('MRImages/HCR{:d}/vti/PH_{:d}_{:04d}.vti'.format(Hcr,i, fr), cellData={'data': np.angle(I[:,:,:,i,fr])}, origin=origin, spacing=spacing)

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
    export_path = "MRImages/HCR{:d}/{:s}".format(Hcr,seq)

    # Make sure the directory exist
    if not os.path.isdir("MRImages/HCR{:d}/xdmf".format(Hcr)):
      os.makedirs("MRImages/HCR{:d}/xdmf".format(Hcr), exist_ok=True)

    # Write data
    xdmffile = "MRImages/HCR{:d}/xdmf/phantom.xdmf".format(Hcr)
    with meshio.xdmf.TimeSeriesWriter(xdmffile) as writer:
      writer.write_points_cells(nodes, elems)

      # Iterate over cardiac phases
      for fr in range(Nfr):

        print(fr)

        # Read velocity data in each time step
        d, point_data, cell_data = reader.read_data(fr)
        velocity = point_data['velocity']

        # Rotate velocity
        velocity = (Rz(tz)@Ry(ty)@Rx(tx)@velocity.T).T

        # Convert everything to meters
        velocity /= 100

        # Export data in the registered frame
        writer.write_data(fr, point_data={"velocity": velocity})