import os

import numpy as np
import yaml
import matplotlib.pyplot as plt
from PyMRStrain.Filters import Tukey_filter
from PyMRStrain.Math import itok, ktoi
from scipy.io import savemat

if __name__ == '__main__':

  # Import imaging parameters
  with open('PARAMETERS.yaml') as file:
    pars = yaml.load(file, Loader=yaml.FullLoader)

  # Imaging parameters
  FOV = np.array(pars["Imaging"]["FOV"])
  RES = np.array(pars["Imaging"]["RES"])
  T2star = pars["Imaging"]["T2star"]/1000.0
  VENC = pars["Imaging"]["VENC"]
  OFAC = pars["Imaging"]["OVERSAMPLING"]

  # Sequences and hematocrits to convert
  sequences = ['FFE', 'EPI']
  hematocrits = [35, 45, 50]

  for seq in sequences:
    for Hcr in hematocrits:

      # Import kspace
      K = np.load('MRImages/HCR{:d}/{:s}.npy'.format(Hcr,seq))

      # Fix the direction of kspace lines measured in the opposite direction
      if seq == 'EPI':
        K[:,1::2,...] = K[::-1,1::2,...]

      # Kspace filtering (as the scanner would do)
      h_meas = Tukey_filter(K.shape[0], width=0.9, lift=0.3)
      h_pha  = Tukey_filter(K.shape[1], width=0.9, lift=0.3)
      h = np.outer(h_meas, h_pha)
      H = np.tile(h[:,:,np.newaxis, np.newaxis, np.newaxis], (1, 1, K.shape[2], K.shape[3], K.shape[4]))
      K_fil = H*K

      fig, ax = plt.subplots(1, 2)
      ax[0].imshow(np.abs(K[:,:,50,1,1]))
      ax[1].imshow(np.abs(K_fil[:,:,50,1,1]))
      plt.show()


      # Apply the inverse Fourier transform to obtain the image
      I = ktoi(K_fil[::2,::-1,...],[0,1,2])

      # Add noise 
      # I = add_cpx_noise(I, relative_std=0.025, mask=1)

      # Make sure the directory exist
      if not os.path.isdir("MRImages/HCR{:d}/mat".format(Hcr)):
        os.makedirs("MRImages/HCR{:d}/mat".format(Hcr), exist_ok=True)

      # Create data to export in mat format
      data = {'MR_FFE_FH': np.abs(I[:,:,:,2,:]),
              'MR_FFE_AP': np.abs(I[:,:,:,0,:]),
              'MR_FFE_RL': np.abs(I[:,:,:,1,:]),
              'MR_PCA_FH': 100.0*VENC/np.pi*np.angle(I[:,:,:,2,:]),
              'MR_PCA_AP': 100.0*VENC/np.pi*np.angle(I[:,:,:,0,:]),
              'MR_PCA_RL': 100.0*VENC/np.pi*np.angle(I[:,:,:,1,:]),
              'voxel_MR': 1000.0*(FOV/RES).reshape((1, 3)),
              'VENC': 100.0*VENC,
              'heart_rate': 42,
              'type': 'DCM'}

      # Export mat
      savemat("MRImages/HCR{:d}/mat/{:s}.mat".format(Hcr,seq), {'data': data})