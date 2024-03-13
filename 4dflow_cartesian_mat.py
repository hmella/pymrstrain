import os

import numpy as np
import yaml
import matplotlib.pyplot as plt
from PyMRStrain.Filters import Tukey_filter
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Noise import add_cpx_noise
from PyMRStrain.Plotter import multi_slice_viewer
from scipy.io import savemat

if __name__ == '__main__':

  # Import imaging parameters
  with open('PARAMETERS.yaml') as file:
    pars = yaml.load(file, Loader=yaml.FullLoader)

  # Imaging parameters
  FOV = np.array(pars["Imaging"]["FOV"])
  RES = np.array(pars["Imaging"]["RES"])
  T2star = pars["Imaging"]["T2star"]/1000.0
  VENCs = np.array(pars["Imaging"]["VENC"])
  OFAC = pars["Imaging"]["OVERSAMPLING"]

  # Simulations, sequences and hematocrits to convert
  simulations = ['Linear','Non-linear']
  sequences = ['EPI', 'FFE']
  hematocrits = [10, 35, 45, 60, 70]

  for sim in simulations:
    for seq in sequences:
      for Hcr in hematocrits:
        for VENC in VENCs:

          # Get VENC in cm/s
          VENC_cms = 100.0*VENC

          # Debug
          print("Exporting {:s} data, Hct{:d}, VENC = {:.0f} cm/s".format(seq, Hcr, VENC_cms))

          # Import kspace
          K = np.load('MRImages/{:s}/HCR{:d}/{:s}_V{:.0f}.npy'.format(sim,Hcr,seq,VENC_cms))

          # Fix the direction of kspace lines measured in the opposite direction
          if seq == 'EPI':
            for ph in range(K.shape[1]):
              # Evaluate readout direction
              if ph % 5 == 0:
                ro = 1

              # Fix kspace ordering
              K[:,ph,...] = K[::ro,ph,...]

              # Reverse readout
              ro = -ro

          # Kspace filtering (as the scanner would do)
          h_meas = Tukey_filter(K.shape[0], width=0.9, lift=0.3)
          h_pha  = Tukey_filter(K.shape[1], width=0.9, lift=0.3)
          h = np.outer(h_meas, h_pha)
          H = np.tile(h[:,:,np.newaxis, np.newaxis, np.newaxis], (1, 1, K.shape[2], K.shape[3], K.shape[4]))
          K_fil = H*K

          # Apply the inverse Fourier transform to obtain the image
          I = ktoi(K_fil[::2,::-1],[0,1,2])

          # fig, ax = plt.subplots(1, 2)
          # ax[0].imshow(np.abs(I[:,:,50,1,1]))
          # ax[1].imshow(np.angle(I[:,:,50,1,1]))
          # plt.show()

          # Make sure the directory exist
          if not os.path.isdir("MRImages/{:s}/HCR{:d}/mat".format(sim,Hcr)):
            os.makedirs("MRImages/{:s}/HCR{:d}/mat".format(sim,Hcr), exist_ok=True)

          # Add noise
          In = add_cpx_noise(I, relative_std=0.025, mask=1)

          # for fr in range(1):
          #   for i in [2]:
          #     M = np.transpose(np.abs(In[:,:,:,i,fr]), (0,2,1))
          #     P = np.transpose(VENC_cms/np.pi*np.angle(In[:,:,:,i,fr]), (0,2,1))
          #     multi_slice_viewer([M, P], caxis=[-40, 40])
          #     plt.show()

          # Create data to export in mat format
          data = {'MR_FFE_FH': np.abs(I[:,:,:,2,:]),
                  'MR_FFE_AP': np.abs(I[:,:,:,0,:]),
                  'MR_FFE_RL': np.abs(I[:,:,:,1,:]),
                  'MR_PCA_FH': VENC_cms/np.pi*np.angle(I[:,:,:,2,:]),
                  'MR_PCA_AP': VENC_cms/np.pi*np.angle(I[:,:,:,0,:]),
                  'MR_PCA_RL': VENC_cms/np.pi*np.angle(I[:,:,:,1,:]),
                  'voxel_MR': 1000.0*(FOV/RES).reshape((1, 3)),
                  'VENC': VENC_cms,
                  'heart_rate': 64.034151547,
                  'type': 'DCM'}
          ndata = {'MR_FFE_FH': np.abs(In[:,:,:,2,:]),
                  'MR_FFE_AP': np.abs(In[:,:,:,0,:]),
                  'MR_FFE_RL': np.abs(In[:,:,:,1,:]),
                  'MR_PCA_FH': VENC_cms/np.pi*np.angle(In[:,:,:,2,:]),
                  'MR_PCA_AP': VENC_cms/np.pi*np.angle(In[:,:,:,0,:]),
                  'MR_PCA_RL': VENC_cms/np.pi*np.angle(In[:,:,:,1,:]),
                  'voxel_MR': 1000.0*(FOV/RES).reshape((1, 3)),
                  'VENC': VENC_cms,
                  'heart_rate': 64.034151547,
                  'type': 'DCM'}

          # Export mats
          savemat("MRImages/{:s}/HCR{:d}/mat/{:s}_V{:.0f}.mat".format(sim,Hcr,seq,VENC_cms), {'data': data})
          savemat("MRImages/{:s}/HCR{:d}/mat/{:s}_V{:.0f}_noisy.mat".format(sim,Hcr,seq,VENC_cms), {'data': ndata})
