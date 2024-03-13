import os
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PyMRStrain.KSpaceTraj import Cartesian


if __name__ == '__main__':

  # Preview partial results
  preview = False

  # Import imaging parameters
  with open('PARAMETERS.yaml') as file:
    pars = yaml.load(file, Loader=yaml.FullLoader)

  # Imaging parameters
  print(pars)
  FOV = np.array(pars["Imaging"]["FOV"])
  RES = np.array(pars["Imaging"]["RES"])
  T2star = pars["Imaging"]["T2star"]/1000.0
  VENCs = np.array(pars["Imaging"]["VENC"])
  OFAC = pars["Imaging"]["OVERSAMPLING"]

  # Generate kspace trajectory
  lps = pars['EPI']["LinesPerShot"]
  traj = Cartesian(FOV=FOV[:-1], res=RES[:-1], oversampling=OFAC, lines_per_shot=11, VENC=2.5, plot_seq=True)