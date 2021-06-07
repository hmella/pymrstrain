import numpy as np

from PyMRStrain.KSpaceTraj import Cartesian, Radial, Spiral

# Imaging parameters
FOV = np.array([0.3, 0.3], dtype=np.float64)
res = np.array([64, 64], dtype=np.int64)

# Trajectories
c_traj = Cartesian(FOV=FOV, res=res, oversampling=2, lines_per_shot=9)
c_traj.plot_trajectory()

r_traj = Radial(FOV=FOV, res=res, oversampling=2, lines_per_shot=9, spokes=64)
r_traj.plot_trajectory()

parameters = {'k0': 0.1, 'k1': 1.0, 'Slew-rate': 45.0, 'gamma': 63.87e+6}
s_traj = Spiral(FOV=FOV, res=res, oversampling=10, lines_per_shot=3, interleaves=15, parameters=parameters)
s_traj.plot_trajectory()