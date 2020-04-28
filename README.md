# PyMRStrain

PyMRStrain is a open source generator of synthetic Complementary SPAMM and DENSE images.

## Get started
To generate CSPAMM images, in Python simply do
```python
from PyMRStrain import *
import numpy as np

# Parameters
p = Parameters(time_steps=18)

# Encoding frequency
ke = 0.07               # encoding frequency [cycles/mm]
ke = 1000*2*np.pi*ke    # encoding frequency [rad/m]

# Create complimentary image
I = CSPAMMImage(FOV=np.array([0.2, 0.2, 0.04]),
          center=np.array([0.0,0.0,0.03]),
          resolution=np.array([100, 100, 1]),
          encoding_frequency=np.array([ke,ke,0]),
          T1=0.85,
          flip_angle=15*np.pi/180,
          encoding_angle=90*np.pi/180,
          off_resonance=phi,
          kspace_factor=15,
          slice_thickness=0.008,
          oversampling_factor=2,
          phase_profiles=50)

# Spins
spins = Spins(Nb_samples=1000000, parameters=p)

# Create phantom object
phantom = Phantom(spins, p, patient=False, write_vtk=False)

# EPI acquisiton object
artifact = EPI(receiver_bw=128*1000,
               echo_train_length=10,
               off_resonance=200,
               acq_matrix=I.acq_matrix,
               spatial_shift='top-down')

# Generate images
kspace_0, kspace_1, kspace_in, mask = I.generate(artifact, phantom, p)

# Complementary images
u = u0 - u1

# Plots
if MPI_rank==0:
    multi_slice_viewer(np.abs(u[:,:,0,0,:]))
    multi_slice_viewer(np.abs(u[:,:,0,1,:]))
    multi_slice_viewer(np.abs(kspace_0.k[:,:,0,0,:]))
    multi_slice_viewer(np.abs(kspace_0.k[:,:,0,1,:]))
```

and run from a terminal with
```bash
mpirun -n nb_proc python3 foo.py
```
Resulting images look like this:
| CSPAMM image with epi-like artifacts  | CSPAMM kspace with epi-like artifacts |
| ------------- | ------------- |
| ![CSPAMM image](/screenshots/Figure_1.png "CSPAMM image with epi-like artifacts")  | ![CSPAMM image](/screenshots/Figure_2.png "CSPAMM image with epi-like artifacts")  |

## Installation
```bash
git clone https://github.com/hernanmella/pymrstrain
cd pymrstrain
python3 setup.py install
```

## Installation with Docker
Build the docker image using:
```bash
docker build . -t pymrstrain
```

and then do:
```bash
sudo apt-get install x11-xserver-utils && xhost +
```
Finally, to run the image with plotting support just run:
```bash
docker run --rm -it \
   --user="pymrstrain" \
   --env="DISPLAY" \
   --volume="/etc/group:/etc/group:ro" \
   --volume="/etc/passwd:/etc/passwd:ro" \
   --volume="/etc/shadow:/etc/shadow:ro" \
   --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   -v "/home/hernan/Git/pymrstrain/demo":"/home/pymrstrain" pymrstrain
```

## Dependencies
- *Eigen:* [https://github.com/eigenteam/eigen-git-mirror.git](https://github.com/eigenteam/eigen-git-mirror.git)
- *matplotlib:* [https://github.com/matplotlib/matplotlib](https://github.com/matplotlib/matplotlib)
- *meshio:* [https://github.com/nschloe/meshio](https://github.com/nschloe/meshio)
- *MPI4py:* [https://github.com/mpi4py/mpi4py](https://github.com/mpi4py/mpi4py)
- *pybind11:* [https://github.com/pybind/pybind11](https://github.com/pybind/pybind11)
- *SciPy:* [https://www.scipy.org/](https://www.scipy.org/)
