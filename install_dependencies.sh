# Install pymrstrain dependencies
sudo apt install build-essential python3-dev python3-pip python3-tk python3-setuptools libopenmpi-dev mpich gmsh
sudo pip3 install pybind11
sudo pip3 install mpi4py
sudo pip3 install meshio
sudo pip3 install scipy
sudo pip3 install matplotlib

# Clone, build and install Eigen3-dev
git clone https://github.com/eigenteam/eigen-git-mirror.git && \
cd eigen-git-mirror && mkdir build && cd build/ && \
cmake -DCMAKE_INSTALL_PREFIX=/usr .. && make install && \
cd .. && cd .. && rm -rf eigen-git-mirror 