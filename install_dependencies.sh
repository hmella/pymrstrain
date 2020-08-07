#!/bin/bash

# Install pymrstrain dependencies
sudo apt-get -y install build-essential python3-dev python3-pip python3-tk \
                 python3-setuptools libopenmpi-dev mpich cmake nano
pip3 install pybind11 --user
pip3 install mpi4py --user
pip3 install scipy --user
pip3 install matplotlib --user

# Download and install matplotlib
sudo apt-get install libfreetype6-dev pkg-config
wget https://github.com/matplotlib/matplotlib/archive/v3.2.2.zip &&
unzip v3.2.2.zip &&
cd matplotlib-3.2.2 &&
python3 setup.py --user &&
cd .. && rm -rf matplotlib-3.2.2 v3.2.2.zip

# Clone, build and install Eigen3-dev
git clone https://github.com/eigenteam/eigen-git-mirror.git && \
cd eigen-git-mirror && mkdir build && cd build/ && \
cmake -DCMAKE_INSTALL_PREFIX=/usr .. && sudo make install && \
cd .. && cd .. && rm -rf eigen-git-mirror
