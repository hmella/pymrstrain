# Download base image ubuntu 18.04
FROM ubuntu:18.04

# Update Ubuntu Software repository
USER root
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && \
    apt-get install -y apt-utils && \
    apt-get upgrade -y -o Dpkg::Options::="--force-confold"

# Install PyMRStrain dependencies
RUN apt-get install -y python3-tk python3-setuptools cmake git
RUN apt-get install -y libopenmpi-dev mpich screen nano
RUN apt-get install -y build-essential python3-dev python3-pip
RUN pip3 install pybind11 && \
    pip3 install mpi4py && \
    pip3 install meshio && \
    pip3 install scipy && \
    pip3 install matplotlib

RUN cd /tmp/ && git clone https://github.com/eigenteam/eigen-git-mirror.git && \
    cd eigen-git-mirror && mkdir build && cd build/ && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr .. && make install

# Install pymrstrain
RUN mkdir /tmp/pymrstrain/ && mkdir /tmp/pymrstrain/PyMRStrain/
COPY PyMRStrain /tmp/pymrstrain/PyMRStrain/
COPY setup.py /tmp/pymrstrain/
RUN cd /tmp/pymrstrain/ && ls && python3 setup.py install --user && \
    rm -rf build/ tmp/ dist/ PyMRStrain.egg-info/
RUN cd && cd /tmp/ && rm -rf eigen-git-mirror pymrstrain