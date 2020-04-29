import os
import pickle

import meshio
import numpy as np
from PyMRStrain.MPIUtilities import MPI_comm, MPI_rank
from scipy.io import savemat


# Save Python objects
def save_pyobject(obj, filename, sep_proc=False):
    # Write file
    if not sep_proc:
        if MPI_rank == 0:
            with open(filename, 'wb') as output:
                pickle.dump(obj, output, -1)
    else:
        # Split filename path
        root, ext = os.path.splitext(filename)
        with open(root+'_{:d}'.format(MPI_rank)+ext, 'wb') as output:
            pickle.dump(obj, output, -1)

# Load Python objects
def load_pyobject(filename, sep_proc=False):

    # Load files
    if not sep_proc:
        if MPI_rank==0:
            with open(filename, 'rb') as output:
                obj = pickle.load(output)
        else:
            obj = None

        # Broadcast object
        obj = MPI_comm.bcast(obj, root=0)
    else:
        # Split filename path
        root, ext = os.path.splitext(filename)
        with open(root+'_{:d}'.format(MPI_rank)+ext, 'rb') as output:
            obj = pickle.load(output)

    return obj


# Write Functions to vtk
def write_vtk(functions, path=None, name=None):

    # Verify if output folder exists
    directory = os.path.dirname(path)
    if directory != []:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Get mesh
    try:
        mesh = functions.spins.mesh
    except:
        mesh = functions[0].spins.mesh

    # Prepare data as a dictionary
    point_data = {}
    for i, (u, n) in enumerate(zip(functions,name)):

        # Element shape
        element_shape = u.dim

        if element_shape == 2:
          d = u.vector()
          data = np.zeros([d.shape[0], d.shape[1]+1], dtype=d.dtype)
          data[:,:-1] = d
        else:
          data = u.vector()

        point_data[n] = data

    mesh.point_data = point_data
    meshio.write(path, mesh)


# Export images
def export_image(data, path=None, name=None):

    if name is None:
        name = 'I'

    # Export data
    savemat(path+'.mat',{name: data})


# Scale images
def scale_image(I,mag=True,pha=False,real=False,compl=False,dtype=np.uint64):

    # slope and intercept
    ScaleIntercept = np.ceil(np.abs(I).max())
    ScaleSlope =  np.iinfo(dtype).max/(2*ScaleIntercept)

    # Data extraction
    if mag:
        I_mag = (ScaleSlope*np.abs(I)+ScaleIntercept).astype(dtype)
    if pha:
        I_pha = (ScaleSlope*1000*(np.angle(I)+np.pi)).astype(dtype)
    if real:
        I_real = (ScaleSlope*(np.real(I)+ScaleIntercept)).astype(dtype)
    if compl:
        I_comp = (ScaleSlope*(np.imag(I)+ScaleIntercept)).astype(dtype)

    # Rescaling parameters
    RescaleSlope = 1.0/ScaleSlope
    RescaleIntercept = -ScaleIntercept

    # output
    output = {}
    if mag:
        output["magnitude"] = {"Image": I_mag,
                          "RescaleSlope": RescaleSlope,
                          "RescaleIntercept": RescaleIntercept}
    if pha:
        output["phase"] = {"Image": I_pha,
                      "RescaleSlope": RescaleSlope,
                      "RescaleIntercept": RescaleIntercept}
    if real:
        output["real"] = {"Image": I_real,
                     "RescaleSlope": RescaleSlope,
                     "RescaleIntercept": RescaleIntercept}
    if compl:
        output["complex"] = {"Image": I_comp,
                        "RescaleSlope": RescaleSlope,
                        "RescaleIntercept": RescaleIntercept}

    return output


# Rescale image
def rescale_image(I):

    # Get images, slope and intercept
    Im = dict()
    for (key, value) in I.items():
        Im[key] = I[key]['Image']*I[key]['RescaleSlope'] + I[key]['RescaleIntercept']

    return Im