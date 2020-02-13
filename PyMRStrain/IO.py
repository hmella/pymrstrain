import os
import pickle

import meshio
import numpy as np
from scipy.io import savemat


# Save Python objects
def save_pyobject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)

# Load Python objects
def load_pyobject(filename):
    with open(filename, 'rb') as output:
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
def scale_image(I,mag=True,pha=False,real=False,compl=False):

    # slope and intercept
    ScaleIntercept = np.ceil(np.abs(I).max())
    ScaleSlope =  2**12

    # Data extraction
    if mag:
        I_mag = (ScaleSlope*np.abs(I)+ScaleIntercept).astype(np.uint16)
    if pha:
        I_pha = (ScaleSlope*1000*(np.angle(I)+np.pi)).astype(np.uint16)
    if real:
        I_real = (ScaleSlope*(np.real(I)+ScaleIntercept)).astype(np.uint16)
    if compl:
        I_comp = (ScaleSlope*(np.imag(I)+ScaleIntercept)).astype(np.uint16)

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
