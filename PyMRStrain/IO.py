import meshio
import numpy as np
import os
from scipy.io import savemat

# Write scalar to vtk
def write_vtk(functions, path=None, name=None):

    # Verify if output folder exists
    directory = os.path.dirname(path)
    if directory != []:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Get mesh
    try:
        mesh = functions.function_space().mesh()
    except:
        mesh = functions[0].function_space().mesh()

    # Prepare data as a dictionary
    point_data = {}
    for i, (u, n) in enumerate(zip(functions,name)):

        # Element shape
        element_shape = u.function_space().element_shape()[0]

        if element_shape == 2:
          d = u.vector().reshape((-1, element_shape))
          data = np.zeros([d.shape[0], d.shape[1]+1])
          data[:,:-1] = d
        else:
          data = u.vector().reshape((-1, element_shape))

        point_data[n] = data

    meshio.write(path, meshio.Mesh(points=mesh.vertex_coordinates(),
                 cells={mesh.cell: mesh.cells_connectivity()},
                 point_data=point_data))


def export_image(data, path=None, name=None):

    if name is None:
        name = 'I'

    # Export data
    savemat(path+'.mat',{name: data})


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
