import numpy as np
from scipy.io import savemat
import os

# Write scalar to vtk
def write_scalar_vtk(u, path=None, name=None):
    # Verify if folder exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Mesh
    mesh = u.function_space().mesh()

    # Type of cell
    if mesh.cell=="triangle":
        cell_type = 5
    if mesh.cell=="quad":
        cell_type = 9
    if mesh.cell=="hexahedron":
        cell_type = 12

    # Define DATASET type
    element_shape = u.function_space().element_shape()[0]
    if element_shape == 1:
        DATASET = "SCALARS"
    elif element_shape > 1:
        DATASET = "VECTORS"

    # Points, cells and data
    points = mesh.vertex_coordinates()
    cells  = mesh.cells_connectivity()
    if element_shape == 2:
      d = u.vector().reshape((-1, element_shape))
      data = np.zeros([d.shape[0], d.shape[1]+1])
      data[:,:-1] = d
    else:
      data = u.vector().reshape((-1, element_shape))


    # Open file and write header
    file = open(path, "w")
    file.write("# vtk DataFile Version 2.0\n")
    file.write("Unstructured Grid example\n")
    file.write("ASCII\n")
    file.write("DATASET UNSTRUCTURED_GRID\n")

    # Write DOF coordinates
    file.write("\nPOINTS {:d} float\n".format(points.shape[0]))
    file.close()
    with open(path, 'ba') as f:
      np.savetxt(f, points, fmt='%0.10f')

    # Cells connectivity
    [m, n] = cells.shape
    connectivity = np.zeros([m, n + 1])
    connectivity[:,0] = n

    for i in range(n):
        connectivity[:,i+1] = cells[:, i]

    file = open(path, "a")
    file.write("\nCELLS {:d} {:d}\n".format(m, m*n + m))
    file.close()
    with open(path, 'ba') as f:
      np.savetxt(f, connectivity, fmt='%d')

    # Write cell types
    cell_types = np.zeros([m, 1]) + cell_type
    file = open(path, "a")
    file.write("\nCELL_TYPES %d\n" % (m))
    file.close()
    with open(path, "ba") as f:
      np.savetxt(f, cell_types, fmt='%d')

    # Write scalar values
    file = open(path, "a")
    file.write("\nPOINT_DATA %d\n" % (data.shape[0]))
    file.write(DATASET+" "+name+" float\n")
    if DATASET=="SCALARS": file.write("LOOKUP_TABLE default\n")
    file.close()
    with open(path, "ba") as f:
      np.savetxt(f, data, fmt='%.10f')


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
