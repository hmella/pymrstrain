from PyMRStrain.FEMGeometry import *
from meshio import read as meshio_read

# Mesh class


class Mesh(object):
    def __init__(self, path):
        self.mesh = meshio_read(path)
        self.cell = self.cell_type()

    # Cell type
    def cell_type(self):
        cell = [i for i in self.mesh.cell_data.keys() if i in CELL_TYPES_3]
        if cell == []:
            cell = [i for i in self.mesh.cell_data.keys() if i in CELL_TYPES_2]
        # return cells_dict_mesh[cell[0]]
        return cell[0]

    # Facet type
    def facet_type(self):
        facet = [i for i in self.mesh.cell_data.keys() if i not in [self.cell]]
        if facet == []:
            facet = [i for i in self.mesh.cell_data.keys() if i not in [
                self.cell]]
        return facet[0]

    # Return number of cells
    def num_cells(self):
        return self.cells_connectivity().shape[0]

    # Return number of vertices
    def num_vertices(self):
        return self.vertex_coordinates().shape[0]

    # Return number of boundary facets
    def num_boundary_facets(self):
        return self.facets_connectivity().shape[0]

    # Return the geometric dimension of the mesh
    def geometric_dimension(self):
        if self.cell in CELL_TYPES_2:
            gdim = 2
        elif self.cell in CELL_TYPES_3:
            gdim = 3
        return gdim

    # Return node coordinates
    def vertex_coordinates(self):
        return self.mesh.points

    # Return cells connectivity
    def cells_connectivity(self):
        return self.mesh.cells[self.cell]

    # Return cell vertices
    def cell_vertices(self, i):
        return self.mesh.cell[self.cell][i, :]

    # Return boundary facets connectivity
    def facets_connectivity(self):
        return self.mesh.cell[self.facet]

    # Return facet vertices
    def facet_vertices(self, i):
        return self.mesh.cell[self.facet][i]

    # Return facet markers
    def facet_markers(self):
        return self.mesh.field_data[self.facet]["gmsh:physical"]

    # Move mesh
    def move(self, u):
        # Function space shape
        shape = u.function_space().element_shape()

        # Displacement
        d = u.vector().reshape((-1, shape[0]))

        # Move
        for i in range(shape[0]):
            self.vertex_coordinates()[:, i] += d[:, i]
