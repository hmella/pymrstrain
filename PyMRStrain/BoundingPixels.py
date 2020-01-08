import numpy as np
from ImageUtilities import getConnectivity

class BoundingPixels(object):
    def __init__(self, image):
        self.image = image
        self.voxel_size = image.voxel_size
        self.voxel_centers = image._get_sparse_grid()

    def connectivity(self):
        print(1)

    def voxel_width(self):
        return image.voxel_size()

class Spins(object):
    def __init__(self, x):
        self.x = x

    def BoundaryDistance(self, BP):
        # Voxels
        v = BP.connectivity()
        n = len(v)
        c = BP.voxel_centers

        # Voxel dimensions
        width = BP.voxel_width()

        # Spins positions
        x = self.x

        # Bounding distances
        # dx0 = [x[v[i]][:,0] - (c[0][i] - 0.5*width) for i in range(n) if v[i] != []]
        # dx1 = [(c[0][i] - 0.5*width) - x[v[i]][:,0] for i in range(n) if v[i] != []]
        # dy0 = [x[v[i]][:,1] - (c[1][i] - 0.5*width) for i in range(n) if v[i] != []]
        # dy1 = [(c[1][i] - 0.5*width) - x[v[i]][:,1] for i in range(n) if v[i] != []]
        dx0 = [x[v[i]][:,0] - (c[0][i] - 0.5*width) for i in range(n)]
        dx1 = [(c[0][i] - 0.5*width) - x[v[i]][:,0] for i in range(n)]
        dy0 = [x[v[i]][:,1] - (c[1][i] - 0.5*width) for i in range(n)]
        dy1 = [(c[1][i] - 0.5*width) - x[v[i]][:,1] for i in range(n)]

    def update_positions(self, x_updated):
        self.x = x_updated
