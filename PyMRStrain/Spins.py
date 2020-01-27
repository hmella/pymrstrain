from PyMRStrain.MPIUtilities import MPI_rank, MPI_size
import numpy as np
import meshio

class Spins:
    def __init__(self, Nb_samples=1000, parameters=[], structured=(False, 20)):
        self.Nb_samples = np.ceil(Nb_samples/MPI_size).astype(int)
        self.R_en = parameters['R_en']
        self.R_ep = parameters['R_ep']
        self.R_inner = parameters['R_inner']
        self.R_outer = parameters['R_outer']
        self.structured = structured
        self.h = parameters['h']
        self.center = parameters['center']
        self.samples, self.regions = self.generate_samples()
        self.mesh = self.build_mesh()

    # Generate spins using cylindrical coordinates
    def generate_samples(self):
        # Number of samples in each process
        N = self.Nb_samples

        # Radius, angle and height
        if self.structured[0]:
            # Number of in-plane samples
            Ns = int(np.sqrt(np.ceil(N/self.structured[1])))
            x = np.linspace(-self.R_outer, self.R_outer, Ns) + self.center[0]
            y = np.linspace(-self.R_outer, self.R_outer, Ns) + self.center[1]
            z = np.linspace(-0.5*self.h, 0.5*self.h, self.structured[1]) + self.center[2]
            [x,y,z] = np.meshgrid(x,y,z)
            x = x.flatten('F')
            y = y.flatten('F')
            z = z.flatten('F')
            r = np.sqrt(np.power(x,2) + np.power(y,2))
            tst = (r <= self.R_outer)*(r >= self.R_inner)
            x = x[tst]
            y = y[tst]
            z = z[tst]
            r = r[tst]
            self.Nb_samples = r.size
        else:
            r = np.sqrt(np.random.uniform(self.R_inner**2, self.R_outer**2, size=[N,]))
            theta = np.random.uniform(0.0, 2*np.pi, size=[N,])
            z = np.random.uniform(-0.5*self.h, 0.5*self.h, size=[N,])

            # Cartesian coordinates
            x = r*np.cos(theta) + self.center[0]
            y = r*np.sin(theta) + self.center[1]
            z = z + self.center[2]

        # Separate regions
        inner = r < self.R_en
        outer = r > self.R_ep
        ventricle = (~inner)*(~outer)
        print('Number of spins in process {:d}: {:d}'.format(MPI_rank, x.shape[0]))

        return np.column_stack((x,y,z)), np.column_stack((inner,outer,ventricle))

    def build_mesh(self):
        N = self.Nb_samples
        vertices = np.linspace(0,N-1,N).reshape((N,1))
        return meshio.Mesh(points=self.samples, cells={'vertex': vertices})
