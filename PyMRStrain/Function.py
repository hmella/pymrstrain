import numpy as np


# Function
class Function:
    def __init__(self, spins, dim=3, type=np.float):
        self.spins = spins
        self.dim = dim
        self.x = spins.samples
        self.type = type
        self.array = np.zeros([self.x.shape[0], dim], dtype=self.type)

    # Assign method
    def assign(self, u):
        self.array[:] = u

    # Vector
    def vector(self):
        return self.array

    # Copy vector values
    def copy_values(self):
      return np.copy(self.array)
