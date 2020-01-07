import numpy as np

# Function
class Function(object):
    def __init__(self, V):
        self.V = V
        self.array = np.zeros([V.num_dofs(), ])

    # Function space
    def function_space(self):
        return self.V
        
    # Assign method
    def assign(self, u):
        self.array[:] = u
        
    # Vector
    def vector(self):
        return self.array
        
    # Copy vector values
    def copy_values(self):
      return np.copy(self.array)
