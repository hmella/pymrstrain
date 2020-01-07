from PyMRStrain import *
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

  # Parameters
  p = {'t_end': 1.0,
       'dt': 1.0/20.0,
       't': 0.0,
       'time_steps': 20,
       'mesh_resolution': 0.0001}

  # Encoding frequency
  ke = 1000*2*np.pi*0.125

  # Create image
  I0 = DENSEImage(FOV=np.array([0.15, 0.15, 0.2]),
            resolution=np.array([55, 55, 25]),
            center=np.array([0.01,-0.01,-0.03]),
            encoding_frequency=[ke,ke,ke],
            T1=0.85,
            flip_angle=15*np.pi/180)

  # Element
  FE = VectorElement('tetrahedron10')

  # Mesh and fem space
  mesh = Mesh('mesh/mesh10.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = Phantom3D(function_space=V, time_steps=p['time_steps'], path='files/')

  # Generator
  g0 = Generator(p, I0, debug=True)

  # Generation
  g0.phantom = phantom
  u0, u1, mask = g0.get_image()

  # Add noise to images
  un0 = add_noise_to_DENSE_(u0, mask, sigma=0.06)
  un1 = add_noise_to_DENSE_(u1, mask, sigma=0.06)

  # Corrected image
  u = un0 - un1

  # Plot
  if rank==0:
    # Export file
    export_image(u, path='I_DENSE', name='I_DENSE')
    export_image(mask, path='mask_DENSE', name='mask_D')    
    export_image(I0._grid[0], path='X_DENSE', name='X')
    export_image(I0._grid[1], path='Y_DENSE', name='Y')