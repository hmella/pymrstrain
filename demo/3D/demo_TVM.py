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

  # Create image
  I = Image(FOV=np.array([0.225, 0.225, 0.275]),
            resolution=np.array([40, 40, 30]),
            center=np.array([0.01,-0.015,-0.03]),
            type="TVM")

  # Element
  FE = VectorElement('tetrahedron10')

  # Mesh and fem space
  mesh = Mesh('mesh/mesh10.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = Phantom3D(function_space=V, time_steps=p['time_steps'], path='files/')

  # Generator
  g = Generator(p, I, phantom=phantom, debug=True)
  v, mask = g.get_image()

  # Add noise
  venc  = 0.15
  sigma = 0.01
  add_noise_to_PC(venc, sigma, v, mask, wrap=True)

  # Plot
  if rank==0:
    # fig, ax = plt.subplots(1, 2)
    # tracker = IndexTracker(ax, v[:,:,5,0,:], v[:,:,5,2,:])
    # fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    # fig.suptitle('TVM volunteer')
    # plt.show()

    # Export file
    export_image(0, 0, path="tvm_image", name="I", data=v)
