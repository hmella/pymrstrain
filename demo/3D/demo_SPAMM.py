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
            resolution=np.array([90, 90, 30]),
            center=np.array([0.01,-0.015,-0.03]),
            type="Tagging",
            encoding_frequency=np.array([1000, 1000]),
            flip_angle = np.pi/4,
            T1=1.0)

  # Element
  FE = VectorElement('tetrahedron10')

  # Mesh and fem space
  mesh = Mesh('mesh/mesh10.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = Phantom3D(function_space=V, time_steps=p['time_steps'], path='files/')

  # Generator
  g = Generator(p, I, phantom=phantom, debug=True)
  t, mask = g.get_image()

  # Add noise
  sigma = 0.03
  add_noise_to_SPAMM(t, mask, sigma)

  # Plot
  if rank==0:
    # fig, ax = plt.subplots(1, 2)
    # tracker = IndexTracker(ax, t[:,:,5,:], t[:,:,5,:])
    # fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    # fig.suptitle('DENSE volunteer {:d}'.format(rank))
    # plt.show()

    # Export file
    export_image(0, 0, path="tagging_image", name="I", data=t)
