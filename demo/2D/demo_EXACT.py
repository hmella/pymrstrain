from PyMRStrain import *
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

  # Parameters
  p = Parameters_2D(mesh_resolution = 0.00035)
  p["time_steps"] = 5
  p["sigma"] = 1e-05

  # Create complimentary images
  FOV = 0.1
  N   = 70
  ke  = 785
  alpha = 15*np.pi/180
  beta  = 90*np.pi/180
  I0 = EXACTImage(FOV=np.array([FOV, FOV]),
            resolution=np.array([N, N]))

  # Element
  FE = VectorElement("triangle")

  # Mesh and fem space
  mesh = fem_ventricle_geometry(p['R_en'], p['tau'], p['mesh_resolution'], filename='mesh/mesh.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = DefaultPhantom(p, time_steps=p["time_steps"],function_space=V, patient=False)

  # Generator
  g = Generator(p, I0, phantom=phantom, debug=True)

  # Generation
  u, mask = g.get_image()

  if rank==0:

    # Scale image
    o = scale_image(u,mag=True,pha=False,real=False,compl=False)
    # t = o["magnitude"]["Image"]

    # Plot generated images
    fig, ax = plt.subplots(1, 2)
    tracker = IndexTracker(ax, np.real(u[:,:,0,:]), np.real(u[:,:,1,:]))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.suptitle('Tagging volunteer {:d}'.format(rank))
    plt.show()