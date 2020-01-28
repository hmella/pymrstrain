from PyMRStrain import *
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":

  # Parameters
  p = Parameters_2D(mesh_resolution = 0.00025)
  p['time_steps'] = 30
  np.save("p.npy", p)

  # Create image
  I = Image(FOV=np.array([0.1, 0.1]),
            resolution=np.array([70, 70]), 
            type="Tagging",
            encoding_frequency=np.array([900, 900]),
            encoding_angle=np.pi/4,
            flip_angle=15*np.pi/180,
            T1=0.85)

  # Generator
  g = Generator(p, I)

  # Element
  FE = VectorElement("triangle")

  # Mesh and fem space
#  mesh_v = Mesh('mesh/mesh.msh')
  mesh = fem_ventricle_geometry(p['R_en'], p['tau'], p['mesh_resolution'], filename='mesh/mesh.msh')
  V = FunctionSpace(mesh, FE)

  # Create phantom object
  phantom = DefaultPhantom(p, function_space=V)

  # Generation
  g.phantom = phantom
  t, mask = g.get_image()

  # Add noise
  add_noise_to_SPAMM(t, mask, 0.03)

  # Plot
  if rank==0:
#    u_min, u_max = t_v.min(), t_v.max()
#    for i in range(30):
#      fig = plt.imshow(t_v[:, :, i, 0], cmap=plt.get_cmap('Greys_r'), 
#                       interpolation='nearest', vmin=0.5*u_min, vmax=u_max)
#      fig.axes.get_xaxis().set_visible(False)
#      fig.axes.get_yaxis().set_visible(False)
#      plt.savefig('tagging_{:04d}'.format(i))

    fig, ax = plt.subplots(1, 2)
    tracker = IndexTracker(ax, np.real(t[:,:,0,:]), np.real(t[:,:,1,:]))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.suptitle('Tagging volunteer {:d}'.format(rank))
    plt.show()
