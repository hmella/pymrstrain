from subprocess import call
import os

if __name__=='__main__':
  # Folder with vtk files
  DIR = 'files/'

  # Number of files in folder
  N = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
  
  # Find displacement
  for i in range(20):  
    # Conversion
    filename = DIR+'u_{:04d}.vtk'.format(i)
    call(['meshio-convert', filename, 'u_{:04d}.msh'.format(i)])
    filename = DIR+'v_{:04d}.vtk'.format(i)
    call(['meshio-convert', filename, 'v_{:04d}.msh'.format(i)])
