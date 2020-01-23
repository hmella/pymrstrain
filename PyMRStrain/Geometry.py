from subprocess import call
from PyMRStrain.Mesh import *
import os


def fem_ventricle_geometry(parameters, filename=None):
    ''' Generate a 2D geometry of the left-ventricle
    Input: - R_en (float): endocardium radius
           - R_ep (float): epicardium radius
           - tau  (float):

    '''
    R_en = parameters['R_en']
    tau  = parameters['tau']
    h    = parameters['h']
    lc   = parameters['mesh_resolution']

    # Verify if folder exist
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Write dimensions for mesh generation
    file = open(folder+"/dimensions", "w")
    file.write("lc  = {:.5f};\n".format(lc))
    file.write("tau = {:.5f};\n".format(tau))
    file.write("Ren = {:.5f};\n".format(R_en))
    file.write("Rep = {:.5f};".format(R_en+tau))
    file.write("h = {:.5f};".format(h))
    file.write("delta = {:.5f};".format(tau))
    file.close()

    # Write geo file
    geo = '/* Generation of myocardium mesh */'
    geo += '\nInclude "dimensions";'

    geo += '\n\n//This code generates myocardium bidimensional mesh'
    geo += '\n// Labels: Endocardial wall = 1'
    geo += '\n// Epicardial wall  = 2'

    geo += '\nPoint(1) = {0, 0, -0.5*h, lc};'
    geo += '\nPoint(2) = {-Ren, 0, -0.5*h, lc};'
    geo += '\nPoint(3) = {0, Ren, -0.5*h, lc};'
    geo += '\nPoint(4) = {Ren, 0, -0.5*h, lc};'
    geo += '\nPoint(5) = {0, -Ren, -0.5*h, lc};'

    geo += '\n\n// Epicardial wall'
    geo += '\nPoint(6) = {0, 0, -0.5*h, lc};'
    geo += '\nPoint(8) = {0, Rep, -0.5*h, lc};'
    geo += '\nPoint(7) = {-Rep, 0, -0.5*h, lc};'
    geo += '\nPoint(9) = {Rep, 0, -0.5*h, lc};'
    geo += '\nPoint(10) = {0, -Rep, -0.5*h, lc};'

    geo += '\n\n// Inner wall'
    geo += '\nPoint(11) = {0, 0, -0.5*h, lc};'
    geo += '\nPoint(12) = {-Ren+delta, 0, -0.5*h, lc};'
    geo += '\nPoint(13) = {0, Ren-delta, -0.5*h, lc};'
    geo += '\nPoint(14) = {Ren-delta, 0, -0.5*h, lc};'
    geo += '\nPoint(15) = {0, -Ren+delta, -0.5*h, lc};'

    geo += '\n\n// Outer wall'
    geo += '\nPoint(16) = {0, 0, -0.5*h, lc};'
    geo += '\nPoint(18) = {0, Rep+delta, -0.5*h, lc};'
    geo += '\nPoint(17) = {-Rep-delta, 0, -0.5*h, lc};'
    geo += '\nPoint(19) = {Rep+delta, 0, -0.5*h, lc};'
    geo += '\nPoint(20) = {0, -Rep-delta, -0.5*h, lc};'

    geo += '\n\n// Circles'
    geo += '\n\nCircle(1) = {2, 1, 3};'
    geo += '\nCircle(2) = {3, 1, 4};'
    geo += '\nCircle(3) = {4, 1, 5};'
    geo += '\nCircle(4) = {5, 1, 2};'

    geo += '\n\nCircle(5) = {7, 6, 8};'
    geo += '\nCircle(6) = {8, 6, 9};'
    geo += '\nCircle(8) = {10, 6, 7};'
    geo += '\nCircle(7) = {9, 6, 10};'

    geo += '\n\nCircle(9) = {12, 1, 13};'
    geo += '\nCircle(10) = {13, 1, 14};'
    geo += '\nCircle(11) = {14, 1, 15};'
    geo += '\nCircle(12) = {15, 1, 12};'

    geo += '\n\nCircle(13) = {17, 16, 18};'
    geo += '\nCircle(14) = {18, 16, 19};'
    geo += '\nCircle(15) = {20, 16, 17};'
    geo += '\nCircle(16) = {19, 16, 20};'

    geo += '\n\n// Ventricle'
    geo += '\nLine Loop(1) = {1,2,3,4};'
    geo += '\nLine Loop(2) = {5,6,7,8};'
    geo += '\nPlane Surface(1) = {2,1};'

    geo += '\n\n// Outside'
    geo += '\nLine Loop(3) = {9,10,11,12};'
    geo += '\nLine Loop(4) = {13,14,15,16};'
    geo += '\nPlane Surface(2) = {4,2};'
    geo += '\nPlane Surface(3) = {1,3};'

    if h != 0:
      geo += '\nExtrude {0, 0, h} {'
      geo += '\nSurface{1,2,3};'
      geo += '\nLayers{Round(h/(lc))};'
      geo += '\n}'
      geo += '\n\nPhysical Volume(1) = {1};'
      geo += '\nPhysical Volume(2) = {2,3};'
    else:
      geo += '\n\nPhysical Surface(1) = {1};'

    # Open file
    file = open(folder+'/mesh.geo', 'w')
    file.write(geo)
    file.close()

    # Mesh generation and conversion
    if h == 0:
      call(['gmsh', folder+'/mesh.geo', '-2', '-v', '0', '-o', filename])
    else:
      call(['gmsh', folder+'/mesh.geo', '-3', '-v', '0', '-o', filename])

    # Remove temp files
    # call(['rm', folder+'/dimensions', folder+'/mesh.geo'])

    # Read mesh from file
    mesh = Mesh(filename)

    return mesh
