from subprocess import call
from PyMRStrain.Mesh import *
import os


def fem_ventricle_geometry(R_en, tau, resolution, filename=None):
    ''' Generate a 2D geometry of the left-ventricle
    Input: - R_en (float): endocardium radius
           - R_ep (float): epicardium radius
           - tau  (float): 

    '''
    # Verify if folder exist
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Write dimensions for mesh generation
    file = open(folder+"/dimensions", "w")
    file.write("lc  = {:.5f};\n".format(resolution))
    file.write("tau = {:.5f};\n".format(tau))
    file.write("Ren = {:.5f};\n".format(R_en))
    file.write("Rep = {:.5f};".format(R_en+tau))
    file.close()

    # Write geo file
    geo = '''/* Generation of myocardium mesh */
              Include "dimensions";

              /* 
                This code generates myocardium bidimensional mesh
                Labels: Endocardial wall = 1
                        Epicardial wall  = 2
              */

              // Endocardial wall
              Point(1) = {0, 0, 0, lc};
              Point(2) = {-Ren, 0, 0, lc};
              Point(3) = {0, Ren, 0, lc};
              Point(4) = {Ren, 0, 0, lc};
              Point(5) = {0, -Ren, 0, lc};

              Circle(1) = {2, 1, 3};
              Circle(2) = {3, 1, 4};
              Circle(3) = {4, 1, 5};
              Circle(4) = {5, 1, 2};

              // Epicardial wall
              Point(6) = {0, 0, 0, lc};
              Point(7) = {-Rep, 0, 0, lc};
              Point(8) = {0, Rep, 0, lc};
              Point(9) = {Rep, 0, 0, lc};
              Point(10) = {0, -Rep, 0, lc};

              Circle(5) = {7, 6, 8};
              Circle(6) = {8, 6, 9};
              Circle(7) = {9, 6, 10};
              Circle(8) = {10, 6, 7};

              Line Loop(1) = {1,2,3,4};
              Line Loop(2) = {5,6,7,8};
              Plane Surface(1) = {2,1};

              // Labels
              Physical Line(1) = {1,2,3,4};
              Physical Line(2) = {5,6,7,8};
              Physical Surface(1) = {1};
              '''

    # Open file
    file = open(folder+'/mesh.geo', 'w')
    file.write(geo)
    file.close()

    # Mesh generation and conversion
    call(['gmsh', folder+'/mesh.geo', '-2', '-v', '0', '-o', filename])

    # Remove temp files
    call(['rm', folder+'/dimensions', folder+'/mesh.geo'])

    # Read mesh from file
    mesh = Mesh(filename)

    return mesh
