# Number of points in cell
TRIANGLE      = 3
QUADRILATERAL = 4
TETRAHEDRON   = 4
HEXAHEDRON    = 8
TETRAHEDRON10 = 10

# Cell types in meshes
CELL_TYPES_2  = ['triangle', 'quad']
CELL_TYPES_3  = ['hexa', 'tetra', 'tetra10']

# Cell to mesh dictionary
cells_dict = {'triangle': 'triangle',
              'tetrahedron': 'tetra',
              'tetrahedron10': 'tetra10',
              'quadrilateral': 'quad',
              'hexahedron': 'hexa'}

# Mesh to cell dictionary
cells_dict_mesh = {'triangle': 'triangle',
                   'tetra': 'tetrahedron',
                   'tetra10': 'tetrahedron10',
                   'quad': 'quadrilateral',
                   'hexahedron': 'hexa'}
