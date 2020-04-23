import numpy as np

# Array orderings, which depend on the emasurement
# direction
order = ['F','C']

# Measurement and phase directions
# The first element of m_dirs[i] (with i the encoding)
# direction) denotes the measurement direction, and the
# second the phase direction
m_dirs = [[0,1], [1,0]]

# Check the croping indices inside the function
# PyMRStrain.MRImaging.acq_to_res
def build_idx(n_lines, acq_matrix, dir):
    # Number of additional lines
    add = int(n_lines/2*acq_matrix[dir[0]])

    # Flattened array indices
    idx = np.array([add,-add])
    # idx = np.flip(np.sort(idx))

    # Check if the last element is zero
    if idx[-1] == 0:
        idx[-1] = np.prod(acq_matrix)

    return idx

# Cropping ranges to correct the shape of generated images
def cropping_ranges(im_resolution, gen_resolution, ovrs_fac):

    # Abbreviations
    gr = gen_resolution
    ir = im_resolution

    # Number of measurements in the extended kspace
    n = gr - ovrs_fac*ir

    # Shift in the kspace due to the oversampling
    shift = np.mod(n, 2)

    # Ranges
    r = [int(n[0]/2) + shift[0], int(n[0]/2 + shift[0] + ovrs_fac*ir[0])]
    c = [int(n[1]/2) + shift[1], int(n[1]/2 + shift[1] + ovrs_fac*ir[1])]
    dr = [1, ovrs_fac]
    dc = [ovrs_fac, 1]

    return r, c, dr, dc

def iseven(arg):
    return (arg % 2 == 0)

def isodd(arg):
    return (arg % 2 != 0)