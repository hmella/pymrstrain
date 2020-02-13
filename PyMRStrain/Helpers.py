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
    S = gen_resolution
    s = im_resolution

    # Ranges
    r = [int(0.5*(S[0]-ovrs_fac*s[0])), int(0.5*(S[0]-ovrs_fac*s[0])+ovrs_fac*s[0])]
    c = [int(0.5*(S[1]-ovrs_fac*s[1])), int(0.5*(S[1]-ovrs_fac*s[1])+ovrs_fac*s[1])]
    dr = [1, ovrs_fac]
    dc = [ovrs_fac, 1]    

    return r, c, dr, dc