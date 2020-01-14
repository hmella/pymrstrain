import numpy as np

# Two dimensional meshgrids
def update_s2p2(s2p, pixel_u, resolution):
    s2p[:] += (resolution[1]*pixel_u[:,1] + pixel_u[:,0]).astype(np.int64)
    return s2p

# Three dimensional images with number of slices greater than 1
def update_s2p3(s2p, pixel_u, resolution):
    if resolution[-1] > 1:
        s2p[:] += (resolution[2]*pixel_u[:,0]              # jump between columns
                + resolution[1]*resolution[2]*pixel_u[:,1] # jump between rows
                + pixel_u[:,2]).astype(np.int64)           # jumpt between slices
    else:
        s2p[:] += (resolution[1]*pixel_u[:,1]             # jump betwen rows
               + resolution[0]*resolution[1]*pixel_u[:,2] # jump betwen slices
               + pixel_u[:,0]).astype(np.int64)           # jump between columns
    return s2p
