import numpy as np

# Spins positions
x = np.array([[0,0],
              [0.25,0.25],
              [-0.25,-0.25],
              [1,0],
              [2,0],
              [3,0]])

# Pixels centers
#      |0|1|2|
#      |3|4|5|
c = [np.array([0,1,2,0,1,2]),np.array([0,0,0,-1,-1,-1])]
width = 1.0
shape = [2,3]

# Connectivity
p2s = [[0,1,2], [3], [4]]                     # pixel-to-spins map
s2p = -np.ones([x.shape[0],],dtype=np.int64)  # spin-to-pixel map
n = len(p2s)
for i in range(n):
    s2p[p2s[i]] = i
print(s2p)

# Spins positions wrt to the voxel center
xpx = np.array([(x[i,:]-[c[0][s2p[i]],c[1][s2p[i]]])/width for i in range(x.shape[0])])

# Displacement field
u = np.array([[0.55,-0.6],
              [0,0],
              [0,0],
              [0,0],
              [0,0],
              [0,0]])

# Updated positions
xu = x + u

# Displacement in terms of pixels
(pixel_u, subpixel_u) = np.divmod(u, width)
print((pixel_u,subpixel_u))
print()

# Check if spins have changed its location inside the
(subpixel_u_, subsubpixel_u) = np.divmod(0.5*width - xpx, 0.5*width)
print(subpixel_u_)

# Change spins connectivity according to the new positions
for j in range(x.shape[0]):
    if s2p[j] != -1:
        s2p[j] -= (shape[1]*pixel_u[j,1] + pixel_u[j,0])
print(s2p)
