import numpy as np

# Spins positions
x = np.array([[0.25,0.25],
              [0.1,0],
              [0,0],
              [1,0],
              [2,0],
              [3,0]])

# Pixels centers
#      |0|1|2|
#      |3|4|5|
#      |6|7|8|
c = [np.array([-1,0,1,-1,0,1,-1,0,1]),
     np.array([-1,-1,-1,0,0,0,1,1,1])]
width = 1.0
shape = [3,3]

# Connectivity
p2s = [[],[],[],
       [],[0,1,2],[3],
       [],[],[]]                     # pixel-to-spins map
s2p = -np.ones([x.shape[0],],dtype=np.int64)  # spin-to-pixel map
n = len(p2s)
for i in range(n):
    s2p[p2s[i]] = i
print(s2p)

# Spins positions wrt to the lower-left voxel corner
xpx = x[:,:]-(np.array([c[0][s2p],c[1][s2p]])-0.5*width).T

# Displacement field
u = np.array([[1.0,0.24],
              [0,0],
              [0,0],
              [0,0],
              [0,0],
              [0,0]])

print(xpx.T)
print(u.T)
print((xpx+u).T)
print()

# Displacement in terms of pixels
pixel_u = ((xpx+u)/width).astype(int)
subpixel_u = xpx+u - pixel_u

# (pixel_u, subpixel_u) = np.divmod(u, width)
# (pixel_u_2, subpixel_u_2) = np.divmod(subpixel_u + xpx, 0.5*width)

print(pixel_u.T)
print(subpixel_u.T)
print()

# Change spins connectivity according to the new positions
for j in range(x.shape[0]):
    if s2p[j] != -1:
        s2p[j] += shape[1]*pixel_u[j,1] + pixel_u[j,0]
print(p2s)
print(s2p)

# Update pixel-to-spins connectivity
p2s = [[] for j in range(n)]
for spin, pixel in enumerate(s2p):
  if pixel != -1:
    p2s[pixel].append(spin)
print(p2s)

# Update relative spins positions
xpx = subpixel_u
print(xpx)
