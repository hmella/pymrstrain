import numpy as np
import time

# Spins positions
x = np.zeros([20000,2])
x[:,0] = 50
x[:,1] = 50

# Pixels centers
cx, cy = np.meshgrid(np.linspace(0,199,200),np.linspace(0,199,200),sparse=True)
width = 1.0
shape = [200,200]

start = time.time()

for i in range(20):

    # Connectivity
    p2s = [[] for i in range(shape[0]*shape[1])]  # pixel-to-spins map
    p2s[51] = [i for i in range(x.shape[0])]  # pixel-to-spins map
    s2p = -np.ones([x.shape[0],],dtype=np.int64)  # spin-to-pixel map
    n = len(p2s)
    for i in range(n):
        s2p[p2s[i]] = i

    # Spins positions wrt to the voxel center
    xpx = np.array([x[i,:]/width for i in range(x.shape[0])])

    # Displacement field
    u = np.random.rand(20000,2)
    u /= np.abs(u).max()
    u *= 10

    # Updated positions
    xu = x + u

    # Displacement in terms of pixels
    (pixel_u, subpixel_u) = np.divmod(u, width)
    (pixel_u_2, subpixel_u_2) = np.divmod(subpixel_u, 0.5*width)

    # Change spins connectivity according to the new positions
    for j in range(x.shape[0]):
        if s2p[j] != -1:
            s2p[j] -= shape[1]*pixel_u[j,1] + shape[1]*pixel_u_2[j,1]
            s2p[j] += pixel_u[j,0] + pixel_u_2[j,0]

    # Update connectivity
    p2s = [[] for j in range(n)]
    for spin, pixel in enumerate(s2p):
      if pixel != -1:
        p2s[pixel].append(spin)

    # Image generation
    I = np.zeros([n,])
    I = np.zeros([n,])
    for i in range(n):
      if p2s[i] != []:
        I[i] = np.mean(x[p2s[i],0])
        I[i] = np.mean(x[p2s[i],1])

end = time.time()
print(end-start)
