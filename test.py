import numpy as np

# Resolution
r = [3,3,3]
pxsz = [1,1,1]

# Grid
X = np.meshgrid(np.linspace(-1,1,r[0]),np.linspace(-1,1,r[1]),np.linspace(-1,1,r[2]))
Xf = [x.flatten('F') for x in X]
# for x in Xf:
#   print(x)

Z = [Xf[2].min(), Xf[2].max()]

# Number of additional slices
new_slices = [2, 2]
r_slice = r[0]*r[1]
for i in range(len(new_slices)):
  for j in range(new_slices[i]):
    for k in range(r[2]):
      if k != 2:
        Xf[k] = np.append(Xf[k], Xf[k][0:r_slice], axis=0)
      else:
        Xf[k] = np.insert(Xf[k], i*Xf[k].size, Z[i]*np.ones([r_slice,]) + (j+1)*pxsz[i]*(-1)**(i+1))

for x in Xf:
  print(x)
