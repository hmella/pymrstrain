import numpy as np
import matplotlib.pyplot as plt

ncor = np.array([1, 2, 3, 4, 5, 6])
base = np.array([16.69]*6)
para = np.array([16.98, 10.14, 8.73, 8.16, 9.8, 20.9])/base[0]
base = base/base[0]

plt.plot(ncor, base)
plt.plot(ncor, para)
plt.show()
