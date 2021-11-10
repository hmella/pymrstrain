import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

  # Number of spins and resolutions
  nb_spins = [1000, 10000, 50000, 100000, 200000]
  res = [16, 32, 64, 128, 256]

  # Arrays to store the runtimes
  time_spins = np.loadtxt('times_spins.txt')
  time_spins_b = np.loadtxt('times_spins_b.txt')
  time_res = np.loadtxt('times_res.txt')

  print(time_spins)
  print(time_res)

  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  # axs[0].plot(nb_spins, np.mean(time_spins,axis=1))
  # axs[0].plot(nb_spins, np.mean(time_spins_b,axis=1))
  # axs[1].plot(res, np.mean(time_res,axis=1))
  axs[0].plot(nb_spins, time_spins)
  axs[0].plot(nb_spins, time_spins_b)
  axs[0].legend(['1 proc','4 proc'])
  axs[1].plot(res, time_res)
  plt.show()


