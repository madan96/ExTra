import numpy as np
import matplotlib.pyplot as plt

d_sa = np.load('/home/rishabh/work/btp/TRAiL/transfer_logs/Dist-sa_FourSmallRooms_11_FourLargeRooms.npy')
# d_sa = np.load('/home/rishabh/work/btp/TRAiL/transfer_logs/Dist-sa_FourSmallRooms_11_NineLargeRooms.npy')
# d_sa = np.load('/home/rishabh/work/btp/TRAiL/transfer_logs/Dist-sa_FourSmallRooms_11_SixLargeRooms.npy')

d_sa = d_sa.flatten()
plt.hist(d_sa, bins=100)
plt.show()