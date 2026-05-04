import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('./motor_0_20260429T020906_0.h5')
# print(f.name)
# print(list(f.items()))
# print(list(f.attrs.items()))
dfastadc = f['fast_adc']
dfastadc = dfastadc[:]
dfastadc = dfastadc.astype(np.uint16).astype(np.float64)
print(len(dfastadc), ' ',len(dfastadc[0]), ' ', len(dfastadc[0][0]))
dataset = [[] for i in range(len(dfastadc[0]))]
for idx1 in range(len(dataset)):
    for idx2 in range(len(dfastadc)):
        for idx3 in range(len(dfastadc[idx2,idx1])):
            dataset[idx1].append(dfastadc[idx2,idx1,idx3])

# V = dataset[:,:3,:] / 65536.0 * 3.3 * 26.45
plt.subplot(1, 3, 1)
plt.plot(dataset[3][:50])
plt.subplot(1, 3, 2)
plt.plot(dataset[4][:50])
plt.subplot(1, 3, 3)
plt.plot(dataset[5][:50])

plt.show()

f.close()
