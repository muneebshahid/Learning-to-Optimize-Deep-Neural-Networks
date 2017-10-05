import numpy as np
import glob
from matplotlib import pyplot as plt
import os
import ntpath

files = glob.glob('../../../results/plots/conv/*')
files = glob.glob('../../../results/plots/mlp/*')

total = 50
x = range(total)

legends = []
for file in files:
    l_file = np.loadtxt(file)
    mean_full = "{:.5f}".format(np.mean(l_file))
    mean_half = "{:.5f}".format(np.mean(l_file[int(total / 2):]))
    print([ntpath.basename(file), mean_full, mean_half])
    legend, = plt.plot(x, l_file, label=os.path.basename(file))
    legends.append(legend)

plt.legend(legends)
plt.show()
