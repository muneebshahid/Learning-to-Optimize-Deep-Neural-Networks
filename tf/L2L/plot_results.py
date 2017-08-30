import numpy as np
import glob
from matplotlib import pyplot as plt
import os

files = glob.glob('../../../results/plots/conv/*')
files = glob.glob('../../../results/plots/mlp/*')
x = range(400)

legends = []
for file in files:
    l_file = np.loadtxt(file)
    legend, = plt.plot(x, l_file, label=os.path.basename(file))
    legends.append(legend)

plt.legend(legends)
plt.show()
