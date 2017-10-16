import numpy as np
import glob
from matplotlib import pyplot as plt
import os
import ntpath

files = glob.glob('../../../results/plots/conv/*')
files = glob.glob('../../../results/plots/mlp/*')


def write_to_file(f_name, list_var):
    with open(f_name, 'a') as log_file:
        for variable in list_var:
            log_file.write(str(variable) + ' ')
        log_file.write('\n')

total = 50
x = range(total)
legends = []
files.sort()
for file in reversed(files):
    l_file = np.loadtxt(file)
    mean_full = "{:.5f}".format(np.mean(l_file))
    mean_half = "{:.5f}".format(np.mean(l_file[int(total / 2):]))
    print([ntpath.basename(file), mean_full, mean_half])
    write_to_file('lr', [mean_full])
    legend, = plt.plot(x, l_file, label=os.path.basename(file))
    legends.append(legend)

plt.legend(legends)
plt.show()
