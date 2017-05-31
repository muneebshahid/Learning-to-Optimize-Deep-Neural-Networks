from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import util

fig_folder = 'fig'
flag_optimizer = 'MLP'
model_id = '1000000'
model_id += '_FINAL'
model_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id)
save_path = model_path + '_fig_'
debug_files = [model_path + '_optim_io.txt']

def plot3d(x, y, z, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    path = save_path + labels[0] + '_' + labels[1] + '_' + labels[2]
    plt.savefig(path + '.png')
    plt.show()


def plot2d(x, y, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    path = save_path + labels[0] + '_' + labels[1]
    plt.savefig(path + '.png')
    plt.show()


log_label, sign_label, grad_label, delta_label = 'log_grad_magnitude', 'grad_sign', 'grad', 'delta'

for debug_file in debug_files:
    debug_file_data = np.loadtxt(debug_file)
    limit = 0

    grad = debug_file_data[:, 0][limit:]
    log = debug_file_data[:, 1][limit:]
    sign = debug_file_data[:, 2][limit:]
    delta = debug_file_data[:, 3][limit:]

    plot3d(log, sign, delta, [log_label, sign_label, delta_label])
    plot2d(sign, delta, [sign_label, delta_label])
    plot2d(log, delta, [log_label, delta_label])
    plot2d(grad, delta, [grad_label, delta_label])
