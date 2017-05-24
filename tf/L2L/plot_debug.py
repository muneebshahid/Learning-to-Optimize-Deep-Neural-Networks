from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot3d(x, y, z, labels):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z)
	ax.set_xlabel(labels[0])
	ax.set_ylabel(labels[1])
	ax.set_zlabel(labels[2])
	plt.show(block=False)

def plot2d(x, y, labels):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x, y)
	ax.set_xlabel(labels[0])
	ax.set_ylabel(labels[1])
	plt.show(block=False)

log_label, sign_label, grad_label, delta_label = 'log_grad_magnitude', 'grad_sign', 'grad', 'delta'
debug_files = ['debug_10000.txt', 'debug_120000.txt']
for debug_file in debug_files:
	debug_file_data = np.loadtxt(debug_file)
	limit = -500

	log = debug_file_data[:, 0][limit:]
	sign = debug_file_data[:, 1][limit:]
	grad = debug_file_data[:, 2][limit:]
	delta = debug_file_data[:, 3][limit:]

	plot3d(log, sign, delta, [log_label, sign_label, delta_label])
	plot2d(log, delta, [log_label, delta_label])
	plot2d(grad, delta, [grad_label, delta_label])
