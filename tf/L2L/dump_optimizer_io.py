import meta_optimizers
import problems
from preprocess import Preprocess
import tensorflow as tf
import numpy as np
import util
import config

second_derivatives = False
meta = True

flag_optimizer = 'MLP'
preprocess = [Preprocess.log_sign, {'k': 10}]
problems, _ = problems.create_batches_all()

model_id = '1000000'
model_id += '_FINAL'
load_path = '../../../results/mnist/mlp/5/mom/350k/Mlp_model_350000'#util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id)

epochs = 1
num_optim_steps_per_epoch = 1
unroll_len = 1
mean_optim_variables = None
is_rnn = False
if is_rnn:
    configs = config.rnn_norm_history()
else:
    configs = config.mlp_norm_history()

optimizer = meta_optimizers.MlpNormHistoryMultiProblems(problems=problems, path=None, args=configs)
optimizer.build()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
optimizer.set_session(sess)
optimizer.run_init()
optimizer.load(load_path)

flatten = lambda mat_array: [element for mat in mat_array for element in mat]
network_in_dims = configs['network_in_dims']

p_ones = tf.ones([1, network_in_dims], dtype=tf.float32)
n_ones = -tf.ones([1, network_in_dims], dtype=tf.float32)
random_ops = tf.random_uniform([1, network_in_dims], maxval=1.0, minval=-1.0)
random_ops_mean = tf.reduce_mean(random_ops)
output = []

output.append(sess.run([optimizer.network({'inputs': p_ones})[0], tf.reduce_mean(p_ones)]))
output.append(sess.run([optimizer.network({'inputs': n_ones})[0], tf.reduce_mean(n_ones)]))

for i in range(10000):
    if (i + 1) % 100 == 0:
        print(i + 1)
    if is_rnn:
        print('')
    else:
        output.append(sess.run([optimizer.network({'inputs': random_ops})[0], random_ops_mean]))

output = np.array(output)
np.savetxt(load_path + '_optim_io.txt', output, fmt='%7f')
print('Results Dumped')
