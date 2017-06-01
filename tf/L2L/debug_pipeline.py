from __future__ import print_function
import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
import util
from preprocess import Preprocess

tf.set_random_seed(0)
preprocess = [Preprocess.log_sign, {'k': 10}]

second_derivatives = False
#########################
epochs = 500
num_optim_steps_per_epoch = 1
unroll_len = 1
epoch_interval = 1000
eval_interval = 10
validation_epochs = 50
test_epochs = 500
#########################
learning_rate = 0.0001
meta_learning_rate = 0.01
layer_width = 20
momentum = False
#########################
meta = True
flag_optim = 'Adam'

problem = problems.Mnist(args={'meta': meta, 'minval':-100, 'maxval':100, 'dims':10, 'gog': second_derivatives})
if meta:
    io_path = None#util.get_model_path('', '1000000_FINAL')
    optimizer = meta_optimizer.mlp(problem, path=io_path, args={'second_derivatives': second_derivatives,
                                                                          'num_layers': 1, 'learning_rate': learning_rate,
                                                                          'meta_learning_rate': meta_learning_rate,
                                                                          'momentum': momentum, 'layer_width': layer_width,
                                                                          'preprocess': preprocess})
    # optimizer = meta_optimizer.l2l(problem, path=None, args={'second_derivatives': second_derivatives,
    #                                                          'state_size': 20, 'num_layers': 2,
    #                                                          'unroll_len': unroll_len,
    #                                                          'learning_rate': 0.001,
    #                                                          'meta_learning_rate': 0.01,
    #                                                          'preprocess': preprocess})

    loss, update, reset, min = optimizer.minimize()
    reset_optim = optimizer.reset_optimizer()
    flat_grads, prep_grads, deltas = optimizer.debug_info
    optimizer_variables = [optimizer.w_1, optimizer.b_1, optimizer.w_out, optimizer.b_out]
    mean_optim_variables = [tf.reduce_mean(optimizer.w_1), tf.reduce_mean(optimizer.b_1),
                            tf.reduce_mean(optimizer.w_out), optimizer.b_out[0][0]]
    mean_deltas = [tf.reduce_mean(delta) for delta in deltas]

else:
    if flag_optim == 'Adam':
        optimizer = tf.train.AdamOptimizer(meta_learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(meta_learning_rate)
    loss = problem.loss(problem.variables)
    min = optimizer.minimize(loss)
    update = []

mean_problem_variables = [tf.reduce_mean(variable) for variable in problem.variables]
grads = tf.gradients(problem.loss(problem.variables), problem.variables)
mean_grads = [tf.reduce_mean(grad) for grad in grads]

def p(i):
    f, u, l, d, m = iis.run([flat_grads, update, loss, deltas, min])
    return f[i][-20:], u[0][i][-20:], l, d[i][-20:]


iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())
def print_min_max():
    shape = len()
    mat = np.zeros((flat_grads, deltas))
    mat[:, 1::2] = 1000
    for i, (grad, delta) in zip(flat_grads, deltas):
        g, d = iis.run([grad, delta])
        min_g = np.min(g)
        max_g = np.max(g)
        min_d = np.min(d)
        max_d = np.max(d)


def itr(itr, print_interval=1000, reset_interval=None):
    loss_final = 0
    print('current loss: ', np.log10(iis.run(loss)))
    for i in range(itr):
        if reset_interval is not None and (i + 1) % reset_interval == 0:
            iis.run(reset)
        _, l, _ = iis.run([update, loss, min])
        loss_final += l
        if (i + 1) % print_interval == 0:
            print(i + 1)
            print('probl: ', iis.run(mean_problem_variables))
            if meta:
                print('optim: ', iis.run(mean_optim_variables))
                print('delta: ', iis.run(mean_deltas))
            print('grads: ', iis.run(mean_grads))
            print('loss: ', np.log10(loss_final / print_interval), np.log10(l))
            loss_final = 0




