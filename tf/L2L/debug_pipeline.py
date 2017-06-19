from __future__ import print_function
import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
import problems
import meta_optimizers
from preprocess import Preprocess

tf.set_random_seed(0)
preprocess = [Preprocess.log_sign, {'k': 10}]

second_derivatives = False
#########################
epochs = 500
unroll_len = 20
epoch_interval = 1000
eval_interval = 10
validation_epochs = 50
test_epochs = 500
#########################
learning_rate = .0001
meta_learning_rate = .01
layer_width = 50
#########################
meta = True
flag_optim = 'mlp'

problem = problems.Mnist(args={'meta': meta, 'minval':-100, 'maxval':100, 'dims':2, 'gog': True})
if meta:
    io_path = None#util.get_model_path('', '1000000_FINAL')
    if flag_optim == 'mlp':
        optimizer = meta_optimizers.MlpXHistory(problem, path=io_path, args={'second_derivatives': False,
                                                                              'num_layers': 1, 'learning_rate': learning_rate,
                                                                              'meta_learning_rate': meta_learning_rate,
                                                                              'layer_width': layer_width,
                                                                              'preprocess': preprocess, 'limit': 5, 'hidden_layers': 2})
    else:
        optimizer = meta_optimizers.l2l(problem, path=None, args={'second_derivatives': False,
                                                                 'state_size': 20, 'num_layers': 2,
                                                                 'unroll_len': unroll_len,
                                                                 'learning_rate': 0.001,
                                                                 'meta_learning_rate': 0.01,
                                                                 'preprocess': preprocess})

    step, updates, loss, meta_step, reset = optimizer.build()
    mean_optim_variables = [tf.reduce_mean(variable) for variable in optimizer.trainable_variables]
    norm_optim_variables = [tf.norm(variable) for variable in optimizer.trainable_variables]
    mean_deltas = [tf.reduce_mean(delta) for delta in step['deltas']]
    norm_deltas = [tf.norm(delta) for delta in step['deltas']]

else:
    if flag_optim == 'Adam':
        optimizer = tf.train.AdamOptimizer(meta_learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(meta_learning_rate)
    loss = problem.loss(problem.variables)
    min = optimizer.minimize(loss)
    update = []

mean_problem_variables = [tf.reduce_mean(variable) for variable in problem.variables]
norm_problem_variables = [tf.norm(variable) for variable in problem.variables]
grads = tf.gradients(problem.loss(problem.variables), problem.variables)
mean_grads = [tf.reduce_mean(grad) for grad in grads]
norm_grads = [tf.norm(grad) for grad in grads]

iis = tf.InteractiveSession()
try:
    iis.run(tf.global_variables_initializer())
except:
    iis.run(tf.global_variables_initializer())

def itr(itr, print_interval=1000, reset_interval=None):
    loss_final = 0
    print('current loss: ', np.log10(iis.run(loss)))
    total_time = 0
    for i in range(itr):
        if reset_interval is not None and (i + 1) % reset_interval == 0:
            iis.run(reset)
        start = timer()
        _, l, _ = iis.run([updates, loss, meta_step])
        end = timer()
        total_time += (end - start)
        loss_final += l
        if (i + 1) % print_interval == 0:
            print(i + 1)
            print('norm_probl: ', iis.run(norm_problem_variables))
            if meta:
                print('norm_optim: ', iis.run(norm_optim_variables))
                print('norm_delta: ', iis.run(norm_deltas))
                print('lrate:' , iis.run(optimizer.learning_rate))
            print('norm_grads: ', iis.run(norm_grads))
            print('loss: ', np.log10(loss_final / print_interval), np.log10(l))
            print('time:' , total_time / print_interval)
            loss_final = 0
            total_time = 0




