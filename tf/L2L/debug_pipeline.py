from __future__ import print_function
import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
import util
from preprocess import Preprocess

tf.set_random_seed(20)
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
layer_width = 10
momentum = False
#########################

io_path = None
problem = problems.Mnist(args={'minval':-100, 'maxval':100, 'dims':10, 'gog': second_derivatives})
optimizer = meta_optimizer.mlp(problem, path=io_path, args={'second_derivatives': second_derivatives,
                                                                      'num_layers': 1, 'learning_rate': learning_rate,
                                                                      'meta_learning_rate': 0.01,
                                                                      'momentum': momentum, 'layer_width': layer_width,
                                                                      'preprocess': preprocess})

loss, update, reset, min = optimizer.minimize()
flat_grads, prep_grads, deltas = optimizer.debug_info
optimizer_variables = [optimizer.w_1, optimizer.b_1,  optimizer.w_out, optimizer.b_out]
mean_optim_variables = [tf.reduce_mean(optimizer.w_1), tf.reduce_mean(optimizer.b_1),
                                tf.reduce_mean(optimizer.w_out), optimizer.b_out[0][0]]

problem_variables = optimizer.problem.variables
mean_problem_variables = [tf.reduce_mean(variable) for variable in optimizer.problem.variables]

iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())

