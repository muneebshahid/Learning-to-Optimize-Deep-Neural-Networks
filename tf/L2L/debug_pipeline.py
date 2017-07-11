from __future__ import print_function
import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
import problems
import meta_optimizers
from preprocess import Preprocess
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(0)
preprocess = None#[Preprocess.log_sign, {'k': 10}]

second_derivatives = False
#########################
epochs = 100000
epoch_interval = 1
eval_interval = 50000
validation_epochs = 50
test_epochs = 500
#########################
learning_rate = 0.0001
layer_width = 50
momentum = False
meta_learning_rate = .01
#########################
meta = True
flag_optim = 'mlp'

problem = problems.ElementwiseSquare(args={'meta': meta, 'minval':-10, 'maxval':10, 'dims':1, 'gog': False, 'path': 'cifar', 'conv': False})
if meta:
    io_path = None#util.get_model_path('', '1000000_FINAL')
    if flag_optim == 'mlp':
        optim = meta_optimizers.MlpXHistoryBin(problem, path=io_path, args={'second_derivatives': False,
                                                                              'num_layers': 1, 'learning_rate': learning_rate,
                                                                              'meta_learning_rate': meta_learning_rate,
                                                                              'layer_width': layer_width,
                                                                              'preprocess': preprocess, 'limit': 5, 'hidden_layers': 1})
    else:
        optim = meta_optimizers.l2l(problem, path=None, args={'second_derivatives': False,
                                                                 'state_size': 20, 'num_layers': 2,
                                                                 'unroll_len': 20,
                                                                 'learning_rate': 0.001,
                                                                 'meta_learning_rate': 0.00001,
                                                                  'optim_per_epoch': 1,
                                                                 'preprocess': preprocess})

    optim.build()
    updates, loss, meta_step = optim.ops_updates, optim.ops_loss, optim.ops_meta_step
    mean_optim_variables = [tf.reduce_mean(variable) for variable in optim.optimizer_variables]
    norm_optim_variables = [tf.norm(variable) for variable in optim.optimizer_variables]
    mean_deltas = [tf.reduce_mean(delta) for delta in optim.ops_step['deltas']]
    norm_deltas = [tf.norm(delta) for delta in optim.ops_step['deltas']]


else:
    if flag_optim == 'Adam':
        optim = tf.train.AdamOptimizer(meta_learning_rate)
    else:
        optim = tf.train.GradientDescentOptimizer(meta_learning_rate)
    loss = problem.loss(problem.variables)
    meta_step = optim.minimize(loss)
    updates = []

norm_problem_variables = [tf.norm(variable) for variable in problem.variables]
input_grads = tf.gradients(problem.loss(problem.variables), problem.variables)
input_grads_norm = [tf.norm(grad) for grad in input_grads]
optim_grad = tf.gradients(optim.loss(optim.ops_step['x_next']), optim.optimizer_variables)
optim_grad_norm = [tf.norm(grad) for grad in optim_grad]

# for i, grad in enumerate(grad_optim):
#     name = 'w' if (i % 2) == 0 else 'b'
#     name += ('_' + str(i))
#     tf.summary.histogram('meta_gradients_' + name, grad)for i, grad in enumerate(grad_optim):
#     name = 'w' if (i % 2) == 0 else 'b'
#     name += ('_' + str(i))
#     tf.summary.histogram('meta_gradients_' + name, grad)

iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())
tf.train.start_queue_runners(iis)
if meta:
    optim.set_session(iis)
    optim.init_with_session()
update_summaries = False
if update_summaries:
    all_summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tf_summary/')
    writer.add_graph(iis.graph)

def write_to_file(f_name, all_variables):
    final_dump = None
    for curr_variable in all_variables:
        if final_dump is None:
            final_dump = curr_variable
        else:
            final_dump = np.hstack((final_dump, curr_variable))
    with open(f_name, 'a') as log_file:
        for variable in final_dump:
            log_file.write(str(variable) + ' ')
        log_file.write('\n')

def itr(itr, print_interval=1000, write_interval=None, reset_interval=None):
    global all_summ
    loss_final = 0
    print('current loss: ', np.log10(iis.run(loss)))
    total_time = 0
    for i in range(itr):
        if reset_interval is not None and (i + 1) % reset_interval == 0:
            iis.run(optim.ops_reset)
            optim.init_with_session(iis)
        start = timer()
        if not update_summaries:
            all_summ = []
        _, _, l, summaries, run_step, hist, grad_norm = iis.run([meta_step, updates, loss, all_summ,
                                                               optim.ops_step,
                                                               optim.variable_history,
                                                               optim_grad_norm])
        if update_summaries:
            writer.add_summary(summaries, i)
        end = timer()
        total_time += (end - start)
        loss_final += l
        if write_interval is not None and (i + 1) % write_interval == 0:
            variables = iis.run(tf.squeeze(optim.problem.variables_flat))
            write_to_file('variables_updates.txt', variables)
        if (i + 1) % print_interval == 0:
            print(i + 1)
            print('-----------')
            print('deltas', run_step['deltas'])
            print('x_next', run_step['x_next'])
            print('history', hist)
            print('O Grad norm', grad_norm)
            print('-----------')
            print('norm_probl: ', iis.run(norm_problem_variables))
            if meta:
                print('norm_optim: ', iis.run(norm_optim_variables))
                print('norm_delta: ', iis.run(norm_deltas))
                print('lrate: ', iis.run(optim.learning_rate))
            print('norm_input_grads: ', iis.run(input_grads_norm))
            print('loss: ', np.log10(loss_final / print_interval), np.log10(l))
            print('time:' , total_time / print_interval)
            loss_final = 0
            total_time = 0
    # if write_interval is not None:
    #     f_data = np.load('variables_updates')
#
# itr(40, 1)
