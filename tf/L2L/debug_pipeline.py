from __future__ import print_function
import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
import problems
import meta_optimizers
from preprocess import Preprocess
import numpy as np
import config

tf.set_random_seed(0)
preprocess = [Preprocess.log_sign, {'k': 10}]

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
meta_learning_rate = .00005#05
#########################
meta = True
flag_optim = 'mlp'

problem = None#problems.ElementwiseSquare(args={'meta': meta, 'minval':-10, 'maxval':10, 'dims':1, 'gog': False, 'path': 'cifar', 'conv': False})
if meta:
    io_path = None#util.get_model_path('', '1000000_FINAL')
    cifar_path = '../../../cifar/'
    problem = problems.cifar10({'prefix': 'train', 'minval': 0, 'maxval': 100, 'conv': True, 'full': True, 'path': cifar_path})
    problem_eval_1 = problems.cifar10({'prefix': 'eval_1', 'minval': 0, 'maxval': 100, 'conv': True, 'full': False, 'path': cifar_path})
    # problem_eval_2 = problems.cifar10(
    #     {'prefix': 'eval_2', 'minval': 0, 'maxval': 100, 'conv': True, 'full': False, 'path': cifar_path})
    # problem = problems.Mnist({'prefix': 'train', 'minval': 0, 'maxval': 100, 'conv': False, 'full': True})
    # problem_eval_1 = problems.Mnist({'prefix': 'eval_1', 'minval': 0, 'maxval': 100, 'conv': False, 'full': False})
    optim = meta_optimizers.AUGOptims([problem], [problem_eval_1], path=io_path, args=config.aug_optim())
    optim.build()
    updates, loss_optim, loss_problem, meta_step, prob_acc = optim.ops_updates, optim.ops_loss, optim.ops_loss_problem, optim.ops_meta_step, optim.ops_prob_acc
    mean_optim_variables = [tf.reduce_mean(variable) for variable in optim.optimizer_variables]
    norm_optim_variables = [tf.norm(variable) for variable in optim.optimizer_variables]
    # norm_deltas = [tf.norm(delta) for step in optim.ops_step for delta in step['deltas']]
else:
    if flag_optim == 'Adam':
        optim = tf.train.AdamOptimizer(meta_learning_rate)
    else:
        optim = tf.train.GradientDescentOptimizer(meta_learning_rate)
    loss_optim = problem.loss(problem.variables)
    meta_step = optim.minimize(loss_optim)
    updates = []
# check_op = tf.add_check_numerics_ops()
norm_problem_variables = [tf.norm(variable) for problem in optim.problems for variable in problem.variables]
# input_grads = tf.gradients(problem.loss(problem.variables), problem.variables)
# input_grads_norm = [tf.norm(grad) for grad in input_grads]
optim_grad = tf.gradients(optim.ops_loss, optim.optimizer_variables)
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
    optim.run_init()
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

check_nan_prob = tf.is_nan(norm_problem_variables)
check_nan_optim = tf.is_nan(norm_optim_variables)

def itr(itr, print_interval=1000, write_interval=None, show_prob=0, reset_interval=None):
    global all_summ
    loss_final_optim = np.zeros(len(loss_optim))
    loss_final_prob = np.zeros(len(loss_problem))
    print('current loss optim: ', iis.run(loss_optim))
    print('current loss prob: ', np.log10(iis.run(loss_problem)))
    total_time = 0
    for i in range(itr):
        problem_index = i % len(optim.problems)
        if reset_interval is not None and (i + 1) % reset_interval == 0:
            optim.run_reset()
        start = timer()
        if not update_summaries:
            all_summ = []
        _, _, loss_optim_run, loss_prob_run, run_step, grad_norm = iis.run([meta_step, updates,
                                                loss_optim,
                                                loss_problem,
                                                optim.ops_step,
                                                optim_grad_norm])
        if True in iis.run(check_nan_prob):
            print('NAN found prob after, exit')
            break
        if True in iis.run(check_nan_optim):
            print('NAN found optim after, exit')
            break

        end = timer()
        total_time += (end - start)
        loss_final_optim += np.array(loss_optim_run)
        loss_final_prob += np.array(loss_prob_run)
        if write_interval is not None and (i + 1) % write_interval == 0:
            variables = iis.run(tf.squeeze(optim.problems.variables_flat))
            write_to_file('variables_updates.txt', variables)
        if (i + 1) % print_interval == 0:
            summaries = iis.run(all_summ)
            if update_summaries:
                writer.add_summary(summaries, i)
            print(i + 1)
            # print('problem: ', problem_index)
            print('-----------')
            # print('deltas', run_step[problem_index]['deltas'])
            # print('x_next', run_step[problem_index]['x_next'])
            # print('history', hist[problem_index])
            print('O Grad norm', grad_norm)
            print('-----------')
            print('norm_probl: ', iis.run(norm_problem_variables))
            if meta:
                print('norm_optim: ', iis.run(norm_optim_variables))
                # print('norm_delta: ', iis.run(norm_deltas))
                # print('lrate: ', iis.run(optim.learning_rate))
            # print('norm_input_grads: ', iis.run(input_grads_norm))
            print('loss optim: ', loss_final_optim / print_interval)
            print('loss prob: ', np.log10(loss_final_prob / print_interval))
            print('acc prob: ', iis.run(prob_acc))
            print('time:' , total_time / print_interval)
            loss_final_optim = 0
            loss_final_prob = 0
            total_time = 0
    # if write_interval is not None:
    #     f_data = np.load('variables_updates')
#
# itr(itr=10000, print_interval=100, reset_interval=50)

# diff_hist_o = tf.Variable(iis.run(optim.dist_mv_avg[1][0]))
# vari_hist_o = tf.Variable(iis.run(optim.vari_hist[1][0]))
# iis.run(tf.variables_initializer([vari_hist_o, diff_hist_o]))
# x_next = iis.run(optim.ops_step)[0]['x_next'][0]
# print(x_next)
#
# itr(1, print_interval=100, reset_interval=100)
# diff_hist_n = tf.Variable(iis.run(optim.dist_mv_avg[1][0]))
# vari_hist_n = tf.Variable(iis.run(optim.vari_hist[1][0]))
# iis.run(tf.variables_initializer([vari_hist_n, diff_hist_n]))
#
# ma = tf.reduce_max(vari_hist_n, axis=1, keep_dims=True)
# mi = tf.reduce_min(vari_hist_n, axis=1, keep_dims=True)
# diff = ma - mi
# n = tf.multiply(diff_hist_o,  optim.momentum_alpha) + tf.multiply(diff, optim.momentum_alpha_inv)
#
# print(n.eval())
# print(diff_hist_n.eval())
