from __future__ import print_function
import tensorflow as tf
import numpy as np
import problems
import meta_optimizers
import util
from preprocess import Preprocess
import config
import time

def write_to_file(f_name, list_var):
    with open(f_name, 'a') as log_file:
        for variable in list_var:
            log_file.write(str(variable) + ' ')
        log_file.write('\n')
results_dir = 'tf_summary/'
model_id = '50000'

load_model = True
meta = False
optimize = False

l2l = tf.Graph()
with l2l.as_default():
    preprocess = [Preprocess.log_sign, {'k': 10}]
    epochs = 50
    total_data_points = 55000
    batch_size = 128
    itr_per_epoch = int(total_data_points / batch_size)
    io_path = util.get_model_path(flag_optimizer='Mlp', model_id=model_id)
    all_summ = []
    writer = None
    # problem_batches, _ = problems.create_batches_all(train=True)
    problem = problems.Mnist({'minval': -100.0, 'maxval': 100.0})
    enable_summaries = False
    optim_meta = meta_optimizers.AUGOptims([problem], path=None, args=config.aug_optim())
    loss = problem.loss(problem.variables)

    # optim_meta = meta_optimizers.MlpNormHistory(problem_batches, path=None, args=config.mlp_norm_history())
    optim_meta.build()
    if optimize:
        meta_step = optim_meta.ops_meta_step
    else:
        meta_step = []
    optim_adam = tf.train.AdamOptimizer(.01)
    adam_min_step = optim_adam.minimize(loss, var_list=problem.variables)
    problem_norms = []
    for problem in optim_meta.problems:
        norm = 0
        for variable in problem.variables:
            norm += tf.norm(variable)
        problem_norms.append(norm)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess)
        optim_meta.set_session(sess)
        optim_meta.restore_problem(0, '/mhome/shahidm/thesis/thesis_code/tf/L2L/mnist_save_vars_mlp/mnist_variables')
        #optim_meta.restore_problem(0, '/home/shahidm/thesis/thesis_code/tf/L2L/mnist_save_vars_conv/mnist_variables')
        optim_meta.run_init()
        l = [loss]
        # for i, problem in enumerate(optim_meta.problems):
        #     name_prefix = problem.__class__.__name__ + "_" + str(i)
        #     with tf.name_scope(name_prefix):
        #         loss = tf.squeeze(problem.loss(problem.variables)[0])
        #         l.append(loss)
        #         if enable_summaries:
        #             tf.summary.scalar('loss', loss)
        #             #for j, (variable, variable_history) in enumerate(zip(problem.variables, problem_variables_history)):
        #             #    # tf.summary.histogram('variable_' + str(j), variable)
        #             #    tf.summary.histogram('variable_history_scl_' + str(j), variable_history)
        # if enable_summaries:
        #     all_summ = tf.summary.merge_all()
        #     writer = tf.summary.FileWriter(results_dir)
        #     writer.add_graph(sess.graph)


        l2l.finalize()
        print('---- Starting Evaluation ----')
        if meta and load_model:
            optim_meta.load(io_path)
            print('Optimizer loaded.')
        for i in range(epochs):
            total_loss = 0
            start = time.time()
            for j in range(itr_per_epoch):
                if meta:
                    _, _, curr_loss, summaries  = sess.run([optim_meta.ops_updates, meta_step, l, all_summ])
                else:
                    _, curr_loss, summaries = sess.run([adam_min_step, l, all_summ])
                total_loss += np.array(curr_loss)
            total_time = time.time() - start
            print(str(i + 0) + '/' + str(epochs))
            print("time: {0:.2f}s".format(total_time))
            avg_loss = np.log10(total_loss / itr_per_epoch)
            write_to_file(results_dir + 'loss', avg_loss)
            print(avg_loss)
            #print(sess.run(optim_meta.min_lr))
            print('PROB NORM: ', sess.run(problem_norms))
            if enable_summaries and ((i + 10) % 10 == 0):
                writer.add_summary(summaries, i)
