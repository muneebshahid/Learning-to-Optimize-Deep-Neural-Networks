from __future__ import print_function
import tensorflow as tf
import numpy as np
import problems
import meta_optimizers
import util
from preprocess import Preprocess
import config


def write_to_file(f_name, list_var):
    with open(f_name, 'a') as log_file:
        for variable in list_var:
            log_file.write(str(variable) + ' ')
        log_file.write('\n')
results_dir = 'tf_summary/'
model_id = '1000000_FINAL'
meta = True

l2l = tf.Graph()
with l2l.as_default():
    preprocess = [Preprocess.log_sign, {'k': 10}]

    #########################
    epochs = 100
    epoch_interval = 1
    eval_interval = 200
    validation_epochs = 50
    test_epochs = 500
    #########################
    learning_rate = 0.0001
    layer_width = 50
    #########################
    num_unrolls_per_epoch = 1
    io_path = util.get_model_path(flag_optimizer='Mlp', model_id=model_id)
    all_summ = []
    writer = None
    problem_batches, _ = problems.create_batches_all(train=True)
    enable_summaries = False
    optim_meta = meta_optimizers.NormHistory(problem_batches, path=None, args=config.norm_history())
    optim_meta.build()
    optim_adam = tf.train.AdamOptimizer(.01)
    adam_min_step = optim_adam.minimize(optim_meta.ops_loss_problem[0], var_list=optim_meta.problems[0].variables)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess)
        optim_meta.set_session(sess)
        optim_meta.restore_problem(0, '/mhome/shahidm/thesis/thesis_code/tf/L2L/mnist_save_vars/mnist_variables')
        optim_meta.run_init()
        l = []
        for i, (problem, problem_variables_history) in enumerate(zip(optim_meta.problems, optim_meta.variable_history)):
            name_prefix = problem.__class__.__name__ + "_" + str(i)
            with tf.name_scope(name_prefix):
                loss = tf.squeeze(problem.loss(problem.variables))
                l.append(loss)
                if enable_summaries:
                    tf.summary.scalar('loss', loss)
                    for j, (variable, variable_history) in enumerate(zip(problem.variables, problem_variables_history)):
                        # tf.summary.histogram('variable_' + str(j), variable)
                        tf.summary.histogram('variable_history_scl_' + str(j), variable_history)
        if enable_summaries:
            all_summ = tf.summary.merge_all()
            writer = tf.summary.FileWriter(results_dir)
            writer.add_graph(sess.graph)


        l2l.finalize()
        print('---- Starting Evaluation ----')
        if meta:
            optim_meta.load(io_path)
            print('Optimizer loaded.')
        total_loss = 0
        total_itr = 20000
        for i in range(total_itr):
            if meta:
                _, curr_loss, summaries = sess.run([optim_meta.ops_updates, l, all_summ])
            else:
                _, curr_loss, summaries = sess.run([adam_min_step, l, all_summ])
            total_loss += np.array(curr_loss)
            if (i + 1) % 50 == 0:
                print(str(i + 1) + '/' + str(total_itr))
                avg_loss = np.log10(total_loss / 50.0)
                write_to_file(results_dir + 'loss', avg_loss)
                print(avg_loss)
                total_loss = 0
                if enable_summaries and ((i + 10) % 10 == 0):
                    writer.add_summary(summaries, i)
