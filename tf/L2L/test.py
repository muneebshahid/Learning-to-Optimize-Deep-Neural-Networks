from __future__ import print_function
import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
import util
from preprocess import Preprocess

l2l = tf.Graph()
with l2l.as_default():
    tf.set_random_seed(20)
    epochs = 100
    epoch_interval = 1
    num_optim_steps_per_epoch = 1
    unroll_len = 1
    num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
    flag_optimizer = 'MLP'
    model_id = '1000000'
    model_id += '_FINAL'
    load_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id)
    second_derivatives = False
    meta_learning_rate = 0.01
    meta = True

    problem = problems.Mnist(args={'gog': second_derivatives, 'meta': meta, 'mode': 'test'})
    eval_loss = problem.loss(problem.variables, 'validation')
    test_loss = problem.loss(problem.variables, 'test')

    preprocess = [Preprocess.log_sign, {'k': 10}]
    final_step = []
    mean_optim_variables = None
    optimizer = None
    loss = None
    if meta:
        if flag_optimizer == 'L2L':
            optimizer = meta_optimizer.l2l(problem, path=load_path, args={'state_size': 20, 'num_layers': 2,
                                                                          'unroll_len': unroll_len,
                                                                          'learning_rate': 0.001,
                                                                          'meta_learning_rate': meta_learning_rate})
            step, updates, loss, meta_step, reset = optimizer.build()
            final_step = [updates]

        elif flag_optimizer == 'MLP':
            optimizer = meta_optimizer.MlpSimple(problem, path=load_path, args={'second_derivatives': second_derivatives,
                                                                          'num_layers': 2, 'learning_rate': 0.0001,
                                                                          'meta_learning_rate': 0.01,
                                                                          'momentum': False, 'layer_width': 1,
                                                                          'preprocess': preprocess})
            step, updates, loss, meta_step, reset = optimizer.build()
            final_step = [updates, meta_step]
            mean_optim_variables = [tf.reduce_mean(optimizer.w_1), tf.reduce_mean(optimizer.w_out),
                                    tf.reduce_mean(optimizer.b_1), optimizer.b_out[0][0]]
    else:
        optimizer = tf.train.AdamOptimizer(meta_learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(meta_learning_rate)
        slot_names = optimizer.get_slot_names()
        optimizer_reset = tf.variables_initializer(slot_names)
        problem_reset = tf.variables_initializer(problem.variables + problem.constants)
        loss = problem.loss(problem.variables)
        final_step = [optimizer.minimize(loss)]
        reset = [problem_reset, optimizer_reset]

    mean_problem_variables = [tf.reduce_mean(variable) for variable in problem.variables]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l2l.finalize()
        print('---- Starting Evaluation ----')
        if meta:
            print('Resotring Optimizer')
            optimizer.load(sess, load_path)
        print('Init Problem Vars: ', sess.run(mean_problem_variables))
        if mean_optim_variables is not None:
            print('Init Optim Vars: ', sess.run(mean_optim_variables))
        total_loss_final = 0
        total_loss = 0
        total_time = 0
        time = 0
        flat_grads_list, pre_pro_grads_list, deltas_list = list(), list(), list()

        print('---------------------------------\n')
        for epoch in range(epochs):
            time, loss_value = util.run_epoch(sess, loss, final_step, None, num_unrolls_per_epoch)
            total_time += time
            total_loss += loss_value
            total_loss_final += loss_value
            if (epoch + 1) % epoch_interval == 0:
                print('Problem Vars: ', sess.run(mean_problem_variables))
                if mean_optim_variables is not None:
                    print('Optim Vars: ', sess.run(mean_optim_variables))
                log10loss = np.log10(total_loss / epoch_interval)
                util.print_update(epoch, epochs, log10loss, epoch_interval, total_time)
                total_loss = 0
                total_time = 0
                mean_mats_values_list = list()
        total_loss_final = np.log10(total_loss_final / epochs)
        print('Final Loss: ', total_loss_final)