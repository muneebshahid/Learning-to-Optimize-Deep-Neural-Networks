import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
import util
from preprocess import Preprocess

l2l = tf.Graph()
with l2l.as_default():

    tf.set_random_seed(20)
    epochs = 50000
    epoch_interval = 1000
    num_optim_steps_per_epoch = 1
    unroll_len = 1
    num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
    model_number = '120000'
    load_path = 'trained_models/model_' + model_number
    second_derivatives = False
    meta_learning_rate = 0.01
    debug = True

    optim = 'MLP'
    meta = True

    problem = problems.Mnist(args={'gog': second_derivatives, 'meta': meta, 'mode': 'train'})
    eval_loss = problem.loss(problem.variables, 'validation')
    test_loss = problem.loss(problem.variables, 'test')

    preprocess = [Preprocess.log_sign, {'k': 10}]
    mean_optim_variables = None
    if meta:
        if optim == 'L2L':
            optimizer = meta_optimizer.l2l(args={'state_size': 20, 'num_layers': 2, \
                                             'unroll_len': unroll_len, 'learning_rate': 0.001,\
                                             'meta_learning_rate': meta_learning_rate})
            loss_final, update, reset = optimizer.meta_loss()

        elif optim == 'MLP':
            optimizer = meta_optimizer.mlp(args={'problem': problem, 'second_derivatives': second_derivatives,
                                                 'num_layers': 2, 'learning_rate': 0.0001, 'meta_learning_rate': 0.01,
                                                 'momentum': False, 'layer_width': 1, 'preprocess': preprocess})
            loss_final, update, reset, step = optimizer.meta_minimize()
            final_step = [update, step]
            mean_optim_variables = [tf.reduce_mean(optimizer.w_1), tf.reduce_mean(optimizer.w_out),
                                    tf.reduce_mean(optimizer.b_1), optimizer.b_out[0][0]]
    else:
        optimizer = tf.train.AdamOptimizer(meta_learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(meta_learning_rate)
        slot_names = optimizer.get_slot_names()
        optimizer_reset = tf.variables_initializer(slot_names)
        problem_reset = tf.variables_initializer(problem.variables + problem.constants)
        loss_final = problem.loss(problem.variables)
        final_step = [optimizer.minimize(loss_final)]
        reset = [problem_reset, optimizer_reset]


    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(trainable_variables)
    mean_problem_variables = [tf.reduce_mean(variable) for variable in problem.variables]


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l2l.finalize()
        print '---- Starting Evaluation ----'
        print 'Init Problem Vars: ', sess.run(mean_problem_variables)
        if mean_optim_variables is not None:
            print 'Init Optim Vars: ', sess.run(mean_optim_variables)
        total_loss_final = 0
        total_loss = 0
        total_time = 0
        time = 0
        flat_grads_list, pre_pro_grads_list, deltas_list = list(), list(), list()
        if meta:
            print 'Resotring Optimizer'
            # saver.restore(sess, load_path)

        print '---------------------------------\n'
        for epoch in range(epochs):
            time, loss = util.run_epoch(sess, loss_final, final_step, None, num_unrolls_per_epoch)
            total_time += time
            total_loss += loss
            total_loss_final += loss
            if (epoch + 1) % epoch_interval == 0:
                print 'Problem Vars: ', sess.run(mean_problem_variables)
                if mean_optim_variables is not None:
                    print 'Optim Vars: ', sess.run(mean_optim_variables)
                log10loss = np.log10(total_loss / epoch_interval)
                util.print_update(epoch, epochs, log10loss, epoch_interval, total_time)
                total_loss = 0
                total_time = 0
                mean_mats_values_list = list()
            if meta and debug and epoch == 0:
                flat_grads, pre_pro_grads, deltas = sess.run(optimizer.debug_info)
                flatten = lambda mat_array: [element for mat in mat_array for element in mat]
                flat_grads_list.extend(flatten(flat_grads))
                pre_pro_grads_list.extend(flatten(pre_pro_grads))
                deltas_list.extend(flatten(deltas))
        total_loss_final = np.log10(total_loss_final / epochs)
        print 'Final Loss: ', total_loss_final
        if meta and debug:
            pre_pro_grads_array = np.array(pre_pro_grads_list)
            flat_grads_array = np.array(flat_grads_list)
            deltas_array = np.array(deltas_list)
            print pre_pro_grads_array.shape
            print flat_grads_array.shape
            print deltas_array.shape
            final_debug_array = np.hstack((np.hstack((pre_pro_grads_array, np.array(flat_grads_array))), np.array(deltas_array)))
            np.savetxt('debug_' + model_number + '.txt', final_debug_array, fmt='%5f')


