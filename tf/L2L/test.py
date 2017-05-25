import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
import util
from preprocess import Preprocess

l2l = tf.Graph()
with l2l.as_default():

    tf.set_random_seed(0)
    epochs = 100
    num_optim_steps_per_epoch = 1
    unroll_len = 1
    num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
    model_number = '1000'
    load_path = 'trained_models/model_' + model_number
    second_derivatives = True
    meta_learning_rate = 0.01
    debug = True

    optim = 'MLP'
    meta = True

    problem = problems.Mnist(args={'gog': second_derivatives, 'meta': meta, 'mode': 'test'})
    preprocess = [Preprocess.log_sign, {'k': 5}]

    if meta:
        if optim == 'L2L':
            optimizer = meta_optimizer.l2l(args={'state_size': 20, 'num_layers': 2, \
                                             'unroll_len': unroll_len, 'learning_rate': 0.001,\
                                             'meta_learning_rate': meta_learning_rate})
            loss_final, update, reset = optimizer.meta_loss()

        elif optim == 'MLP':
            optimizer = meta_optimizer.mlp(args={'problem': problem, 'second_derivatives': second_derivatives,
                                                 'num_layers': 2, 'learning_rate': 0.0001, 'meta_learning_rate': 0.01,
                                                 'momentum': False, 'layer_width': 10, 'preprocess': preprocess})
            loss_final, update, reset = optimizer.meta_loss()
    else:
        optimizer = tf.train.AdamOptimizer(meta_learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(meta_learning_rate)
        slot_names = optimizer.get_slot_names()
        optimizer_reset = tf.variables_initializer(slot_names)
        problem_reset = tf.variables_initializer(problem.variables + problem.constants)
        loss_final = problem.loss(problem.variables)
        update = optimizer.minimize(loss_final)
        reset = [problem_reset, optimizer_reset]

    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(trainable_variables)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l2l.finalize()
        total_loss = 0
        total_time = 0
        time = 0
        flat_grads_list, pre_pro_grads_list, deltas_list = list(), list(), list()
        if meta:
            print 'Resotring Optimizer'
            saver.restore(sess, load_path)

        print 'Starting Evaluation'
        for epoch in range(epochs):
            time, loss = util.run_epoch(sess, loss_final, [update], None, num_unrolls_per_epoch)
            if meta and debug and epoch == 0:
                flat_grads, pre_pro_grads, deltas = sess.run(optimizer.debug_info)
                flatten = lambda mat_array: [element for mat in mat_array for element in mat]
                flat_grads_list.extend(flatten(flat_grads))
                pre_pro_grads_list.extend(flatten(pre_pro_grads))
                deltas_list.extend(flatten(deltas))
            total_time += time
            total_loss += loss
            print 'Epoch: ', epoch
            print 'loss: ', np.log10(loss)
        total_loss = np.log10(total_loss / epochs)
        print 'Final Loss: ', total_loss
        if meta and debug:
            pre_pro_grads_array = np.array(pre_pro_grads_list)
            flat_grads_array = np.array(flat_grads_list)
            deltas_array = np.array(deltas_list)
            print pre_pro_grads_array.shape
            print flat_grads_array.shape
            print deltas_array.shape
            final_debug_array = np.hstack((np.hstack((pre_pro_grads_array, np.array(flat_grads_array))), np.array(deltas_array)))
            np.savetxt('debug_' + model_number + '.txt', final_debug_array, fmt='%5f')


