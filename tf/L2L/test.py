import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
import util

l2l = tf.Graph()
with l2l.as_default():
    tf.set_random_seed(0)
    epochs = 100
    num_optim_steps_per_epoch = 100
    unroll_len = 20
    num_unrolls_per_epoch = num_optim_steps_per_epoch
    eval_interval = 1000
    load_path = 'trained_models/rnn_model'
    second_derivatives = False

    problem = problems.Mnist(args={'gog': second_derivatives})
    optimizer = meta_optimizer.l2l(problem, processing_constant=5, second_derivatives=second_derivatives,
                                   args={'state_size': 20, 'num_layers': 2, \
                                         'unroll_len': unroll_len, 'learning_rate': 0.001,\
                                         'meta_learning_rate': 0.01})

    loss_final, step, update, reset = optimizer.step()
    mean_mats = [tf.reduce_mean(variable) for variable in optimizer.problem.variables]
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(trainable_variables)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l2l.finalize()
        total_loss = 0
        total_time = 0
        time = 0

        print 'Resotring Optimizer'
        saver.restore(sess, load_path)

        for epoch in range(epochs):
            time, loss = util.run_epoch(sess, loss_final, [update], reset, num_unrolls_per_epoch)

            total_time += time
            total_loss += loss
        total_loss = np.log10(total_loss / epochs)
        print 'Final Loss: ', total_loss


