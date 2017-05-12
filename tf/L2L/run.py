import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
import util

#
l2l = tf.Graph()
with l2l.as_default():
    tf.set_random_seed(0)
    epochs = 50000
    num_optim_steps_per_epoch = 100
    unroll_len = 20
    num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
    second_derivatives = False
    epoch_interval = 10
    eval_epochs = 20
    eval_interval = 100
    save_path = 'trained_models/rnn_model'
    restore_network = False

    # problem = problems.Quadratic(args={'batch_size': batch_size, 'dims': dim, 'stddev': .01, 'dtype': tf.float32})
    # problem = problems.TwoVars(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.ElementwiseSquare(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.FitX(args={'dims': 20, 'dtype': tf.float32})

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

        best_evaluation = float("inf")
        if restore_network:
            print 'Resotring Optimizer'
            saver.restore(sess, save_path)
            best_evaluation = np.load('best_eval.npy')
            print 'Best Eval loaded', best_evaluation

        mean_mats_values_list = list()

        print 'Starting Training...'
        for epoch in range(epochs):
            mean_mats_values = sess.run(mean_mats)
            # print mean_mats_values
            mean_mats_values_list.append(mean_mats_values)
            time, loss = util.run_epoch(sess, loss_final, [step, update], reset, num_unrolls_per_epoch)
            total_loss += loss
            total_time += time
            if (epoch + 1) % epoch_interval == 0:
                log10loss = np.log10(total_loss / epoch_interval)
                util.print_update(epoch, epochs, log10loss, epoch_interval, total_time)
                util.write_update(log10loss, total_time, mean_mats_values_list)
                total_loss = 0
                total_time = 0
                mean_mats_values_list = list()

            if (epoch + 1) % eval_interval == 0:
                print 'EVALUATION'
                loss_eval_total = 0
                for eval_epoch in range(eval_epochs):
                    time_eval, loss_eval = util.run_epoch(sess, loss_final, [update], reset, num_unrolls_per_epoch)
                    loss_eval_total += loss_eval
                loss_eval_total = np.log10(loss_eval_total)
                print 'LOSS: ', loss_eval_total
                if loss_eval_total < best_evaluation:
                    print 'Better Loss Found'
                    saver.save(sess, save_path)
                    np.save('best_eval', loss_eval_total)
                    print 'RNN Saved'
                    best_evaluation = loss_eval_total
