import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
import util
from preprocess import Preprocess

#
l2l = tf.Graph()
with l2l.as_default():
    tf.set_random_seed(20)
    second_derivatives = False

    save_path = 'trained_models/model_'
    load_path = 'trained_models/model_'
    restore_network = False
    optim = 'MLP'
    # problem = problems.Quadratic(args={'batch_size': batch_size, 'dims': dim, 'stddev': .01, 'dtype': tf.float32})
    # problem = problems.TwoVars(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.ElementwiseSquare(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.FitX(args={'dims': 20, 'dtype': tf.float32})

    preprocess = [Preprocess.log_sign, {'k': 5}]

    problem = problems.Mnist(args={'gog': second_derivatives})
    eval_loss = problem.loss(problem.variables, 'validation')
    test_loss = problem.loss(problem.variables, 'test')
    if optim == 'L2L':
        print 'Using L2L'
        epochs = 10000
        num_optim_steps_per_epoch = 100
        unroll_len = 20
        epoch_interval = 10
        eval_epochs = 20
        eval_interval = 1000
        num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
        optimizer = meta_optimizer.l2l(args={'problem': problem, 'second_derivatives': second_derivatives,
                                             'state_size': 20, 'num_layers': 2, 'unroll_len': unroll_len,
                                             'learning_rate': 0.001,\
                                             'meta_learning_rate': 0.01,
                                             'preprocess': preprocess})
    else:
        # multiply by 100 to get the same number of epochs
        print 'Using MLP'
        epochs = 10000 * 100

        # used for running epoch
        num_unrolls_per_epoch = 1

        epoch_interval = 1000
        validation_epochs = 500
        test_epochs = 500
        eval_interval = 10000
        optimizer = meta_optimizer.mlp(args={'problem': problem, 'second_derivatives': second_derivatives,
                                             'num_layers': 2, 'learning_rate': 0.0001, 'meta_learning_rate': 0.01,
                                             'momentum': False, 'layer_width': 10, 'preprocess': preprocess})

    loss_final, update, reset, step = optimizer.meta_minimize()
    mean_problem_variables = [tf.reduce_mean(variable) for variable in optimizer.problem.variables]
    mean_optim_variables = [tf.reduce_mean(optimizer.w_1), tf.reduce_mean(optimizer.w_out), tf.reduce_mean(optimizer.b_1) , optimizer.b_out[0][0]]
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(trainable_variables, max_to_keep=100)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l2l.finalize()
        total_loss = 0
        total_time = 0
        time = 0

        best_evaluation = float("inf")
        if restore_network:
            print 'Resotring Optimizer'
            saver.restore(sess, load_path)
            record = np.load('best_eval.npy')
            print 'Best Eval loaded: ', record[0], 'Epoch: ', record[1]

        mean_mats_values_list = list()

        print 'Starting Training...'
        for epoch in range(epochs):
            mean_mats_values = sess.run(mean_problem_variables)
            mean_mats_values_list.append(mean_mats_values)
            time, loss = util.run_epoch(sess, loss_final, [step, update], None, num_unrolls_per_epoch)
            total_loss += loss
            total_time += time
            if (epoch + 1) % epoch_interval == 0:
                print mean_mats_values
                print sess.run(mean_optim_variables)
                log10loss = np.log10(total_loss / epoch_interval)
                util.print_update(epoch, epochs, log10loss, epoch_interval, total_time)
                util.write_update(log10loss, total_time, mean_mats_values_list)
                total_loss = 0
                total_time = 0
                mean_mats_values_list = list()

            if (epoch + 1) % eval_interval == 0:
                print 'VALIDATION'
                loss_eval_total = 0
                for eval_epoch in range(validation_epochs):
                    time_eval, loss_eval = util.run_epoch(sess, eval_loss, None, None, num_unrolls_per_epoch)
                    loss_eval_total += loss_eval
                loss_eval_total = np.log10(loss_eval_total / validation_epochs)
                print 'VALIDATION LOSS: ', loss_eval_total

                print 'TEST'
                loss_test_total = 0
                for eval_epoch in range(test_epochs):
                    time_test, loss_test = util.run_epoch(sess, test_loss, None, None, num_unrolls_per_epoch)
                    loss_test_total += loss_test
                loss_test_total = np.log10(loss_test_total / test_epochs)
                print 'TEST LOSS: ', loss_eval_total

                if loss_eval_total < best_evaluation:
                    print 'Better Loss Found'
                    saver.save(sess, save_path + str(epoch + 1))
                    record = [loss_eval_total, epoch + 1]
                    np.save('best_eval', record)
                    print 'Network Saved'
                    best_evaluation = loss_eval_total
