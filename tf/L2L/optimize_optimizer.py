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
    preprocess = [Preprocess.log_sign, {'k': 10}]

    second_derivatives = False
    restore_network = False
    save_network = True

    epochs = None
    num_optim_steps_per_epoch = None
    unroll_len = None
    epoch_interval = None
    eval_interval = None
    validation_epochs = None
    test_epochs = None
    learning_rate = None
    layer_width = None
    momentum = None

    flag_optimizer = 'MLP'

    model_id = '10'



    # problem = problems.Quadratic(args={'batch_size': batch_size, 'dims': dim, 'stddev': .01, 'dtype': tf.float32})
    # problem = problems.TwoVars(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.ElementwiseSquare(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.FitX(args={'dims': 20, 'dtype': tf.float32})

    problem = problems.Mnist(args={'gog': second_derivatives})
    eval_loss = problem.loss(problem.variables, 'validation')
    test_loss = problem.loss(problem.variables, 'test')

    if flag_optimizer == 'L2L':
        print('Using L2L')
        #########################
        epochs = 10000
        num_optim_steps_per_epoch = 100
        unroll_len = 20
        epoch_interval = 1000
        eval_interval = 10000
        validation_epochs = 5
        test_epochs = 5
        #########################
        num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
        io_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id,
                                      preprocess_args=preprocess,
                                      learning_rate=learning_rate, layer_width=layer_width,
                                      momentum=momentum, second_derivative=second_derivatives) if restore_network else None
        optimizer = meta_optimizer.l2l(problem, path=None, args={'second_derivatives': second_derivatives,
                                                                 'state_size': 20, 'num_layers': 2,
                                                                 'unroll_len': unroll_len,
                                                                 'learning_rate': 0.001,
                                                                 'meta_learning_rate': 0.01,
                                                                 'preprocess': preprocess})
        loss_final, update, reset, step = optimizer.minimize()
        reset = None
    else:
        print('Using MLP')
        #########################
        epochs = 500
        num_optim_steps_per_epoch = 1
        unroll_len = 1
        epoch_interval = 1000
        eval_interval = 10
        validation_epochs = 50
        test_epochs = 500
        #########################
        learning_rate = 0.0001
        layer_width = 10
        momentum = False
        #########################

        num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
        io_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id)
                                      # preprocess_args=preprocess,
                                      # learning_rate=learning_rate, layer_width=layer_width,
                                      # momentum=momentum) if restore_network else None
        optimizer = meta_optimizer.mlp(problem, path=io_path, args={'second_derivatives': second_derivatives,
                                                                      'num_layers': 1, 'learning_rate': learning_rate,
                                                                      'meta_learning_rate': 0.01,
                                                                      'momentum': momentum, 'layer_width': layer_width,
                                                                      'preprocess': preprocess})
        loss_final, update, reset, step = optimizer.minimize()
        reset = None
        mean_optim_variables = [tf.reduce_mean(optimizer.w_1), tf.reduce_mean(optimizer.w_out),
                                tf.reduce_mean(optimizer.b_1), optimizer.b_out[0][0]]

    mean_problem_variables = [tf.reduce_mean(variable) for variable in optimizer.problem.variables]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l2l.finalize()
        print('---- Starting Training ----')
        if restore_network:
            optimizer.load(sess, io_path)
        print('Init Problem Vars: ', sess.run(mean_problem_variables))
        # print 'Init Optim Vars: ', sess.run(mean_optim_variables)
        total_loss = 0
        total_time = 0
        time = 0

        best_evaluation = float("inf")
        mean_mats_values_list = list()

        print('---------------------------------\n')
        for epoch in range(epochs):
            mean_mats_values = sess.run(mean_problem_variables)
            mean_mats_values_list.append(mean_mats_values)
            time, loss = util.run_epoch(sess, loss_final, [step, update], reset, num_unrolls_per_epoch)
            total_loss += loss
            total_time += time
            if (epoch + 1) % epoch_interval == 0:
                print('Problem Vars: ', mean_mats_values)
                # print 'Optim Vars: ', sess.run(mean_optim_variables)
                log10loss = np.log10(total_loss / epoch_interval)
                util.print_update(epoch, epochs, log10loss, epoch_interval, total_time)
                util.write_update(log10loss, total_time, mean_mats_values_list)
                total_loss = 0
                total_time = 0
                mean_mats_values_list = list()

            if (epoch + 1) % eval_interval == 0:
                print('VALIDATION')
                loss_eval_total = 0
                for eval_epoch in range(validation_epochs):
                    time_eval, loss_eval = util.run_epoch(sess, eval_loss, None, reset, num_unrolls_per_epoch)
                    loss_eval_total += loss_eval
                loss_eval_total = np.log10(loss_eval_total / validation_epochs)
                print('VALIDATION LOSS: ', loss_eval_total)

                print('TEST')
                loss_test_total = 0
                for eval_epoch in range(test_epochs):
                    time_test, loss_test = util.run_epoch(sess, test_loss, None, None, num_unrolls_per_epoch)
                    loss_test_total += loss_test
                loss_test_total = np.log10(loss_test_total / test_epochs)
                print('TEST LOSS: ', loss_eval_total)

                if save_network and loss_eval_total < best_evaluation:
                    print('Better Loss Found')
                    save_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=str(epoch + 1))
                                                    # preprocess_args=preprocess,
                                                    # learning_rate=learning_rate, layer_width=layer_width,
                                                    # momentum=momentum, second_derivative=second_derivatives)
                    print(save_path)
                    optimizer.save(sess, save_path)
                    print('Network Saved')
                    best_evaluation = loss_eval_total
        if save_network:
            save_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=str(epochs) + '_FINAL')
                                            # preprocess_args=preprocess,
                                            # learning_rate=learning_rate, layer_width=layer_width,
                                            # momentum=momentum, second_derivative=second_derivatives)
            print(save_path)
            optimizer.save(sess, save_path)
            print('Final Network Saved')
        print(flag_optimizer + ' optimized.')
