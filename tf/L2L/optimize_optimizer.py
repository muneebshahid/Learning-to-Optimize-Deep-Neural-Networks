from __future__ import print_function
import tensorflow as tf
import numpy as np
import problems
import meta_optimizers
import util
import config
from preprocess import Preprocess

l2l = tf.Graph()
with l2l.as_default():
    tf.set_random_seed(0)
    preprocess = [Preprocess.log_sign, {'k': 10}]

    second_derivatives = False
    restore_network = False
    io_path = ''
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

    flag_optimizer = 'Mlp'

    model_id = '10'


    # problem = problems.Quadratic(args={'batch_size': batch_size, 'dims': dim, 'stddev': .01, 'dtype': tf.float32})
    # problem = problems.TwoVars(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.ElementwiseSquare(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.FitX(args={'dims': 20, 'dtype': tf.float32})

    problem = problems.ElementwiseSquare(args={'minval':-100, 'maxval':100, 'dims':2, 'gog': False, 'path': 'cifar', 'conv': True})
    eval_loss = problem.loss(problem.variables, 'validation')
    test_loss = problem.loss(problem.variables, 'test')
    problem_batches, reset_limits = problems.create_batches_all()
    config_args = config.rnn_norm_history()
    save_network_interval = 50000 / config_args['unroll_len']
    reset_epoch_ext = 20000
    if flag_optimizer == 'L2L':
        print('Using MLP')
        #########################
        epochs = 15000
        epoch_interval = 1000
        eval_interval = 5000
        validation_epochs = 500
        #########################

        num_unrolls_per_epoch = 1
        io_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id) if restore_network else None
                                      # preprocess_args=preprocess,
                                      # learning_rate=learning_rate, layer_width=layer_width,
                                      # momentum=momentum) if restore_network else None
        optim = meta_optimizers.GRUNormHistory(problem_batches, path=io_path, args=config.mlp_norm_history())
        optim.build()
        # print('Using L2L')
        # #########################
        # epochs = 10000
        # num_optim_steps_per_epoch = 100
        # unroll_len = 20
        # epoch_interval = 1
        # eval_interval = 10000
        # validation_epochs = 5
        # test_epochs = 5
        # #########################
        # num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
        # io_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id) if restore_network else None
        #                               # preprocess_args=preprocess,
        #                               # learning_rate=learning_rate, layer_width=layer_width,
        #                               # momentum=momentum, second_derivative=second_derivatives)
        #
        # optim = meta_optimizers.l2l(problem, path=None, args={'optim_per_epoch': num_optim_steps_per_epoch,
        #                                                          'state_size': 20, 'num_layers': 2,
        #                                                          'unroll_len': unroll_len,
        #                                                          'learning_rate': 0.001,
        #                                                          'meta_learning_rate': 0.01,
        #                                                           'preprocess': preprocess})
        # optim.build()
    else:
        print('Using MLP')
        #########################
        epochs = 1000000 / config_args['unroll_len']
        epoch_interval = 500 / config_args['unroll_len']
        eval_interval = save_network_interval
        validation_epochs = 10000 / config_args['unroll_len']
        #########################

        num_unrolls_per_epoch = 1
        io_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id) if restore_network else None
                                      # preprocess_args=preprocess,
                                      # learning_rate=learning_rate, layer_width=layer_width,
                                      # momentum=momentum) if restore_network else None
        optim = meta_optimizers.GRUNormHistory(problem_batches, path=io_path, args=config.rnn_norm_history())
        optim.build()

    optim_grad = tf.gradients(optim.ops_loss, optim.optimizer_variables)
    optim_grad_norm = [tf.norm(grad) for grad in optim_grad]
    optim_norm = [tf.norm(variable) for variable in optim.optimizer_variables]
    # norm_grads = [tf.norm(gradients) for gradients in optim.problems.get_gradients()]
    problem_norms = []
    for problem in optim.problems:
        norm = 0
        for variable in problem.variables:
            norm += tf.norm(variable)
        problem_norms.append(norm)
    reset_upper_limit = np.array([np.random.uniform(reset_limit[0][0], reset_limit[0][1]) for reset_limit in reset_limits])
    reset_counter = np.zeros(len(optim.problems))
    optim_loss_record = np.ones(len(optim.problems)) * np.inf
    prob_loss_record =  np.ones(len(optim.problems)) * np.inf
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess)
        optim.set_session(sess)
        optim.run_init()
        l2l.finalize()
        print('---- Starting Training ----')
        if restore_network:
            optim.load(io_path)
        # print('Init Prob Grad Norm: ', sess.run(norm_grads))
        print('Optim Norm: ', sess.run(optim_norm))
        print('Init Optim Grad Norm: ', sess.run(optim_grad_norm))
        # print 'Init Optim Vars: ', sess.run(mean_optim_variables)
        total_loss_optim = 0
        total_loss_prob = 0
        total_time = 0
        time = 0

        best_evaluation = float("inf")
        mean_mats_values_list = list()

        print('---------------------------------\n')
        for epoch in range(epochs):
            time, loss_optim, loss_prob = optim.run({'train': True})
            problem_norms_run = sess.run(problem_norms)
            total_loss_optim += loss_optim
            total_loss_prob += loss_prob
            total_time += time
            for i, (ops_reset, curr_loss_prob, norm) in enumerate(zip(optim.ops_reset_problem, loss_prob, problem_norms_run)):
                curr_loss_flatten = np.squeeze(curr_loss_prob)
                if curr_loss_flatten < 1e-15 or reset_counter[i] >= reset_upper_limit[i] or norm > 1e4:
                    optim.run_reset(index=i)
                    if epoch < reset_epoch_ext:
                        reset_index = 0
                    else:
                        reset_index = 1
                    optim_loss_record[i] = total_loss_optim[i] / reset_counter[i]
                    prob_loss_record[i] = total_loss_prob[i] / reset_counter[i]
                    reset_upper_limit[i] = np.random.uniform(reset_limits[i][reset_index][0], reset_limits[i][reset_index][1])
                    reset_counter[i] = 0
                else:
                    reset_counter[i] += 1

            if (epoch + 1) % epoch_interval == 0:
                indices = np.where(optim_loss_record == np.inf)
                optim_loss_record[indices] = total_loss_optim[indices] / epoch_interval
                prob_loss_record[indices] = total_loss_prob[indices] / epoch_interval
                # print 'Optim Vars: ', sess.run(mean_optim_variables)
                util.print_update(epoch, epochs, optim_loss_record, np.log10(prob_loss_record), epoch_interval, total_time, sess.run(optim_norm), sess.run(optim_grad_norm))
                print('PROBLEM NORM: ', problem_norms_run)
                total_loss_optim = 0
                total_loss_prob = 0
                total_time = 0
                mean_mats_values_list = list()

            if (epoch + 1) % eval_interval == 0:
                print('--- VALIDATION ---')
                total_eval_loss = 0
                total_eval_time = 0
                for eval_epoch in range(validation_epochs):
                    time_eval, _, loss_eval = optim.run({'train': False})
                    total_eval_loss += loss_eval
                    total_eval_time += time_eval
                avg_eval_loss = np.log10(total_eval_loss / validation_epochs)
                avg_eval_time = total_eval_time / validation_epochs
                util.write_update(avg_eval_loss, avg_eval_time)
                print('VALIDATION LOSS: ', avg_eval_loss)
            if (epoch + 1) % save_network_interval == 0:
                print('SAVING NETWORK')
                save_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=str(epoch + 1))
                optim.save(save_path)
                # print('TEST')
                # loss_test_total = 0
                # for eval_epoch in range(test_epochs):
                #     time_test, loss_test = util.run_epoch(sess, test_loss, None, None, num_unrolls_per_epoch)
                #     loss_test_total += loss_test
                # loss_test_total = np.log10(loss_test_total / test_epochs)
                # print('TEST LOSS: ', loss_eval_total)

                # if save_network:# and avg_eval_loss < best_evaluation:
                #     print('Better Loss Found')
                #     save_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=str(epoch + 1))
                #                                     # preprocess_args=preprocess,
                #                                     # learning_rate=learning_rate, layer_width=layer_width,
                #                                     # momentum=momentum, second_derivative=second_derivatives)
                #     print(save_path)
                #     optim.save(save_path)
                #     print('Network Saved')
                #     best_evaluation = avg_eval_loss
                # print('---------------------------------------')
        if save_network:
            save_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=str(epochs) + '_FINAL')
                                            # preprocess_args=preprocess,
                                            # learning_rate=learning_rate, layer_width=layer_width,
                                            # momentum=momentum, second_derivative=second_derivatives)
            print(save_path)
            optim.save(save_path)
            print('Final Network Saved')
        print(flag_optimizer + ' optimized.')
