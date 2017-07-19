from __future__ import print_function
import tensorflow as tf
import numpy as np
import problems
import meta_optimizers
import util
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
    problem_batches = problems.create_batches_all()
    if flag_optimizer == 'L2L':
        print('Using L2L')
        #########################
        epochs = 10000
        num_optim_steps_per_epoch = 100
        unroll_len = 20
        epoch_interval = 1
        eval_interval = 10000
        validation_epochs = 5
        test_epochs = 5
        #########################
        num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
        io_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id) if restore_network else None
                                      # preprocess_args=preprocess,
                                      # learning_rate=learning_rate, layer_width=layer_width,
                                      # momentum=momentum, second_derivative=second_derivatives)

        optim = meta_optimizers.l2l(problem, path=None, args={'optim_per_epoch': num_optim_steps_per_epoch,
                                                                 'state_size': 20, 'num_layers': 2,
                                                                 'unroll_len': unroll_len,
                                                                 'learning_rate': 0.001,
                                                                 'meta_learning_rate': 0.01,
                                                                  'preprocess': preprocess})
        optim.build()
    else:
        print('Using MLP')
        #########################
        epochs = 1000
        epoch_interval = 100
        eval_interval = 200
        validation_epochs = 50
        test_epochs = 500
        #########################
        learning_rate = 0.0001
        layer_width = 50
        momentum = False
        #########################

        num_unrolls_per_epoch = 1
        io_path = util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id) if restore_network else None
                                      # preprocess_args=preprocess,
                                      # learning_rate=learning_rate, layer_width=layer_width,
                                      # momentum=momentum) if restore_network else None
        optim = meta_optimizers.MlpHistoryGradNorm(problem_batches, path=io_path, args={'second_derivatives': second_derivatives,
                                                                      'hidden_layers': 1, 'learning_rate': learning_rate,
                                                                      'meta_learning_rate': 0.0001,
                                                                      'momentum': momentum, 'layer_width': layer_width,
                                                                      'preprocess': preprocess, 'limit': 5})
        optim.build()

    optim_grad = tf.gradients(optim.ops_loss, optim.optimizer_variables)
    optim_grad_norm = [tf.norm(grad) for grad in optim_grad]
    optim_norm = [tf.norm(variable) for variable in optim.optimizer_variables]
    # norm_grads = [tf.norm(gradients) for gradients in optim.problems.get_gradients()]

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
        total_loss = 0
        total_time = 0
        time = 0

        best_evaluation = float("inf")
        mean_mats_values_list = list()

        print('---------------------------------\n')
        for epoch in range(epochs):
            time, loss_value = optim.run({'num_steps': 100,
                                          'ops_loss': True,
                                          'ops_reset': True,
                                          'ops_meta_step': True,
                                          'ops_updates': True})
            total_loss += loss_value
            total_time += time
            if (epoch + 1) % epoch_interval == 0:
                # print 'Optim Vars: ', sess.run(mean_optim_variables)
                avg_epoch_loss = total_loss / epoch_interval
                util.print_update(epoch, epochs, avg_epoch_loss, epoch_interval, total_time, sess.run(optim_norm), sess.run(optim_grad_norm))
                total_loss = 0
                total_time = 0
                mean_mats_values_list = list()

            if (epoch + 1) % eval_interval == 0:
                print('--- VALIDATION ---')
                avg_eval_loss = 0
                for eval_epoch in range(validation_epochs):
                    time_eval, loss_eval = optim.run({'num_steps': 100,
                                                      'ops_loss': True,
                                                      'ops_reset': True,
                                                      'ops_updates': True})
                    avg_eval_loss += loss_eval
                avg_eval_loss = avg_eval_loss / validation_epochs
                print('VALIDATION LOSS: ', avg_eval_loss)

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
