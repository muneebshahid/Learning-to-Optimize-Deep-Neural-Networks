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
    restore_network = False
    io_path = None
    save_network = True

    config_args = config.mlp_norm_history()
    reset_epoch_ext = 5000 / config_args['unroll_len']
    reset_limits_lower = []
    reset_limits_upper = []
    #########################
    epochs = int(1000000 / config_args['unroll_len'])
    epoch_interval = int(500 / config_args['unroll_len'])
    eval_interval = int(500 / config_args['unroll_len'])
    validation_epochs = int(100)
    #########################
    cifar_path = '../../../cifar/'
    # problem = problems.cifar10(
    #     {'prefix': 'train', 'minval': 0, 'maxval': 100, 'conv': True, 'full': True, 'path': cifar_path})
    # problem_eval_1 = problems.cifar10(
    #     {'prefix': 'eval_1', 'minval': 0, 'maxval': 100, 'conv': True, 'full': False, 'path': cifar_path})

    problem = problems.Mnist({'prefix': 'train', 'minval': 0, 'maxval': 100, 'conv': False, 'full': True})
    problem_eval_1 = problems.Mnist({'prefix': 'eval_1', 'minval': 0, 'maxval': 100, 'conv': False, 'full': False})
    # problem_eval_2 = problems.Mnist({'prefix': 'eval_2', 'minval': 0, 'maxval': 100})
    problems_eval = [problem_eval_1]#, problem_eval_2]
    if restore_network:
        io_path = util.get_model_path(flag_optimizer='Mlp', model_id='xx') if restore_network else None
    optim = meta_optimizers.AUGOptims([problem], problems_eval, path=io_path, args=config.aug_optim())
    optim.build()

    optim_grad = tf.gradients(optim.ops_loss, optim.optimizer_variables)
    optim_grad_norm = [tf.norm(grad) for grad in optim_grad]
    optim_norm = [tf.norm(variable) for variable in optim.optimizer_variables]
    # norm_grads = [tf.norm(gradients) for gradients in optim.problems.get_gradients()]
    problem_norm_train = 0
    for variable in problem.variables:
        problem_norm_train += tf.norm(variable)

    problem_eval_norms = []
    for problem in problems_eval:
        norm = 0
        for variable in problem.variables:
            norm += tf.norm(variable)
        problem_eval_norms.append(norm)

    reset_limit_init = [50, 200]
    reset_limit_later = [200, 10000]

    with tf.Session() as sess:
        reset_upper_limit = np.random.uniform(reset_limit_init[0], reset_limit_init[1])
        reset_counter = 1

        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess)
        optim.set_session(sess)
        optim.run_init()
        l2l.finalize()
        print('---- Starting Training ----')
        if restore_network:
            optim.load(io_path)
        print('Optim Norm: ', sess.run(optim_norm))
        print('Init Optim Grad Norm: ', sess.run(optim_grad_norm))
        print('Next Update after ', reset_upper_limit)

        total_loss_optim = 0
        total_loss_prob = 0
        total_time = 0
        time = 0
        best_evaluation = float("inf")
        mean_mats_values_list = list()


        print('---------------------------------\n')
        for epoch in range(epochs):
            time, loss_optim, loss_prob = optim.run({'train': True})
            loss_optim = loss_optim[0]
            loss_prob = loss_prob[0]

            problem_norm_run = sess.run(problem_norm_train)
            total_loss_optim += loss_optim
            total_loss_prob += loss_prob
            total_time += time

            avg_optim_loss = total_loss_optim / reset_counter
            avg_prob_loss = total_loss_prob / reset_counter
            avg_time = total_time / reset_counter

            if (epoch + 1) % epoch_interval == 0:
                util.print_update(epoch, epochs, avg_optim_loss, np.log10(avg_prob_loss),
                                  avg_time, sess.run(optim_norm), sess.run(optim_grad_norm))
                print('PROBLEM NORM: ', problem_norm_run)

            if loss_prob < 1e-15 or reset_counter >= reset_upper_limit:
                util.print_update(epoch, epochs, avg_optim_loss, np.log10(avg_prob_loss),
                                  avg_time, sess.run(optim_norm), sess.run(optim_grad_norm))
                print('PROBLEM NORM: ', problem_norm_run)
                total_loss_optim = 0
                total_loss_prob = 0
                total_time = 0
                reset_counter = 0
                if epoch < reset_epoch_ext:
                    reset_upper_limit = np.random.uniform(reset_limit_init[0], reset_limit_init[1])
                else:
                    reset_upper_limit = np.random.uniform(reset_limit_later[0], reset_limit_later[1])
                optim.run_reset(0)
            reset_counter += 1

            if (epoch + 1) % eval_interval == 0:
                optim.run_reset(val=True, index=0)
                potential_nan = False
                print('--- VALIDATION ---')
                total_eval_loss = 0
                total_eval_time = 0
                for eval_epoch in range(validation_epochs):
                    time_eval, _, loss_eval = optim.run({'train': False})
                    total_eval_loss += np.array(loss_eval)
                    total_eval_time += time_eval
                    eval_norms = np.array(sess.run(problem_eval_norms))
                    check = np.sum(np.where(eval_norms > 1e4))
                    if check > 0:
                        potential_nan = True
                        print('Potential Nan')
                        break
                if not potential_nan:
                    avg_eval_loss = np.log10(total_eval_loss / validation_epochs)
                    avg_eval_time = total_eval_time / validation_epochs
                    util.write_update(avg_eval_loss, avg_eval_time)
                    print('VALIDATION LOSS: ', avg_eval_loss)
                    print('SAVING NETWORK')
                    save_path = util.get_model_path(flag_optimizer='Mlp', model_id=str(epoch + 1))
                    optim.save(save_path)

        if save_network:
            save_path = util.get_model_path(flag_optimizer='Mlp', model_id=str(epochs) + '_FINAL')
            # preprocess_args=preprocess,
            # learning_rate=learning_rate, layer_width=layer_width,
            # momentum=momentum, second_derivative=second_derivatives)
            print(save_path)
            optim.save(save_path)
            print('Final Network Saved')
        print('Mlp' + ' optimized.')
