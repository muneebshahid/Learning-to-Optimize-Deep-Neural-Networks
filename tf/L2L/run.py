import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
from timeit import default_timer as timer
from tensorflow.contrib.learn.python.learn import monitored_session as ms

#
l2l = tf.Graph()
with l2l.as_default():
    tf.set_random_seed(0)
    epochs = 50000
    num_optim_steps_per_epoch = 100
    unroll_len = 20
    num_unrolls_per_epoch = num_optim_steps_per_epoch // unroll_len
    second_derivatives = False



    # problem = problems.Quadratic(args={'batch_size': batch_size, 'dims': dim, 'stddev': .01, 'dtype': tf.float32})
    # problem = problems.TwoVars(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.ElementwiseSquare(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.FitX(args={'dims': 20, 'dtype': tf.float32})
    problem = problems.Mnist(args={'gog': second_derivatives, 'preprocess': True, 'p': 5})
    optimizer = meta_optimizer.l2l(problem, args={'state_size': 20, 'num_layers': 2, \
                                                  'unroll_len': unroll_len, 'learning_rate': 0.001,\
                                                  'meta_learning_rate': 0.001})
    loss_final, step, update, reset = optimizer.step()
    mean_mats = [tf.reduce_mean(variable) for variable in optimizer.problem.variables]
    with ms.MonitoredSession() as sess:
        l2l.finalize()
        total_loss = 0
        total_time = 0
        time = 0
        epoch_interval = 10
        mean_mats_values_list = list()
        for epoch in range(epochs):
            mean_mats_values = sess.run(mean_mats)
            print mean_mats_values
            mean_mats_values_list.append(mean_mats_values)
            start = timer()
            sess.run(reset)
            for unroll in range(num_unrolls_per_epoch):
                _, upd, loss, variables = sess.run([step, update, loss_final, optimizer.problem.variables])
            total_loss += loss
            total_time += timer() - start
            if (epoch + 1) % epoch_interval == 0:
                log10loss = np.log10(total_loss / epoch_interval)
                print 'Epoch/Total Epocs: ', epoch + 1, '/', epochs
                print 'Mean Log Loss: ', log10loss
                print 'Mean Epoch Time: ', total_time / epoch_interval
                # print variables
                print '-----\n'
                with open('loss_file_upd', 'a') as log_file:
                    log_file.write("{:.5f}".format(log10loss) + " " + "{:.2f}".format(total_time) + "\n")
                with open('mean_var_upd', 'a') as log_file:
                    for mean_vars in mean_mats_values_list:
                        for value in mean_vars:
                            log_file.write(str(value) + ' ')
                        log_file.write('\n')
                total_loss = 0
                total_time = 0
                mean_mats_values_list = list()
