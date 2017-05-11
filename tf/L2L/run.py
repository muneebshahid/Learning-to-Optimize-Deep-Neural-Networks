import tensorflow as tf
import numpy as np
import problems
import meta_optimizer
from timeit import default_timer as timer

#
l2l = tf.Graph()
with l2l.as_default():
    tf.set_random_seed(20)
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
                                                  'unroll_len': unroll_len, 'learning_rate': 0.001})
    loss_final, step, update, reset = optimizer.step()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0
        total_time = 0
        time = 0
        epoch_interval = 10
        for epoch in range(epochs):
            # print sess.run([tf.reduce_mean(variable) for variable in optimizer.problem.variables])
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
                total_loss = 0
                total_time = 0
