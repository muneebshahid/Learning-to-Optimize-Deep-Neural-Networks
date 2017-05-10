import tensorflow as tf
import numpy as np
import problems
import meta_optimizer

#
l2l = tf.Graph()
with l2l.as_default():
    tf.set_random_seed(20)
    epochs = 50000
    num_optim_steps_per_epoch = 100
    unroll_len = 20
    num_unrolls_per_epoch = 100 // 20



    # problem = problems.Quadratic(args={'batch_size': batch_size, 'dims': dim, 'stddev': .01, 'dtype': tf.float32})
    # problem = problems.TwoVars(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.ElementwiseSquare(args={'dims': dim, 'dtype':tf.float32})
    # problem = problems.FitX(args={'dims': 20, 'dtype': tf.float32})
    problem = problems.Mnist(args={'preprocess': True, 'p': 5})
    optimizer = meta_optimizer.l2l(problem, args={'state_size': 20, 'num_layers': 2, \
                                                  'unroll_len': unroll_len})
    loss_final, step, update, reset = optimizer.step()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss = 0
        eval = 10
        for epoch in range(epochs):
            print sess.run([tf.reduce_mean(variable) for variable in optimizer.problem.variables])
            sess.run(reset)
            for unroll in range(num_unrolls_per_epoch):
                _, upd, loss_curr, variables = sess.run([step, update, loss_final, optimizer.problem.variables])
                loss += loss_curr
            if epoch % eval == 0:
                log10loss = np.log10(loss / (eval * num_unrolls_per_epoch))
                print 'Epoch/Total Epocs: ', epoch, '/', epochs
                print 'Mean Log Loss: ', log10loss
                # print variables
                print '-----\n'
                with open('loss_file_upd', 'a') as log_file:
                    log_file.write("{:.5f} \n".format(log10loss))
                loss = 0
