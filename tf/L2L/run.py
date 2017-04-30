import tensorflow as tf
import numpy as np
import problems
import meta_optimizer


l2l = tf.Graph()
with l2l.as_default():
    epochs = 1000
    batch_size = 1
    dim = 1
    tf.set_random_seed(100)
    # problem = problems.Quadratic(args={'batch_size': batch_size, 'dims': dim, 'stddev': .01, 'dtype': tf.float32})
    problem = problems.ElementwiseSquare(args={'batch_size': batch_size, 'dims': dim, 'dtype':tf.float32})
    # problem = problems.TwoVars(args={'batch_size': batch_size, 'dims': dim, 'dtype':tf.float32})
    optimizer = meta_optimizer.l2l(problem, args={'state_size': 20, 'num_layers': 2, 'unroll_len': 20})
    loss_final, step, update = optimizer.step()


    with tf.Session() as sess:
        loss = 0
        sess.run(tf.global_variables_initializer())
        loss = 0
        eval = 100
        print sess.run(optimizer.problem.variables)
        for epoch in range(epochs):         
            _, upd, loss_curr, variables = sess.run([step, update, loss_final, optimizer.problem.variables])
            # _, loss_curr, variables = sess.run([step, loss_final, optimizer.problem.variables])
            loss += loss_curr
            # variables = vars[0][0]
            if epoch % eval == 0:
                print 'Epoch/Total Epocs: ', epoch, '/', epochs
                print 'Mean Log Loss: ', np.log10(loss / eval)
                print variables
                print '-----\n'
                with open('loss_file_upd', 'a') as log_file:
                    log_file.write("{:.5f} ".format(np.log10(loss / eval)))
                    # log_file.write(' {}\n'.format(variables))
                loss = 0
