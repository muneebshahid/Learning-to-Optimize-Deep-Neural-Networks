import tensorflow as tf
import numpy as np
import problems
import meta_optimizer

epochs = 10000
batch_size = 128
dim = 10
problem = problems.Quadratic(args={'batch_size': batch_size, 'dims': dim, 'stddev': .01, 'dtype': tf.float32})
# problem = problems.ElementwiseSquare(args={'batch_size': batch_size, 'dims': dim})
optimizer = meta_optimizer.l2l(problem, args={'state_size': 20, 'num_layers': 2, 'unroll_len': 20})
loss_final, step = optimizer.step()

with tf.Session() as sess:
    loss = 0
    sess.run(tf.global_variables_initializer())
    loss = 0
    eval = 100
    for epoch in range(epochs):
         _, loss_curr, vars = sess.run([step, loss_final, optimizer.problem.vars])
         loss += loss_curr
         if epoch % eval == 0:
             print 'Epoch/Total Epocs: ', epoch, '/', epochs
             print 'Mean Log Loss: ', np.log10(loss / eval)
             print vars
             print '-----\n'
             loss = 0
