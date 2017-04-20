import tensorflow as tf
import problems
import meta_optimizer

epochs = 100

params, func = problems.var_square()
optimizer = meta_optimizer.l2l(params, func, args={'state_size': 20, 'num_layers': 2, 'unroll_len': 20})
loss_final, step = optimizer.step()

with tf.Session() as sess:
    loss = 0
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print 'x, loss: ', sess.run([loss_final, step, optimizer.params])

