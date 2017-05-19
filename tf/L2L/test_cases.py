import tensorflow as tf
import numpy as np

# def ele():
#
#     with tf.variable_scope('variables'):
#         x = tf.get_variable('x', dtype=tf.float32, #initializer=tf.constant([[1.0, -1.0], [0.5, 3.5]]), trainable=False)
#                                initializer=tf.random_uniform_initializer(minval=-5, maxval=5), shape=[1, 1], trainable=False)
#
#     def loss(vars):
#         return tf.reduce_sum(tf.square(vars, name='var_squared'))
#
#     return x, loss
#
#
# dim_1, dim_2 = 2, 2
# flat = dim_1 * dim_2
#
# lstm_cell = tf.contrib.rnn.BasicLSTMCell(10)
# hidden_states = lstm_cell.zero_state(flat, tf.float32)
# x = tf.get_variable(
#         "x",
#         shape=[dim_1, dim_2],
#         dtype=tf.float32,
#         initializer=tf.random_normal_initializer(), trainable=False)
#
# def loss(x):
#     return tf.reduce_sum(tf.matmul(x, x))
#
# w = tf.get_variable('softmax_w', [10, 1])
# b = tf.get_variable('softmax_b', [1])
#
# gradients = tf.gradients(loss(x), x)[0]
# gradients_reshaped = tf.reshape(gradients, [flat, 1])
#
# with tf.variable_scope('rnn'):
#     lstm_cell(gradients_reshaped, hidden_states)
#
# def get_deltas(gradients, hidden_states):
#     with tf.variable_scope('rnn', reuse=True):
#         output, hidden_state_next = lstm_cell(gradients, hidden_states)
#         deltas = tf.add(tf.matmul(output, w), b)
#     return tf.reshape(deltas, [dim_1, dim_2]), hidden_state_next
#
# deltas, hidden_states_next = get_deltas(gradients_reshaped, hidden_states)
#
# iis = tf.InteractiveSession()
# iis.run(tf.global_variables_initializer())
# iis.run(x)
# iis.run(deltas)
# lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn')
#
# print 'default;'
# print x.eval(), lstm_vars[-1].eval()
# saver = tf.train.Saver(lstm_vars)
#
# saver.save(iis, 'model_rnn')
# iis.run(tf.assign(lstm_vars[-1], tf.random_normal(shape=[40])))
# iis.run(tf.assign(x, tf.random_normal(shape=[dim_1, dim_2])))
#
# print 'ass;'
# print x.eval(), lstm_vars[-1].eval()
#
# print 'restored'
# saver.restore(iis, 'model_rnn')
# print x.eval(), lstm_vars[-1].eval()
p = 5.

def clamp(gradients, min_value=None, max_value=None):
    if min_value is not None:
        gradients = np.maximum(gradients, min_value)
    if max_value is not None:
        gradients = np.minimum(gradients, max_value)
    return gradients

def process(gradients):

    eps = np.finfo(float).eps
    # gradients = np.array(gradients)
    log = np.log(np.abs(gradients) + eps)
    clamped_log = clamp(log / p, min_value=-1.0)
    sign = clamp(gradients * np.exp(p), min_value=-1.0, max_value=1.0)

    # return np.concat([clamped_log, sign], 1)
    # return np.hstack((clamped_log, sign))
    return clamped_log, sign
