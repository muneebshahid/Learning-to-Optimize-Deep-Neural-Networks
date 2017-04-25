from abc import ABCMeta
import tensorflow as tf


class Problem():
    __metaclass__ = ABCMeta
    
    batch_size = None
    dim = None
    vars = None
    const = None

    def __init__(self, args=None, meta=True):
        self.batch_size = 1
        self.dim = 1
        pass

    def loss(self, vars):
        pass

    def gradients(self):
        pass


class ElementwiseSquare(Problem):     

    def __init__(self, args, meta=True):
        self.batch_size = args['batch_size']
        self.dim = args['dim']
        with tf.variable_scope('variables'):
            self.vars = tf.get_variable('x', shape=[self.batch_size, self.dim], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(), trainable=(not meta))
    def loss(self, vars):
        return tf.reduce_sum(tf.square(vars, name='var_squared'))

    def gradients(self, vars):
        return tf.gradients(self.loss(vars), vars)[0]


class Quadratic(Problem):

    # num_dims = None
    W, Y = None, None
    def __init__(self, args):
        self.batch_size = args['batch_size']
        num_dims = args['num_dims']
        stddev = args['stddev']
        dtype = args['dtype']

        with tf.variable_scope('variables'):
            self.vars = tf.get_variable("x", shape=[self.batch_size, num_dims], dtype=dtype,
                            initializer=tf.random_normal_initializer(stddev=stddev), trainable=False)

        with tf.variable_scope('constant'):
            self.Y = tf.get_variable("Y", shape=[self.batch_size, num_dims], dtype=dtype,
                            initializer=tf.random_uniform_initializer(), trainable=False)
            self.W = tf.get_variable("W", shape=[self.batch_size, num_dims, num_dims], dtype=dtype,
                        initializer=tf.random_uniform_initializer(), trainable=False)
            self.const = [self.W, self.Y]

    def loss(self, vars):
        product = tf.squeeze(tf.matmul(self.W, tf.expand_dims(vars, -1)))
        return tf.reduce_mean(tf.reduce_sum((product - self.Y) ** 2, 1))
    
    def gradients(self, vars):
        return tf.gradients(self.loss(vars), vars)[0]
