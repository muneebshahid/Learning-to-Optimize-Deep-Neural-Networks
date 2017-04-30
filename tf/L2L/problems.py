from abc import ABCMeta
import tensorflow as tf
import numpy as np


class Problem():
    __metaclass__ = ABCMeta
    
    batch_size = None
    dims = None
    dtype = None
    variables = None
    constants = None
    variables_flattened_shape = None    
    meta = None
    variable_scope = 'variables'
    constant_scope = 'constants'

    def __init__(self, args={'batch_size': 1, 'dims': 1, 'dtype': tf.float32}, meta=True):
        self.batch_size = args['batch_size']
        self.dims = args['dims']
        self.dtype = args['dtype']
        self.meta = meta
        self.variables = []
        self.constants = []
        self.variables_flattened_shape = []

    def create_variable(self, name, initializer, variable=True, dims=None):
        shape = [self.batch_size, self.dims] if dims is None else dims
        variable = tf.get_variable(name, shape=shape, dtype=self.dtype,
                                   initializer=initializer, trainable=self.is_trainalbe)
        if variable:
            self.variables.append(variable)
            self.variables_flattened_shape.append(np.multiply.reduce(shape))
        else:
            self.constants.append(variable)
        return variable

    @property
    def is_trainalbe(self):
        return not self.meta

    def loss(self, vars):
        pass

    def get_gradients(self, vars):
        return tf.gradients(self.loss(vars), vars)


class ElementwiseSquare(Problem):     

    x = None

    def __init__(self, args, meta=True):
        super(ElementwiseSquare, self).__init__(args=args, meta=meta)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', tf.random_uniform_initializer())

    def loss(self, vars):
        return tf.reduce_sum(tf.square(vars[0], name='x_squared'))

class TwoVars(Problem):

    x, y = None, None

    def __init__(self, args):
        super(TwoVars, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', initializer=tf.random_normal_initializer())
            self.y = self.create_variable('y', initializer=tf.random_normal_initializer())

    def loss(self, vars):
        return tf.reduce_sum(tf.add(tf.square(vars[0], name='x_square'), tf.square(vars[1], name='y_square'), name='sum'))


class FitW(Problem):

    w, x, y = None, None, None

    def __init__(self, args):
        super(FitW, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.w = self.create_variable('w', initializer=tf.random_uniform_initializer(), shape=[self.dims, self.dims])

        with tf.variable_scope(self.constant_scope):
            self.x = self.create_variable('x', initializer=tf.random_uniform_initializer(), shape=[self.dims, 1], variable=False)
            self.y = self.create_variable('y', initializer=tf.random_uniform_initializer(), shape=[self.dims, 1], variable=False)
            

    def loss(self, vars):
        return tf.subtract(tf.matmul(vars[0], self.constants[0]), self.constants[1])

class Quadratic(Problem):

    W, Y = None, None

    def __init__(self, args, meta=True):
        super(Quadratic, self).__init__(args=args, meta=meta)
        stddev = args['stddev']
        dtype = args['dtype']

        with tf.variable_scope('variables'):
            self.variables = tf.get_variable("x", shape=[self.batch_size, self.dims], dtype=self.dtype,
                            initializer=tf.random_normal_initializer(stddev=stddev), trainable=self.is_trainalbe)

        with tf.variable_scope('constant'):
            self.Y = tf.get_variable("Y", shape=[self.batch_size, self.dims], dtype=dtype,
                            initializer=tf.random_uniform_initializer(), trainable=self.is_trainalbe)
            self.W = tf.get_variable("W", shape=[self.batch_size, self.dims, self.dims], dtype=self.dtype,
                        initializer=tf.random_uniform_initializer(), trainable=self.is_trainalbe)
            self.const = [self.W, self.Y]

    def loss(self, vars):
        product = tf.squeeze(tf.matmul(self.W, tf.expand_dims(vars, -1)))
        return tf.reduce_mean(tf.reduce_sum((product - self.Y) ** 2, 1))
