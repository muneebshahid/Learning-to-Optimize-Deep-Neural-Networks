from abc import ABCMeta
import tensorflow as tf
import numpy as np


class Problem():
    __metaclass__ = ABCMeta

    dims = None
    dtype = None
    variables = None
    constants = None
    variables_flattened_shape = None    
    meta = None
    variable_scope = 'variables'
    constant_scope = 'constants'

    def __init__(self, args={'dims': 1, 'dtype': tf.float32}, meta=True):
        self.dims = args['dims']
        self.dtype = args['dtype']
        self.meta = meta        
        self.variables = []
        self.constants = []
        self.variables_flattened_shape = []

    def create_variable(self, name, initializer, constant=False, dims=None):
        shape = [self.dims, 1] if dims is None else dims
        variable = tf.get_variable(name, shape=shape, dtype=self.dtype,
                                   initializer=initializer, trainable=self.is_trainalbe)
        if constant:
            self.constants.append(variable)
        else:
            self.variables.append(variable)
            self.variables_flattened_shape.append(np.multiply.reduce(shape))
        return variable

    @property
    def is_trainalbe(self):
        return not self.meta

    def loss(self, vars):
        pass

    def get_gradients(self, vars):
        gradients = tf.gradients(self.loss(vars), vars)
        for i, gradient in enumerate(gradients):
            gradients[i] = tf.reshape(gradients[i], [self.variables_flattened_shape[i], 1])
        return gradients


class ElementwiseSquare(Problem):     

    x = None

    def __init__(self, args, meta=True):
        super(ElementwiseSquare, self).__init__(args=args, meta=meta)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', tf.random_uniform_initializer())

    def loss(self, vars):
        return tf.reduce_sum(tf.square(vars[0], name='x_squared'))        


class FitX(Problem):

    w, x, y = None, None, None

    def __init__(self, args):
        super(FitX, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.w = self.create_variable('w', initializer=tf.random_uniform_initializer(), dims=[self.dims, self.dims])

        with tf.variable_scope(self.constant_scope):
            self.x = self.create_variable('x', initializer=tf.random_uniform_initializer(), dims=[self.dims, 1], constant=True)
            self.y = self.create_variable('y', initializer=tf.random_uniform_initializer(), dims=[self.dims, 1], constant=True)

    def loss(self, vars):
        return tf.reduce_sum(tf.square(tf.subtract(tf.matmul(vars[0], self.x), self.y)))

class TwoVars(Problem):

    x, y = None, None

    def __init__(self, args):
        super(TwoVars, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', initializer=tf.random_normal_initializer(), dims=[1, self.dims])
            self.y = self.create_variable('y', initializer=tf.random_normal_initializer())

    def loss(self, vars):
        return tf.reduce_sum(tf.matmul(tf.square(vars[0], name='x_square'), tf.square(vars[1], name='y_square'), name='matmul'))

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
