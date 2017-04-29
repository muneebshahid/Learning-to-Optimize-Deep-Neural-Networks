from abc import ABCMeta
import tensorflow as tf


class Problem():
    __metaclass__ = ABCMeta
    
    batch_size = None
    dims = None
    dtype = None
    vars = None
    vars_flattened_shape = None
    const = None
    meta = None
    variables = 'variables'
    constants = 'constants'

    def __init__(self, args={'batch_size': 1, 'dims': 1, 'dtype': tf.float32}, meta=True):
        self.batch_size = args['batch_size']
        self.dims = args['dims']
        self.dtype = args['dtype']
        self.meta = meta
        self.vars = []
        self.vars_flattened_shape = []

    def create_variable(self, name, initializer):        
        variable = tf.get_variable(name, shape=[self.batch_size, self.dims], dtype=self.dtype,
                                   initializer=initializer, trainable=self.is_trainalbe)
        self.vars.append(variable)
        self.vars_flattened_shape.append(self.batch_size * self.dims)
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
        with tf.variable_scope(self.variables):
            self.x = self.create_variable('x', tf.random_uniform_initializer())

    def loss(self, vars):
        return tf.reduce_sum(tf.square(vars[0], name='x_squared'))

class TwoVars(Problem):

    x, y = None, None

    def __init__(self):
        super(TwoVars, self).__init__()
        with tf.variable_scope(self.variables):
            self.x = self.create_variable('x', initializer=tf.random_uniform_initializer())
            self.y = self.create_variable('y', initializer=tf.random_uniform_initializer())

    def loss(self, vars):
        return tf.add(tf.square(vars[0], name='x_square'), tf.square(vars[1], name='y_square'), name='sum')

class Quadratic(Problem):

    W, Y = None, None

    def __init__(self, args, meta=True):
        super(Quadratic, self).__init__(args=args, meta=meta)
        stddev = args['stddev']
        dtype = args['dtype']

        with tf.variable_scope('variables'):
            self.vars = tf.get_variable("x", shape=[self.batch_size, self.dims], dtype=self.dtype,
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
