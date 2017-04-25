from abc import ABCMeta
import tensorflow as tf


class Problem():
    __metaclass__ = ABCMeta
    
    batch_size = None
    dims = None
    dtype = None
    vars = None
    const = None
    meta = None

    def __init__(self, args={'batch_size': 1, 'dims': 1, 'dtype': tf.float32}, meta=True):
        self.batch_size = args['batch_size']
        self.dims = args['dims']
        self.dtype = args['dtype']
        self.meta = meta
    
    @property
    def is_trainalbe(self):
        return not self.meta

    def loss(self, vars):
        pass

    def gradients(self, vars):
        return tf.gradients(self.loss(vars), vars)[0]


class ElementwiseSquare(Problem):     

    def __init__(self, args, meta=True):
        super(ElementwiseSquare, self).__init__(args=args, meta=meta)
        with tf.variable_scope('variables'):
            self.vars = tf.get_variable('x', shape=[self.batch_size, self.dims], dtype=self.dtype,
                                   initializer=tf.random_uniform_initializer(), trainable=self.is_trainalbe)
    def loss(self, vars):
        return tf.reduce_sum(tf.square(vars, name='var_squared'))


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
