from abc import ABCMeta
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset
import preprocess


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
    do_preprocessing = None

    def __init__(self, args={}):
        self.dims = args['dims'] if args.has_key('dims') else 1
        self.dtype = args['dtype'] if args.has_key('dtype') else tf.float32
        self.meta = args['meta'] if args.has_key('meta') else True
        self.variables = []
        self.constants = []
        self.variables_flattened_shape = []

    def create_variable(self, name, initializer=tf.random_normal_initializer(mean=0, stddev=0.01), constant=False, dims=None):
        shape = [self.dims, 1] if dims is None else dims
        variable = tf.get_variable(name, shape=shape, dtype=self.dtype,
                                   initializer=initializer, trainable=self.is_trainable)
        if constant:
            self.constants.append(variable)
        else:
            self.variables.append(variable)
            self.variables_flattened_shape.append(np.multiply.reduce(shape))
        return variable

    @property
    def is_trainable(self):
        return not self.meta

    def loss(self, variables):
        pass

    def pre_process(self, gradients):
        return gradients

    def gradients(self, variables):
        return tf.gradients(self.loss(variables), variables)

    def get_gradients(self, variables):
        gradients = self.gradients(variables)
        for i, gradient in enumerate(gradients):
            gradients[i] = self.pre_process(tf.reshape(gradients[i], [self.variables_flattened_shape[i], 1]))
        return gradients


class ElementwiseSquare(Problem):

    x = None

    def __init__(self, args):
        super(ElementwiseSquare, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', tf.random_uniform_initializer())

    def loss(self, variables):
        return tf.reduce_sum(tf.square(variables[0], name='x_squared'))


class FitX(Problem):

    w, x, y = None, None, None

    def __init__(self, args):
        super(FitX, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.w = self.create_variable('w', initializer=tf.random_uniform_initializer(), dims=[self.dims, self.dims])

        with tf.variable_scope(self.constant_scope):
            self.x = self.create_variable('x', initializer=tf.random_uniform_initializer(), dims=[self.dims, 1], constant=True)
            self.y = self.create_variable('y', initializer=tf.random_uniform_initializer(), dims=[self.dims, 1], constant=True)

    def loss(self, variables):
        return tf.reduce_sum(tf.square(tf.subtract(tf.matmul(variables[0], self.x), self.y)))

class TwoVars(Problem):

    x, y = None, None

    def __init__(self, args):
        super(TwoVars, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', initializer=tf.random_normal_initializer(), dims=[1, self.dims])
            self.y = self.create_variable('y', initializer=tf.random_normal_initializer())

    def loss(self, variables):
        return tf.reduce_sum(tf.matmul(tf.square(variables[0], name='x_square'), tf.square(variables[1], name='y_square'), name='matmul'))

class Quadratic(Problem):

    W, Y = None, None

    def __init__(self, args):
        super(Quadratic, self).__init__(args=args)
        stddev = args['stddev']
        dtype = args['dtype']

        with tf.variable_scope('variables'):
            self.variables = tf.get_variable("x", shape=[self.batch_size, self.dims], dtype=self.dtype,
                                             initializer=tf.random_normal_initializer(stddev=stddev), trainable=self.is_trainable)

        with tf.variable_scope('constant'):
            self.Y = tf.get_variable("Y", shape=[self.batch_size, self.dims], dtype=dtype,
                                     initializer=tf.random_uniform_initializer(), trainable=self.is_trainable)
            self.W = tf.get_variable("W", shape=[self.batch_size, self.dims, self.dims], dtype=self.dtype,
                                     initializer=tf.random_uniform_initializer(), trainable=self.is_trainable)
            self.const = [self.W, self.Y]

    def loss(self, variables):
        product = tf.squeeze(tf.matmul(self.W, tf.expand_dims(variables, -1)))
        return tf.reduce_mean(tf.reduce_sum((product - self.Y) ** 2, 1))

class Mnist(Problem):

    train_data = None
    test_data = None
    w_1 = None
    w_out = None
    b_1 = None
    b_out = None
    pre = None
    def __init__(self, args):
        super(Mnist, self).__init__(args=args)
        if args['preprocess']:
            self.pre = preprocess.LogAndSign(args['p'])
        def get_data(data, mode):
            mode_data = getattr(data, mode)
            images = tf.constant(mode_data.images, dtype=tf.float32, name="MNIST_images_" + mode)
            # labels = tf.constant(mode_data.labels, dtype=tf.int64, name="MNIST_labels_" + mode)
            labels = tf.one_hot(mode_data.labels, 10, name="MNIST_labels_" + mode)
            return images, labels
        data = mnist_dataset.load_mnist()
        self.train_data = {}
        self.train_data['images'], self.train_data['labels'] = get_data(data, 'train')
        # self.test_data['images'], self.test_data['labels'] = get_data(data, 'test')

        with tf.variable_scope(self.variable_scope):
            with tf.variable_scope('network_variables'):
                self.w_1 = self.create_variable('w_1', dims=[self.train_data['images'].get_shape()[1].value, 20])
                self.b_1 = self.create_variable('b_1', dims=[1, 20])
                self.w_out = self.create_variable('w_out', dims=[20, 10])
                self.b_out = self.create_variable('b_out', dims=[1, 10])
    
    def __xent_loss(self, output, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels)
        return tf.reduce_mean(loss)

    def network(self, batch, variables):
        layer_1 = tf.sigmoid(tf.add(tf.matmul(batch, variables[0]), variables[1]))
        layer_out = tf.add(tf.matmul(layer_1, variables[2]), variables[3])
        return layer_out

    def get_batch(self):
        indices = tf.random_uniform([128], 0, self.train_data['images'].get_shape()[0].value, tf.int64)
        # indices = tf.range(1)
        batch_images = tf.gather(self.train_data['images'], indices)
        batch_labels = tf.gather(self.train_data['labels'], indices)
        return batch_images, batch_labels
    
    def loss(self, variables):
        batch_images, batch_labels = self.get_batch()
        output = self.network(batch_images, variables)
        return self.__xent_loss(output, batch_labels)

    def pre_process(self, gradients):
        if self.pre is not None:
            return self.pre.process(gradients)
        else:
            return gradients


            

                




