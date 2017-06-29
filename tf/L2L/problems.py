from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset
import os, tarfile
import six.moves
from six.moves import urllib, xrange


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
    allow_gradients_of_gradients = None
    variables_flat = None

    def __init__(self, args={}):
        self.allow_gradients_of_gradients = args['gog'] if 'gog' in args else False
        self.dims = args['dims'] if 'dims' in args else 1
        self.dtype = args['dtype'] if 'dtype' in args else tf.float32
        self.meta = args['meta'] if 'meta' in args else True
        self.variables = []
        self.variables_flat = []
        self.constants = []
        self.variables_flattened_shape = []

    def create_variable(self, name, initializer=tf.random_normal_initializer(mean=0, stddev=0.01), constant=False, dims=None):
        shape = [self.dims, 1] if dims is None else dims
        with tf.name_scope('problem_variables'):
            variable = tf.get_variable(name, shape=shape, dtype=self.dtype,
                                       initializer=initializer, trainable=self.is_trainable)
        if constant:
            self.constants.append(variable)
        else:
            flat_shape = np.multiply.reduce(shape)
            self.variables.append(variable)
            self.variables_flattened_shape.append(flat_shape)
            self.variables_flat.append(tf.reshape(variable, [flat_shape, 1]))
        return variable

    @property
    def is_trainable(self):
        return not self.meta

    def loss(self, variables, mode='train'):
        pass

    def flatten_input(self, i, inputs):
        return tf.reshape(inputs, [self.variables_flattened_shape[i], 1])

    def get_shape(self, i=None, variable=None):
        shape = self.variables_flat[i].get_shape()[0] if i is not None else variable.get_shape()[0]
        return shape

    def set_shape(self, input, i=None, like_variable=None, op_name=''):
        shape = self.variables[i].get_shape() if i is not None else like_variable.get_shape()
        return tf.reshape(input, shape=shape, name=op_name)

    def get_gradients_raw(self, variables=None):
        variables = self.variables if variables is None else variables
        gradients = tf.gradients(self.loss(variables), variables)
        if not self.allow_gradients_of_gradients:
            gradients = [tf.stop_gradient(gradient) for gradient in gradients]
        return gradients

    def get_gradients(self, variables=None):
        variables = variables if variables is not None else self.variables
        gradients = self.get_gradients_raw(variables)
        gradients = [self.flatten_input(i, gradient) for i, gradient in enumerate(gradients)]
        return gradients


class ElementwiseSquare(Problem):

    x = None

    def __init__(self, args):
        super(ElementwiseSquare, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', initializer=tf.random_uniform_initializer(minval=args['minval'], maxval=args['maxval']), dims=args['dims'])

    def loss(self, variables, mode='train'):
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

    def loss(self, variables, mode='train'):
        return tf.reduce_sum(tf.square(tf.subtract(tf.matmul(variables[0], self.x), self.y)))

class TwoVars(Problem):

    x, y = None, None

    def __init__(self, args):
        super(TwoVars, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', initializer=tf.random_normal_initializer(), dims=[1, self.dims])
            self.y = self.create_variable('y', initializer=tf.random_normal_initializer())

    def loss(self, variables, mode='train'):
        return tf.reduce_sum(tf.matmul(tf.square(variables[0], name='x_square'), tf.square(variables[1], name='y_square'), name='matmul'))

class Quadratic(Problem):

    W, Y = None, None

    def __init__(self, args):
        super(Quadratic, self).__init__(args=args)
        stddev = args['stddev']

        with tf.variable_scope('variables'):
            self.variables = tf.get_variable("x", shape=[self.batch_size, self.dims], dtype=self.dtype,
                                             initializer=tf.random_normal_initializer(stddev=stddev), trainable=self.is_trainable)

        with tf.variable_scope('constant'):
            self.Y = tf.get_variable("Y", shape=[self.batch_size, self.dims], dtype=self.dtype,
                                     initializer=tf.random_uniform_initializer(), trainable=self.is_trainable)
            self.W = tf.get_variable("W", shape=[self.batch_size, self.dims, self.dims], dtype=self.dtype,
                                     initializer=tf.random_uniform_initializer(), trainable=self.is_trainable)
            self.const = [self.W, self.Y]

    def loss(self, variables, mode='train'):
        product = tf.squeeze(tf.matmul(self.W, tf.expand_dims(variables, -1)))
        return tf.reduce_mean(tf.reduce_sum((product - self.Y) ** 2, 1))


class Mnist(Problem):

    training_data, test_data, validation_data = None, None, None
    conv = False

    def weight_variable(self, name, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


    def __init__(self, args):
        super(Mnist, self).__init__(args=args)
        self.conv = False if 'conv' not in args else args['conv']

        def get_data(data, mode='train'):
            mode_data = getattr(data, mode)
            images = tf.constant(mode_data.images, dtype=tf.float32, name="MNIST_images_" + mode)
            if self.conv:
                images = tf.reshape(images, [-1, 28, 28, 1])
            if self.allow_gradients_of_gradients:
                labels = tf.one_hot(mode_data.labels, 10, name="MNIST_labels_" + mode)
            else:
                labels = tf.constant(mode_data.labels, dtype=tf.int64, name="MNIST_labels_" + mode)
            return images, labels

        data = mnist_dataset.load_mnist()
        self.training_data, self.test_data, self.validation_data = dict(), dict(), dict()
        self.training_data['images'], self.training_data['labels'] = get_data(data, 'train')
        self.test_data['images'], self.test_data['labels'] = get_data(data, 'test')
        self.validation_data['images'], self.validation_data['labels'] = get_data(data, 'validation')

        with tf.variable_scope(self.variable_scope):
            with tf.variable_scope('network_variables'):
                if self.conv:
                    self.create_variable('w_1', dims=[5, 5, 1, 32])
                    self.create_variable('b_1', dims=[32])

                    self.create_variable('w_2', dims=[5, 5, 32, 64])
                    self.create_variable('b_2', dims=[64])

                    self.create_variable('w_3', dims=[7 * 7 * 64, 1024])
                    self.create_variable('b_3', dims=[1024])

                    self.create_variable('w_out', dims=[1024, 10])
                    self.create_variable('b_out', dims=[10])
                else:
                    self.create_variable('w_1', dims=[self.training_data['images'].get_shape()[1].value, 20])
                    self.create_variable('b_1', dims=[1, 20])

                    # self.create_variable('w_2', dims=[20, 20])
                    # self.create_variable('b_2', dims=[20])
                    #
                    # self.create_variable('w_3', dims=[20, 20])
                    # self.create_variable('b_3', dims=[20])

                    self.create_variable('w_out', dims=[20, 10])
                    self.create_variable('b_out', dims=[1, 10])


    def __xent_loss(self, output, labels):
        if self.allow_gradients_of_gradients:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels)
        return tf.reduce_mean(loss)

    def network(self, batch, variables):
        if self.conv:
            h_conv1 = tf.nn.relu(Mnist.conv2d(batch, variables[0]) + variables[1])
            h_pool1 = Mnist.max_pool_2x2(h_conv1)
            h_conv2 = tf.nn.relu(Mnist.conv2d(h_pool1, variables[2]) + variables[3])
            h_pool2 = Mnist.max_pool_2x2(h_conv2)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, variables[4]) + variables[5])
            y_conv = tf.matmul(h_fc1, variables[6]) + variables[7]
            return y_conv
        else:
            layer_1 = tf.sigmoid(tf.add(tf.matmul(batch, variables[0]), variables[1]))
            # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, variables[2]), variables[3]))
            # layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, variables[4]), variables[5]))
            layer_out = tf.add(tf.matmul(layer_1, variables[2]), variables[3])
            return layer_out

    def get_batch(self, mode='train'):
        data_holder = None
        if mode == 'train':
            data_holder = self.training_data
        elif mode == 'validation':
            data_holder = self.validation_data
        elif mode == 'test':
            data_holder = self.test_data
        indices = tf.random_uniform([128], 0, data_holder['images'].get_shape()[0].value, tf.int64)
        batch_images = tf.gather(data_holder['images'], indices)
        batch_labels = tf.gather(data_holder['labels'], indices)
        return batch_images, batch_labels
    
    def loss(self, variables, mode='train'):
        batch_images, batch_labels = self.get_batch(mode)
        output = self.network(batch_images, variables)
        return self.__xent_loss(output, batch_labels)


            
class cifar10(Problem):

    def __init__(self, args):
        CIFAR10_URL = "http://www.cs.toronto.edu/~kriz"
        CIFAR10_FILE = "cifar-10-binary.tar.gz"
        CIFAR10_FOLDER = "cifar-10-batches-bin"
        def maybe_download_cifar10(path):
            """Download and extract the tarball from Alex's website."""
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.join(path, CIFAR10_FILE)
            if not os.path.exists(filepath):
                print("Downloading CIFAR10 dataset to {}".format(filepath))
                url = os.path.join(CIFAR10_URL, CIFAR10_FILE)
                filepath, _ = urllib.request.urlretrieve(url, filepath)
                statinfo = os.stat(filepath)
                print("Successfully downloaded {} bytes".format(statinfo.st_size))
                tarfile.open(filepath, "r:gz").extractall(path)
        path = args['path']
        maybe_download_cifar10(path)
        super(cifar10, self).__init__(args)

        self.w1 = self.create_variable('w1', dims=[5, 5, 3, 16])
        self.b1 = self.create_variable('b1', dims=[16])
        self.w2 = self.create_variable('w2', dims=[5, 5, 16, 16])
        self.b2 = self.create_variable('b2', dims=[16])
        self.w3 = self.create_variable('w3', dims=[5, 5, 16, 16])
        self.b3 = self.create_variable('b3', dims=[16])
        self.w4 = self.create_variable('w4', dims=[256, 32])
        self.b4 = self.create_variable('b4', dims=[32])
        self.w5 = self.create_variable('w5', dims=[32, 10])
        self.b5 = self.create_variable('b5', dims=[10])
        # Read images and labels from disk.
        filenames = [os.path.join(path, CIFAR10_FOLDER, "data_batch_{}.bin".format(i)) for i in six.moves.xrange(1, 6)]

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        depth = 3
        height = 32
        width = 32
        label_bytes = 1
        image_bytes = depth * height * width
        record_bytes = label_bytes + image_bytes
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, record = reader.read(tf.train.string_input_producer(filenames))
        record_bytes = tf.decode_raw(record, tf.uint8)

        label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
        raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
        image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
        # height x width x depth.
        image = tf.transpose(image, [1, 2, 0])
        image = tf.div(image, 255)

        self.queue = tf.RandomShuffleQueue(capacity=1000 + 3 * 128,
                                      min_after_dequeue=1000,
                                      dtypes=[tf.float32, tf.int32],
                                      shapes=[image.get_shape(), label.get_shape()])
        enqueue_ops = [self.queue.enqueue([image, label]) for _ in six.moves.xrange(4)]
        tf.train.add_queue_runner(tf.train.QueueRunner(self.queue, enqueue_ops))



    def network(self, batch, variables):
        def conv_activation(x):
            return tf.nn.max_pool(tf.nn.relu(x),
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME")
        conv1 = conv_activation(tf.nn.bias_add(tf.nn.conv2d(batch, variables[0], [1, 1, 1, 1], padding='SAME'), variables[1]))
        conv2 = conv_activation(
            tf.nn.bias_add(tf.nn.conv2d(conv1, variables[2], [1, 1, 1, 1], padding='SAME'), variables[3]))
        conv3 = conv_activation(
            tf.nn.bias_add(tf.nn.conv2d(conv2, variables[4], [1, 1, 1, 1], padding='SAME'), variables[5]))
        reshaped_conv3 = tf.reshape(conv3, [batch.shape[0].value, -1], 'reshape_fully_connected')
        linear = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshaped_conv3, variables[6]), variables[7]))
        out = tf.nn.bias_add(tf.matmul(linear, variables[8]), variables[9])
        return out

    def loss(self, variables, mode='train'):
        def xent_loss(output, labels):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                                  labels=labels)
            return tf.reduce_mean(loss)
        image_batch, label_batch = self.queue.dequeue_many(128)
        label_batch = tf.reshape(label_batch, [128])
        output = self.network(image_batch, variables)
        return xent_loss(output, label_batch)








