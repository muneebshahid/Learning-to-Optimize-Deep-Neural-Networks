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

def create_batches(problem, batches=5, dims=5, args={}):
    batch_list = []
    for batch in range(batches):
        batch_list.extend(problem(args))
    return batch_list

def create_batches_all(train=True):
    batches = []
    reset_limit = []
    def original_mirror(class_ref, name, dims, minval, maxval):
        original = class_ref(
            {'prefix': class_ref.__name__ + '_' + name, 'dims': dims, 'minval': minval, 'maxval': maxval})

        mirror = class_ref({'prefix': class_ref.__name__ + '_' + name + '_mirror', 'dims': dims,
                                  'init': [variable.initialized_value() * -1.0 for variable in original.variables]})

        batches.append(original)
        # batches.append(mirror)

    if train:
        # ElementSquare
        # batches.append(
        #     ElementwiseSquare(
        #         {'prefix': ElementwiseSquare.__name__ + '_0_', 'dims': 1000, 'minval': -10.0, 'maxval': 10.0}))
        # reset_limit.append([[50, 300], [100, 500]])
        # #
        # Rosenbrock
        # original_mirror(Rosenbrock, '_0_', None, -10, 10)
        # reset_limit.append([[50, 300], [500, 5000]])
        # reset_limit.append([[50, 300], [500, 5000]])
        # #
        # batches.append(RosenbrockMulti({'prefix': RosenbrockMulti.__name__ + '_0_', 'dims': 20, 'minval': -10.0, 'maxval': 10.0}))
        # reset_limit.append([[50, 300], [500, 5000]])
        # #
        # original_mirror(Booth, '_0_', 2, -10.0, 10.0)
        # reset_limit.append([[50, 300], [500, 5000]])
        # reset_limit.append([[50, 300], [500, 5000]])
        # # #
        # # # # DifferentPower
        # for i in range(4):
        #     batches.append(DifferentPowers({'prefix': DifferentPowers.__name__ + '_'+ str(i) + '_', 'dims': i + 3, 'minval': -10.0, 'maxval': 10.0}))
        #     reset_limit.append([[50, 200], [100, 500]])
        # #
        # batches.append(FitX({'prefix': FitX.__name__ + '_0_', 'dims': 10, 'minval': -100.0, 'maxval': 100.0}))
        # reset_limit.append([[50, 200], [100, 500]])

         batches.append(Mnist({'minval': -100.0, 'maxval': 100.0}))
         reset_limit.append([[50, 200], [200, 10000]])
    else:
        batches.append(
            ElementwiseSquare({'prefix': ElementwiseSquare.__name__ + '_0_', 'dims': 4, 'init': tf.constant_initializer([100, 500, 600, 1000])}))
        batches.append(
            ElementwiseSquare({'prefix': ElementwiseSquare.__name__ + '_1_', 'dims': 4, 'init': tf.constant_initializer([-100, -500, -600, -1000])}))
        batches.append(
            ElementwiseSquare({'prefix': ElementwiseSquare.__name__ + '_2_', 'dims': 4, 'init': tf.constant_initializer([-0.5, -.36, 0, .4])}))

        batches.append(
            ElementwiseSquare({'prefix': ElementwiseSquare.__name__ + '_3_', 'dims': 1000, 'minval': -10.0, 'maxval': 10.0}))

        # batches.append(DifferentPowers({'prefix': DifferentPowers.__name__ + '_0_', 'dims': 5, 'minval': -10.0, 'maxval': 10.0}))
        batches.append(Rosenbrock({'prefix': Rosenbrock.__name__ + '_0_', 'init': [tf.constant_initializer([-3]), tf.constant_initializer([3.0])]}))
        batches.append(
            Rosenbrock({'prefix': Rosenbrock.__name__ + '_1_', 'init': [tf.constant_initializer([0]), tf.constant_initializer([0.0])]}))
        batches.append(
            Rosenbrock({'prefix': Rosenbrock.__name__ + '_2_', 'init': [tf.constant_initializer([10]), tf.constant_initializer([-10])]}))
        # batches.append(Mnist({}))
    return batches, reset_limit


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
    problem_prefix = None
    var_count = None
    init = None
    io_handle = None

    def __init__(self, args={}):
        self.allow_gradients_of_gradients = args['gog'] if 'gog' in args else True
        self.dims = args['dims'] if 'dims' in args else 1
        self.dtype = args['dtype'] if 'dtype' in args else tf.float32
        self.meta = args['meta'] if 'meta' in args else True
        self.variables = []
        self.variables_flat = []
        self.constants = []
        self.variables_flattened_shape = []
        self.var_count = args['var_count']
        self.problem_prefix = '' if 'prefix' not in args else args['prefix']
        self.init = args['init'] if 'init' in args else [tf.random_uniform_initializer(minval=args['minval'], maxval=args['maxval'])
                                             for variable in range(self.var_count)]

    def create_variable(self, name, initializer=tf.random_normal_initializer(mean=0, stddev=0.01), constant=False, dims=None):
        def add_to_list(variable):
            if constant:
                self.constants.append(variable)
            else:
                flat_shape = np.multiply.reduce(shape)
                # tf.summary.histogram(name, variable)
                self.variables.append(variable)
                self.variables_flattened_shape.append(flat_shape)
                self.variables_flat.append(tf.reshape(variable, [flat_shape, 1]))

        shape = [self.dims, 1] if dims is None else dims
        with tf.name_scope('problem_variables'):
            try:
                variable = tf.get_variable(self.problem_prefix + name, shape=shape, dtype=self.dtype,
                                       initializer=initializer, trainable=self.is_trainable)
            except ValueError:
                variable = tf.get_variable(self.problem_prefix + name, initializer=initializer,
                                           trainable=self.is_trainable)
            add_to_list(variable)
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
        # if not self.allow_gradients_of_gradients:
        #     gradients = [tf.stop_gradient(gradient) for gradient in gradients]
        return gradients

    def get_gradients(self, variables=None):
        variables = variables if variables is not None else self.variables
        gradients = self.get_gradients_raw(variables)
        gradients = [self.flatten_input(i, gradient) for i, gradient in enumerate(gradients)]
        return gradients

    def end_init(self):
        self.io_handle = tf.train.Saver(self.variables)

    def restore(self, sess, path):
        if self.io_handle is not None:
            self.io_handle.restore(sess, path)

    def accuracy(self, mode='train'):
        return []


class ElementwiseSquare(Problem):

    x = None

    def __init__(self, args):
        args['var_count'] = 1
        super(ElementwiseSquare, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', initializer=self.init[0])

    def loss(self, variables, mode='train'):
        return tf.reduce_sum(tf.square(variables[0], name='x_squared'))


class Booth(Problem):

    def __init__(self, args):
        args['var_count'] = 2
        super(Booth, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.create_variable('x_0', initializer=self.init[0], dims=[1, 1])
            self.create_variable('x_1', initializer=self.init[1], dims=[1, 1])

    def loss(self, variables, mode='train'):
        x_1, x_2 = variables[0], variables[1]
        return tf.square(x_1 + 2 * x_2 - 7) + tf.square(2 * x_1 + x_2 - 5)

class Rosenbrock(Problem):
    def __init__(self, args):
        args['var_count'] = 2
        super(Rosenbrock, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', initializer=self.init[0], dims=[1, 1])
            self.y = self.create_variable('y', initializer=self.init[1], dims=[1, 1])

    def loss(self, variables, mode='train'):
        return tf.square(1.0 - variables[0]) + 100 * tf.square(variables[1] - tf.square(variables[0]))


class RosenbrockMulti(Problem):

    def __init__(self, args):
        if 'dims' not in args:
            args['dims'] = 2
        args['var_count'] = args['dims']
        super(RosenbrockMulti, self).__init__(args)
        for i, dim in enumerate(range(self.dims)):
            self.create_variable('var_' + str(i), initializer=self.init[0], dims=[1, 1])

    def loss(self, variables, mode='train'):
        half_length = self.dims // 2
        sum = 0
        for i in range(half_length):
            index_i = 2 * i
            index_j = 2 * i + 1
            sum = tf.add(100 * tf.square(tf.square(variables[index_i]) - variables[index_j]) + tf.square(variables[index_i] - 1), sum)
        return sum

class DifferentPowers(Problem):

    def __init__(self, args):
        if 'dims' not in args or args['dims'] == 1:
            args['dims'] = 2
        args['var_count'] = args['dims']
        super(DifferentPowers, self).__init__(args)
        for i, dim in enumerate(range(args['var_count'])):
            self.create_variable('var_' + str(i), initializer=self.init[i], dims=[1, 1])

    def loss(self, variables, mode='train'):
        loss = 0
        for i, variable in enumerate(variables):
            loss += tf.pow(tf.abs(variable), 2 + 4 * (i) / (len(variables) - 1))
        return loss

class FitX(Problem):

    w, x, y = None, None, None

    def __init__(self, args):
        args['var_count'] = 3
        super(FitX, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.w = self.create_variable('w', initializer=self.init[0], dims=[self.dims, self.dims])

        with tf.variable_scope(self.constant_scope):
            self.x = self.create_variable('x', initializer=self.init[1], dims=[self.dims, 1], constant=True)
            self.y = self.create_variable('y', initializer=self.init[2], dims=[self.dims, 1], constant=True)

    def loss(self, variables, mode='train'):
        return tf.reduce_sum(tf.square(tf.subtract(tf.matmul(variables[0], self.x), self.y)))


class TwoVars(Problem):

    x, y = None, None

    def __init__(self, args):
        super(TwoVars, self).__init__(args=args)
        with tf.variable_scope(self.variable_scope):
            self.x = self.create_variable('x', initializer=self.init[0], dims=[1, self.dims])
            self.y = self.create_variable('y', initializer=self.init[1])

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
    enable_l2_norm = False
    weight_norm_loss = None

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
        args['var_count'] = 4
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
                self.weight_norm_loss = 0
                for variable in self.variables:
                    self.weight_norm_loss += tf.nn.l2_loss(variable)
        self.end_init()


    def accuracy(self, mode='train'):
        batch_images, batch_labels = self.get_batch(mode)
        output = self.network(batch_images, self.variables)
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(batch_labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction)

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
        return self.__xent_loss(output, batch_labels) + (.01 * self.weight_norm_loss if self.enable_l2_norm else 0.0)


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
        args['var_count'] = 1
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


    def accuracy(self, mode='train'):
        image_batch, label_batch = self.queue.dequeue_many(128)
        label_batch = tf.cast(tf.reshape(label_batch, [128, 1]), tf.int64)
        print(label_batch)
        output = self.network(image_batch, self.variables)
        print(output)
        #correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label_batch, 1))

        correct_prediction = tf.equal(tf.argmax(output, 1), label_batch)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction)


class cifar10_full(Problem):

    # Process images of this size. Note that this differs from the original CIFAR
    # image size of 32 x 32. If one alters this number, then the entire model
    # architecture will change and any model would need to be retrained.
    IMAGE_SIZE = 24

    # Global constants describing the CIFAR-10 data set.
    NUM_CLASSES = 10
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

    def read_cifar10(self, filename_queue):
        """Reads and parses examples from CIFAR10 data files.
        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.
        Args:
          filename_queue: A queue of strings with the filenames to read from.
        Returns:
          An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
              for this example.
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
        """

        class CIFAR10Record(object):
            pass

        result = CIFAR10Record()

        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        label_bytes = 1  # 2 for CIFAR-100
        result.height = 32
        result.width = 32
        result.depth = 3
        image_bytes = result.height * result.width * result.depth
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.  No
        # header or footer in the CIFAR-10 format, so we leave header_bytes
        # and footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        result.label = tf.cast(
            tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes],
                             [label_bytes + image_bytes]),
            [result.depth, result.height, result.width])
        # Convert from [depth, height, width] to [height, width, depth].
        result.uint8image = tf.transpose(depth_major, [1, 2, 0])

        return result

    def _generate_image_and_label_batch(self, image, label, min_queue_examples,
                                        batch_size, shuffle):
        """Construct a queued batch of images and labels.
        Args:
          image: 3-D Tensor of [height, width, 3] of type.float32.
          label: 1-D Tensor of type.int32
          min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
          batch_size: Number of images per batch.
          shuffle: boolean indicating whether to use a shuffling queue.
        Returns:
          images: Images. 4D tensor of [batch_size, height, width, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.
        num_preprocess_threads = 16
        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

        # Display the training images in the visualizer.
        tf.summary.image('images', images)
        return images, tf.reshape(label_batch, [batch_size])

    def distorted_inputs(self, data_dir, batch_size):
        """Construct distorted input for CIFAR training using the Reader ops.
        Args:
          data_dir: Path to the CIFAR-10 data directory.
          batch_size: Number of images per batch.
        Returns:
          images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        """
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = self.read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = self.IMAGE_SIZE
        width = self.IMAGE_SIZE

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d CIFAR images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(float_image, read_input.label,
                                               min_queue_examples, batch_size,
                                               shuffle=True)

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable
        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            dtype = tf.float32#tf.float16 if FLAGS.use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        Returns:
          Variable Tensor
        """
        dtype = tf.float32#tf.float16 if FLAGS.use_fp16 else
        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return var

    def inputs(self, eval_data, data_dir, batch_size):
        """Construct input for CIFAR evaluation using the Reader ops.
        Args:
          eval_data: bool, indicating if one should use the train or eval data set.
          data_dir: Path to the CIFAR-10 data directory.
          batch_size: Number of images per batch.
        Returns:
          images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        """
        if not eval_data:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                         for i in xrange(1, 6)]
            num_examples_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            num_examples_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = self.read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = self.IMAGE_SIZE
        width = self.IMAGE_SIZE

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               height, width)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(float_image, read_input.label,
                                               min_queue_examples, batch_size,
                                               shuffle=False)

    def _add_to_list(self, variable, shape):
        flat_shape = np.multiply.reduce(shape)
        # tf.summary.histogram(name, variable)
        self.variables.append(variable)
        self.variables_flattened_shape.append(flat_shape)
        self.variables_flat.append(tf.reshape(variable, [flat_shape, 1]))

    def __init__(self, args):
        CIFAR10_URL = "http://www.cs.toronto.edu/~kriz"
        CIFAR10_FILE = "cifar-10-binary.tar.gz"
        CIFAR10_FOLDER = "cifar-10-batches-bin"
        self.full = args['full']

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
        path = os.path.join(path, CIFAR10_FOLDER)
        args['var_count'] = 1
        super(cifar10_full, self).__init__(args)
        if self.full:
            k_channels = 64
            f1_in = 6 * 6 * 64
            f1_out = 384
            f2_out = 192
        else:
            k_channels = 16
            f1_in = 6 * 6 * 16
            f1_out = 96
            f2_out = 46


        kernel_1 = self._variable_with_weight_decay('conv1_weights', shape=[5, 5, 3, k_channels], stddev=5e-2, wd=0.0)
        biases_1 = self._variable_on_cpu('conv1_biases', [k_channels], tf.constant_initializer(0.0))
        self._add_to_list(kernel_1, [5, 5, 3, k_channels])
        self._add_to_list(biases_1, [k_channels])

        kernel_2 = self._variable_with_weight_decay('conv2_weights', shape=[5, 5, k_channels, k_channels], stddev=5e-2, wd=0.0)
        biases_2 = self._variable_on_cpu('conv2_biases', [k_channels], tf.constant_initializer(0.1))
        self._add_to_list(kernel_2, [5, 5, k_channels, k_channels])
        self._add_to_list(biases_2, [k_channels])

        f1_weights = self._variable_with_weight_decay('f1_weights', shape=[f1_in, f1_out],
                                              stddev=0.04, wd=0.004)
        f1_biases = self._variable_on_cpu('f1_biases', [f1_out], tf.constant_initializer(0.1))
        self._add_to_list(f1_weights, [f1_in, f1_out])
        self._add_to_list(f1_biases, [f1_out])

        f2_weights = self._variable_with_weight_decay('f2_weights', shape=[f1_out, f2_out],
                                              stddev=0.04, wd=0.004)
        f2_biases = self._variable_on_cpu('f2_biases', [f2_out], tf.constant_initializer(0.1))
        self._add_to_list(f2_weights, [f1_out, f2_out])
        self._add_to_list(f2_biases, [f2_out])

        f3_weights = self._variable_with_weight_decay('f3_weights', shape=[f2_out, self.NUM_CLASSES],
                                                      stddev=1/f2_out, wd=0.0)
        f3_biases = self._variable_on_cpu('f3_biases', [self.NUM_CLASSES], tf.constant_initializer(0.0))
        self._add_to_list(f3_weights, [f2_out, self.NUM_CLASSES])
        self._add_to_list(f3_biases, [self.NUM_CLASSES])
        # else:
        #     self.w1 = self.create_variable('w1', dims=[5, 5, 3, 16])
        #     self.b1 = self.create_variable('b1', dims=[16])
        #     self.w2 = self.create_variable('w2', dims=[5, 5, 16, 16])
        #     self.b2 = self.create_variable('b2', dims=[16])
        #     self.w3 = self.create_variable('w3', dims=[5, 5, 16, 16])
        #     self.b3 = self.create_variable('b3', dims=[16])
        #     self.w4 = self.create_variable('w4', dims=[144, 32])
        #     self.b4 = self.create_variable('b4', dims=[32])
        #     self.w5 = self.create_variable('w5', dims=[32, 10])
        #     self.b5 = self.create_variable('b5', dims=[10])

        with tf.device('/cpu:0'):
            self.train_images_batch, self.train_labels_batch = self.distorted_inputs(path, batch_size=128)
            self.eval_images_batch, self.eval_labels_batch = self.inputs(False, path, batch_size=128)

    def network(self, batch, variables):
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(batch, variables[0], [1, 1, 1, 1], padding='SAME'), variables[1]))
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                       name='norm1')

        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm1, variables[2], [1, 1, 1, 1], padding='SAME'), variables[3]))
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                       name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        reshape = tf.reshape(pool2, [128, -1])
        dim = reshape.get_shape()[1].value
        local3 = tf.nn.relu(tf.matmul(reshape, variables[4]) + variables[5])
        local4 = tf.nn.relu(tf.matmul(local3, variables[6]) + variables[7])
        softmax_linear = tf.add(tf.matmul(local4, variables[8]), variables[9])
        return softmax_linear

    def loss(self, variables, mode='train'):
        """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
          logits: Logits from inference().
          labels: Labels from distorted_inputs or inputs(). 1-D tensor
                  of shape [batch_size]
        Returns:
          Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        if mode == 'train':
            images = self.train_images_batch
            labels = self.train_labels_batch
        else:
            images = self.eval_images_batch
            labels = self.eval_labels_batch

        logits = self.network(images, variables)
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).

        return tf.add_n(tf.get_collection('losses'), name='total_loss')

