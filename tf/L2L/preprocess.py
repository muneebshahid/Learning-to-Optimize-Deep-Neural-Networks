import tensorflow as tf
import numpy as np
from abc import ABCMeta

class Preprocess():

    def __init__(self):
        pass

    @staticmethod
    def clamp(inputs, args):
        min_value = args['min'] if 'min' in args else None
        max_value = args['max'] if 'max' in args else None
        outputs = inputs
        if min_value is not None:
            outputs = tf.maximum(outputs, min_value)
        if max_value is not None:
            outputs = tf.minimum(outputs, max_value)
        return outputs

    @staticmethod
    def sep_sign(inputs, args):
        return tf.concat([tf.abs(inputs), tf.sign(inputs)], 1)

    @staticmethod
    def log_sign(inputs, args):
        # eps = np.finfo(gradients.dtype.as_numpy_dtype).eps
        # cond1_indices = tf.squeeze(tf.slice(tf.where(tf.greater_equal(tf.abs(gradients), tf.exp(-self.__k))), [0, 0], [-1, 1]))
        # cond2_indices = tf.squeeze(tf.slice(tf.where(tf.less(tf.abs(gradients), tf.exp(-self.__k))), [0, 0], [-1, 1]))

        eps = np.finfo(inputs.dtype.as_numpy_dtype).eps
        ndims = inputs.get_shape().ndims
        log = tf.log(tf.abs(inputs) + eps)
        clamped_log = Preprocess.clamp(log / args['k'], args={'min': -1.0})
        sign = Preprocess.clamp(inputs * np.exp(args['k']), args={'min': -1.0, 'max': 1.0})
        return tf.concat([clamped_log, sign], ndims - 1)


