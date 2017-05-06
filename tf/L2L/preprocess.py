import tensorflow as tf
import numpy as np
from abc import ABCMeta

class Preprocess():
    __metaclass__ = ABCMeta

    def process(self, gradients):
        pass


class LogAndSign(Preprocess):

    __p = None

    def __init__(self, k):
        self.__p = k

    def __clamp(self, gradients, min_value=None, max_value=None):
        if min_value is not None:
            gradients = tf.maximum(gradients, min_value)
        if max_value is not None:
            gradients = tf.minimum(gradients, max_value)
        return gradients

    def process(self, gradients):
        # eps = np.finfo(gradients.dtype.as_numpy_dtype).eps
        # cond1_indices = tf.squeeze(tf.slice(tf.where(tf.greater_equal(tf.abs(gradients), tf.exp(-self.__k))), [0, 0], [-1, 1]))
        # cond2_indices = tf.squeeze(tf.slice(tf.where(tf.less(tf.abs(gradients), tf.exp(-self.__k))), [0, 0], [-1, 1]))

        eps = np.finfo(gradients.dtype.as_numpy_dtype).eps
        ndims = gradients.get_shape().ndims
        log = tf.log(tf.abs(gradients) + eps)
        clamped_log = self.__clamp(log / self.__p, min_value=-1.0)
        sign = self.__clamp(gradients * np.exp(self.__p), min_value=-1.0, max_value=1.0)

        return tf.concat([clamped_log, sign], ndims - 1)


