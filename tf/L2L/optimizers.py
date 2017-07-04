from __future__ import print_function
from abc import ABCMeta
import tensorflow as tf
from tensorflow.python.util import nest
import pickle
from preprocess import Preprocess
from timeit import default_timer as timer

class Optimizer():

    __metaclass__ = ABCMeta
    problem = None
    global_args = None
    session = None

    ops_step = None
    ops_updates = None
    ops_loss = None

    def __init__(self, problem, args):
        self.problem = problem
        self.global_args = args

    def set_session(self, session):
        self.session = session

    def get_gradients(self, variables=None):
        variables = self.problem.variables if variables is None else variables
        return [gradient for gradient in self.problem.get_gradients(variables)]

    def loss(self, variables=None):
        variables = self.problem.variables if variables is None else variables
        return self.problem.loss(variables)

    def step(self):
        pass

    def updates(self, variables):
        pass

    def build(self):
        pass

class XhistorySign(Optimizer):

    limit = None
    def __init__(self, problem, args):
        super(XhistorySign, self).__init__(problem, args)
        self.limit = args['limit']
        with tf.name_scope('optimizer_input_init'):
            self.history_ptr = tf.Variable(0, 'history_ptr')
            self.guide_optimizer = tf.train.AdamOptimizer(.01, name='guide_optimizer')
            self.guide_step = self.guide_optimizer.minimize(self.problem.loss(self.problem.variables),
                                                            var_list=self.problem.variables, name='guide_step')
            self.variable_history = [tf.get_variable('variable_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, self.limit], trainable=False)
                                     for i, shape in enumerate(self.problem.variables_flattened_shape)]
            self.grad_sign_history = [tf.get_variable('gradients_sign_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, self.limit], trainable=False)
                                      for i, shape in enumerate(self.problem.variables_flattened_shape)]
            for i, variable in enumerate(self.variable_history):
                tf.summary.histogram('variable_history_' + str(i), variable)

    def init_with_session(self, args=None):
        with tf.name_scope('mlp_x_init_with_session'):
            for col in range(self.global_args['limit']):
                for variable_ptr, (variable, gradient) in enumerate(zip(self.problem.variables_flat, self.problem.get_gradients())):
                    update_ops = self.update_history_ops(variable_ptr, (variable, tf.sign(gradient)))
                    self.session.run(update_ops)
                if col < self.global_args['limit'] - 1:
                    self.session.run(self.guide_step)
                    self.session.run(tf.assign_add(self.history_ptr, 1))
            self.session.run(tf.assign(self.history_ptr, 0))

    @staticmethod
    def normalize_value(value, min=-1.0, max=1.0):
        diff = tf.subtract(max, min)
        return 2.0 * tf.divide((value - min), diff) - 1.0

    def step(self):
        x_next = list()
        deltas_list = []
        for i, (variable, variable_history, variable_grad_sign_history) in enumerate(zip(self.problem.variables,
                                                                                         self.variable_history,
                                                                                         self.grad_sign_history)):
            sum = tf.reduce_sum(variable_grad_sign_history, 1)
            sum = tf.reshape(sum, [tf.shape(sum)[0], 1])
            deltas = XhistorySign.normalize_value(sum, -self.limit * 1.0, self.limit * 1.0)
            deltas_list.append(deltas)
            max_values = tf.reduce_max(variable_history, 1)
            min_values = tf.reduce_min(variable_history, 1)
            max_values = tf.reshape(max_values, [tf.shape(max_values)[0], 1])
            min_values = tf.reshape(min_values, [tf.shape(min_values)[0], 1])
            diff = max_values - min_values
            ref_points = max_values + min_values
            new_points = tf.subtract(tf.divide(ref_points, 2.0), tf.multiply(deltas, diff), 'new_points')
            new_points = self.problem.set_shape(new_points, like_variable=variable, op_name='reshaped_new_points')
            x_next.append(new_points)
        return {'x_next': x_next, 'deltas': deltas_list}

    def update_history_ops(self, variable_ptr, inputs):
        variable, grad_sign = inputs
        history_ops = []
        shape = variable.shape[0].value
        indices = [[i, self.history_ptr] for i in range(shape)]
        history_ops.append(tf.scatter_nd_update(self.variable_history[variable_ptr], indices, tf.reshape(variable, [shape])))
        history_ops.append(tf.scatter_nd_update(self.grad_sign_history[variable_ptr], indices, tf.reshape(grad_sign, [shape])))
        return history_ops

    def updates(self, args):
        update_list = [tf.assign(variable, updated_var) for variable, updated_var in
                       zip(self.problem.variables, args['x_next'])]
        flat_gradients = self.problem.get_gradients(args['x_next'])
        flat_variables = [self.problem.flatten_input(i, variable) for i, variable in enumerate(args['x_next'])]
        for i, (variable, grads) in enumerate(zip(flat_variables, flat_gradients)):
            new_input = [variable, tf.sign(grads)]
            update_list.extend(self.update_history_ops(i, new_input))
        with tf.control_dependencies(update_list):
            update_itr = tf.cond(self.history_ptr < self.global_args['limit'] - 1,
                                 lambda: tf.assign_add(self.history_ptr, 1),
                                 lambda: tf.assign(self.history_ptr, 0))
        return update_list + [update_itr]

    def build(self):
        self.ops_step = self.step()
        self.ops_updates = self.updates({'x_next': self.ops_step['x_next']})
        self.ops_loss = self.loss(self.ops_step['x_next'])