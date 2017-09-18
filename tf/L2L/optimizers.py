from __future__ import print_function
from abc import ABCMeta
import tensorflow as tf
from tensorflow.python.util import nest
import pickle
from preprocess import Preprocess
from timeit import default_timer as timer
import numpy as np

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

    def step(self, args=None):
        pass

    def updates(self, args=None):
        pass

    def build(self, args=None):
        pass

class Adam(Optimizer):

    ms = None
    vs = None
    beta_1 = None
    beta_2 = None
    t = None
    lr = None
    eps = None
    optim_params = None
    learn_betas = None
    def __init__(self, problem, args=None):
        super(Adam, self).__init__(problem, args)
        self.learn_betas = args['learn_betas']
        if self.learn_betas:
            self.beta_1 = [tf.Variable(beta_1) for beta_1 in args['beta_1']]
            self.beta_2 = [tf.Variable(beta_2) for beta_2 in args['beta_2']]
        else:
            beta_1_var = tf.Variable(args['beta_1'])
            beta_2_var = tf.Variable(args['beta_2'])
            self.beta_1 = [beta_1_var for variable in problem.variables]
            self.beta_2 = [beta_2_var for variable in problem.variables]
        self.lr = args['lr']
        self.eps = args['eps']
        self.eps_squared = tf.square(self.eps)
        self.t = tf.Variable(1.0)
        self.ms = [tf.Variable(tf.zeros([shape, 1])) for shape in self.problem.variables_flattened_shape]
        self.vs = [tf.Variable(tf.zeros([shape, 1])) for shape in self.problem.variables_flattened_shape]
        self.optim_params = [self.ms, self.vs]
        if self.learn_betas:
            self.optim_params.extend([self.beta_1, self.beta_2])

    def set_variable(self, variable_key, args, default):
        if args is not None and variable_key in args:
            return args[variable_key]
        else:
            return default

    def step(self, args=None):
        vars_next = []
        vars_steps = []
        ms_next = []
        vs_next = []
        problem_variables = self.set_variable('variables', args, self.problem.variables)
        problem_variables_flat = self.set_variable('variables_flat', args, self.problem.variables_flat)
        problem_gradients = self.set_variable('gradients', args, self.get_gradients(self.problem.variables))
        optim_params = self.set_variable('optim_params', args, self.optim_params)
        problem_ms = optim_params[0]
        problem_vs = optim_params[1]
        if self.learn_betas:
            betas_1 = optim_params[2]
            betas_2 = optim_params[3]
        else:
            betas_1 = self.beta_1
            betas_2 = self.beta_2

        for var, var_flat, gradient, var_m, var_v, beta_1, beta_2 in zip(problem_variables, problem_variables_flat, problem_gradients,
                                                                          problem_ms, problem_vs, betas_1, betas_2):
            m = beta_1 * var_m + (1.0 - beta_1) * gradient
            v = beta_2 * var_v + (1.0 - beta_2) * tf.square(gradient)
            # v = tf.maximum(beta_2 * var_v, tf.abs(gradient))
            ms_next.append(m)
            vs_next.append(v)
            m_hat = m / (1 - tf.pow(beta_1, self.t))
            v_hat = v / (1 - tf.pow(beta_2, self.t))
            var_step = -self.lr * m_hat / (tf.sqrt(v_hat + self.eps_squared))
            # var_step = -self.lr * m_hat / v
            var_next = var_flat + var_step
            var_next = self.problem.set_shape(var_next, like_variable=var, op_name='reshape_variable')
            vars_steps.append(var_step)
            vars_next.append(var_next)
        return {'vars_next': vars_next, 'vars_steps': vars_steps, 'optim_params_next': [ms_next, vs_next]}

    def updates(self, args=None):
        vars_next = args['vars_next'] if 'vars_next' in args else None
        ms_next = args['optim_params_next'][0]
        vs_next = args['optim_params_next'][1]
        if vars_next is None:
            updates_list = []
        else:
            updates_list = [tf.assign(variable, variable_next) for variable, variable_next in zip(self.problem.variables, vars_next)]
        updates_list.append([tf.assign(m, m_next) for m, m_next in zip(self.ms, ms_next)])
        updates_list.append([tf.assign(v, v_next) for v, v_next in zip(self.vs, vs_next)])
        updates_list.append(tf.assign_add(self.t, 1.0))
        if self.learn_betas:
            betas_1_next = args['optim_params_next'][2]
            betas_2_next = args['optim_params_next'][3]
            updates_list.append([tf.assign(beta_1, beta_1_next) for beta_1, beta_1_next in zip(self.beta_1, betas_1_next)])
            updates_list.append([tf.assign(beta_2, beta_2_next) for beta_2, beta_2_next in zip(self.beta_2, betas_2_next)])
        return updates_list

    def build(self, args=None):
        self.ops_step = self.step(args)
        self.ops_loss = self.problem.loss(self.ops_step['vars_next'])
        self.ops_updates = self.updates(self.ops_step)

class XHistoryGradNorm(Optimizer):


    variable_history = None
    grad_history = None
    history_ptr = None
    update_window = None
    guide_optimizer = None
    guide_step = None
    ops_init = None

    def __init__(self, problem, args):
        super(XHistoryGradNorm, self).__init__(problem, args)
        self.limit = args['limit']
        with tf.name_scope('optimizer_input_init'):
            self.history_ptr = tf.Variable(0, 'history_ptr')
            self.guide_optimizer = tf.train.AdamOptimizer(.01, name='guide_optimizer')
            self.guide_step = self.guide_optimizer.minimize(self.problem.loss(self.problem.variables),
                                                            var_list=self.problem.variables, name='guide_step')
            self.variable_history = [tf.get_variable('variable_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, self.limit], trainable=False)
                                     for i, shape in enumerate(self.problem.variables_flattened_shape)]
            self.grad_history = [tf.get_variable('gradients_sign_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, self.limit], trainable=False)
                                 for i, shape in enumerate(self.problem.variables_flattened_shape)]

    def run_init(self, args=None):
        with tf.name_scope('mlp_x_init_with_session'):
            for col in range(self.global_args['limit']):
                self.session.run(self.ops_init)
                if col < self.global_args['limit'] - 1:
                    self.session.run(self.guide_step)

    @staticmethod
    def normalize_values(history_tensor, switch=0):
        with tf.name_scope('mlp_x_normalize_variable_history'):
            if switch == 0:
                norm = tf.norm(history_tensor, ord=np.inf, axis=1, keep_dims=True)
                ones = tf.ones(tf.shape(norm))
                divisor = tf.where(tf.equal(norm, 0.0), ones, norm)
                normalized_values = tf.divide(history_tensor, divisor)
            else:
                max_values = tf.reduce_max(history_tensor, 1)
                min_values = tf.reduce_min(history_tensor, 1)
                max_values = tf.reshape(max_values, [tf.shape(max_values)[0], 1])
                min_values = tf.reshape(min_values, [tf.shape(min_values)[0], 1])
                diff = max_values - min_values
                normalized_values = 2 * (history_tensor - min_values) / diff - 1.0
            return normalized_values

    def sort_input(self, args):
        with tf.name_scope('mlp_x_sort_input'):
            inputs = args['inputs']
            history_ptr = args['history_ptr']
            read_ptr = history_ptr + 1
            start = tf.slice(inputs, [0, 0], [-1, read_ptr], name='start')
            end = tf.slice(inputs, [0, read_ptr], [-1, self.limit - read_ptr], name='start')
            rev_start = tf.reverse(start, [1])
            rev_end = tf.reverse(end, [1])
            return tf.concat([rev_start, rev_end], 1, name='sorted_input')

    def step(self, args=None):
        x_next = list()
        deltas_list = []
        problem_variables = args['x_next']
        variables_history = args['variable_history']
        grads_history = args['grad_history']
        for i, (variable, variable_history, variable_grad_history) in enumerate(zip(problem_variables,
                                                                                         variables_history,
                                                                                         grads_history)):
            normalized_grad_history = XHistoryGradNorm.normalize_values(variable_grad_history)
            deltas = tf.reduce_mean(normalized_grad_history, 1)
            deltas = tf.expand_dims(deltas, 1)
            deltas_list.append(deltas)
            max_values = tf.expand_dims(tf.reduce_max(variable_history, 1), 1)
            min_values = tf.expand_dims(tf.reduce_min(variable_history, 1), 1)
            diff = max_values - min_values
            ref_points = tf.divide(max_values + min_values, 2.0)
            noise = tf.random_normal([ref_points.shape[0].value, 1], 0, .01)
            mean = tf.multiply(deltas, diff)
            noisey_mean = mean * (1 + noise)
            new_points = tf.subtract(ref_points, noisey_mean, 'new_points')
            new_points = self.problem.set_shape(new_points, like_variable=variable, op_name='reshaped_new_points')
            x_next.append(new_points)
        return {'x_next': x_next, 'deltas': deltas_list}

    def update_history_ops(self, batch_variables, batch_gradients_sign, batch_variables_history, batch_grad_sign_history, history_ptr):
        history_ops = []
        shape = batch_variables.shape[0].value
        indices = [[i, history_ptr] for i in range(shape)]
        history_ops.append(tf.scatter_nd_update(batch_variables_history, indices, tf.reshape(batch_variables, [shape])))
        history_ops.append(tf.scatter_nd_update(batch_grad_sign_history, indices, tf.reshape(batch_gradients_sign, [shape])))
        return history_ops

    def updates(self, args=None):
        with tf.name_scope('mlp_x_optimizer_updates'):
            x_next = args['x_next']
            variable_history = args['variable_history']
            grad_history = args['grad_history']
            history_ptr = args['history_ptr']
            update_list = [tf.cond(history_ptr < self.global_args['limit'] - 1,
                                lambda: tf.assign_add(history_ptr, 1),
                                lambda: tf.assign(history_ptr, 0))]
            with tf.control_dependencies(update_list):
                if not args['init_ops']:
                    update_list.extend([tf.assign(variable, updated_var) for variable, updated_var in
                                        zip(self.problem.variables, args['x_next'])])
                flat_gradients = self.problem.get_gradients(x_next)
                flat_variables = [self.problem.flatten_input(i, variable) for i, variable in enumerate(x_next)]
                for variable, grads, batch_variable_history, batch_grad_history in zip(flat_variables, flat_gradients, variable_history, grad_history):
                    update_list.extend(self.update_history_ops(variable, grads, batch_variable_history, batch_grad_history, history_ptr))
            return update_list

    def build(self):
        args = {'x_next': [variable.initialized_value() for variable in self.problem.variables],
                'variable_history': self.variable_history, 'grad_history': self.grad_history,
                'history_ptr': self.history_ptr, 'init_ops': True}
        self.ops_init = self.updates(args)
        step = self.step(args)
        args['x_next'] = step['x_next']
        args['init_ops'] = False
        updates = self.updates(args)
        self.ops_step = step
        self.ops_updates = updates
        self.ops_loss = self.problem.loss(step['x_next'])

class XHistorySign(XHistoryGradNorm):

    def step(self, args=None):
        args_xhistory_sign = dict(args)
        args_xhistory_sign['grad_history'] = [tf.sign(grad) for grad in args_xhistory_sign['grad_history']]
        return super(XHistorySign, self).step(args_xhistory_sign)


class XSign(Optimizer):
    limit = None
    beta = None
    def __init__(self, problem, args):
        super(XSign, self).__init__(problem, args)
        with tf.name_scope('optimizer_input_init'):
            self.beta = tf.get_variable('beta', initializer=tf.constant_initializer(args['beta']), shape=[1, 1])
            self.guide_optimizer = tf.train.AdamOptimizer(.01, name='guide_optimizer')
            self.guide_step = self.guide_optimizer.minimize(self.problem.loss(self.problem.variables),
                                                            var_list=self.problem.variables, name='guide_step')
            self.variable_avg = [tf.get_variable('variable_avg' + str(i), initializer=tf.zeros_initializer, shape=[shape, 1])
                                 for i, shape in enumerate(self.problem.variables_flattened_shape)]
            self.sign_avg = [tf.get_variable('sign_avg' + str(i), initializer=tf.zeros_initializer, shape=[shape, 1])
                             for i, shape in enumerate(self.problem.variables_flattened_shape)]

    def init_with_session(self, args=None):
        for itr in range(5):
            update_ops = self.update_avg_ops([self.problem.variables_flat, self.problem.get_gradients()])
            self.session.run(update_ops)
            self.session.run(self.guide_step)

    def step(self):
        x_next = list()
        deltas_list = []
        for i, (variable_flat, variable_avg, sign_avg) in enumerate(zip(self.problem.variables_flat, self.variable_avg,
                                                                   self.sign_avg)):
            ref_points = (variable_avg + variable_flat) / 2.0
            diff = tf.abs(variable_avg - variable_flat)
            mean = tf.subtract(ref_points, tf.multiply(sign_avg, diff))
            noise = tf.random_normal([mean.shape[0].value, 1], 0, .01)
            noisey_mean = mean * (1 + noise)
            new_points = tf.subtract(ref_points, noisey_mean, 'new_points')
            deltas_list.append(sign_avg)
            new_points = self.problem.set_shape(new_points, i=i, op_name='reshaped_new_points')
            x_next.append(new_points)
        return {'x_next': x_next, 'deltas': deltas_list}

    def update_avg_ops(self, inputs):
        variables, gradients = inputs
        updates_list = [tf.assign(variable_avg, variable_avg * self.beta + variable * (1.0 - self.beta))
                        for variable, variable_avg in zip(variables, self.variable_avg)]
        updates_list.extend([tf.assign(sign_avg, sign_avg * self.beta + tf.sign(gradient) * (1.0 - self.beta))
                             for gradient, sign_avg in zip(gradients, self.sign_avg)])
        return updates_list

    def updates(self, args):
        update_list = [tf.assign(variable, updated_var) for variable, updated_var in
                       zip(self.problem.variables, args['x_next'])]
        flat_gradients = self.problem.get_gradients(args['x_next'])
        flat_variables = [self.problem.flatten_input(i, variable) for i, variable in enumerate(args['x_next'])]
        update_list.extend(self.update_avg_ops([flat_variables, flat_gradients]))
        return update_list

    def build(self):
        self.ops_step = self.step()
        self.ops_updates = self.updates({'x_next': self.ops_step['x_next']})
        self.ops_loss = self.loss(self.ops_step['x_next'])

