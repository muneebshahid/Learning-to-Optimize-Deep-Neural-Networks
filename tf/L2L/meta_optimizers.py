from __future__ import print_function
from abc import ABCMeta
import tensorflow as tf
from tensorflow.python.util import nest
import pickle
from preprocess import Preprocess
from timeit import default_timer as timer



class Meta_Optimizer():
    __metaclass__ = ABCMeta

    global_args = None
    io_handle = None
    problem = None
    meta_optimizer_optimizer = None
    preprocessor = None
    preprocessor_args = None
    optimizer_variables = None
    session = None

    ops_step = None
    ops_updates = None
    ops_loss = None
    ops_meta_step = None
    ops_reset = None
    allow_reset = False

    def __init__(self, problem, path, args):
        if path is not None:
            print('Loading optimizer args, ignoring provided args...')
            self.global_args = self.load_args(path)
            print('Args Loaded, call load_optimizer with session to restore the optimizer graph.')
        else:
            self.global_args = args
        self.problem = problem
        if 'preprocess' in self.global_args and self.global_args['preprocess'] is not None:
            self.preprocessor = self.global_args['preprocess'][0]
            self.preprocessor_args = self.global_args['preprocess'][1]
        self.learning_rate = tf.get_variable('learning_rate', initializer=tf.constant(self.global_args['learning_rate']
                                                                                      if 'learning_rate' in self.global_args
                                                                                      else .0001,
                                                                                      dtype=tf.float32), trainable=False)
        self.meta_optimizer_optimizer = tf.train.AdamOptimizer(self.global_args['meta_learning_rate'] if 'meta_learning_rate' in
                                                                                                         self.global_args else .01, name='meta_optimizer_optimizer')
        self.optimizer_variables = []

    def __init_trainable_vars_list(self):
        return
        # self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def __init_io_handle(self):
        self.io_handle = tf.train.Saver([variable for layer in self.optimizer_variables for variable in layer], max_to_keep=100)

    def end_init(self):
        return
        self.__init_trainable_vars_list()
        self.__init_io_handle()

    def preprocess_input(self, inputs):
        if self.preprocessor is not None:
            return self.preprocessor(inputs, self.preprocessor_args)
        else:
            return inputs

    def is_availble(self, param, args=None):
        args = self.global_args if args is None else args
        return param in args and args[param] is not None

    def get_preprocessed_gradients(self, variables=None):
        return [self.preprocess_input(gradient) for gradient in self.problem.get_gradients(variables)]

    @property
    def meta_optimizer_input_stack(self):
        variables = self.problem.variables_flat
        gradients_raw = self.problem.get_gradients_raw(variables)
        flat_gradients = [self.problem.flatten_input(i, gradient) for i, gradient in enumerate(gradients_raw)]
        preprocessed_gradients = [self.preprocess_input(gradient) for gradient in flat_gradients]
        stacked_inputs = [{
            'x': variable,
            'gradient_raw': gradients_raw,
            'flat_gradient': flat_gradient,
            'preprocessed_gradient': preprocessed_gradient
        }
            for (variable, gradient_raw, flat_gradient, preprocessed_gradient) in
            zip(variables, gradients_raw, flat_gradients, preprocessed_gradients)]
        return stacked_inputs

    def updates(self, args):
        pass

    def network(self, inputs):
        pass

    def loss(self, variables=None):
        pass

    def step(self):
        pass

    def minimize(self, loss):
        return self.meta_optimizer_optimizer.minimize(loss)

    def build(self):
        pass

    def reset_optimizer(self):
        return [tf.variables_initializer(variable) for variable in self.optimizer_variables]

    def reset_problem(self):
        return [tf.variables_initializer(self.problem.variables + self.problem.constants)]

    @staticmethod
    def load_args(path):
        pickled_args = pickle.load(open(path + '_config.p', 'rb'))
        pickled_args['preprocess'][0] = getattr(Preprocess, pickled_args['preprocess'][0])
        return pickled_args

    def save_args(self, path):
        self.global_args['preprocess'][0] = self.global_args['preprocess'][0].func_name
        pickle.dump(self.global_args, open(path + '_config.p', 'wb'))
        self.global_args['preprocess'][0] = getattr(Preprocess, self.global_args['preprocess'][0])

    def load(self, sess, path):
        self.io_handle.restore(sess, path)
        print('Optimizer Restored')

    def save(self, sess, path):
        print('Saving optimizer')
        self.io_handle.save(sess, path)
        self.save_args(path)

    def init_with_session(self, args=None):
        return

    def set_session(self, session):
        self.session = session

    def run(self, args=None):
        num_steps = 1 if 'num_steps' not in args else args['num_steps']
        if self.allow_reset and self.ops_reset is not None:
            self.session.run(self.ops_reset)
        loss = 0
        start = timer()
        for _ in range(num_steps):
            loss += self.session.run([self.ops_loss, self.ops_meta_step, self.ops_updates])[0]
        return timer() - start, loss / num_steps

class l2l(Meta_Optimizer):

    state_size = None
    unroll_len = None
    optim_per_epoch = None
    W, b = None, None
    lstm = None
    fx_array = None

    @property
    def meta_optimizer_input_stack(self):
        inputs = super(l2l, self).meta_optimizer_input_stack
        for (input, hidden_state) in zip(inputs, self.hidden_states):
            input['hidden_state'] = hidden_state
        return inputs

    def __init__(self, problem, path, args):
        super(l2l, self).__init__(problem, path, args)
        self.state_size = self.global_args['state_size']
        self.num_layers = self.global_args['num_layers']
        self.unroll_len = self.global_args['unroll_len']
        self.num_step = self.global_args['optim_per_epoch'] // self.unroll_len
        self.meta_optimizer = tf.train.AdamOptimizer(self.global_args['meta_learning_rate'])
        self.fx_array = tf.TensorArray(tf.float32, size=self.unroll_len, clear_after_read=False)

        # initialize for later use.
        with tf.variable_scope('optimizer_core'):
            # Formulate variables for all states as it allows to use tf.assign() for states
            def get_states(batch_size):
                state_variable = []
                for state_c, state_h in self.lstm.zero_state(batch_size, tf.float32):
                    state_variable.append(tf.contrib.rnn.LSTMStateTuple(tf.Variable(state_c, trainable=False),
                                                                        tf.Variable(state_h, trainable=False)))
                return tuple(state_variable)

            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.state_size)
            self.lstm = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.BasicLSTMCell(self.state_size) for _ in range(self.num_layers)])
            gradients = self.preprocess_input(self.problem.get_gradients()[0])

            with tf.variable_scope('hidden_states'):
                self.hidden_states = [get_states(self.problem.get_shape(variable=variable)) for variable in self.problem.variables_flat]

            with tf.variable_scope('rnn_init'):
                self.lstm(gradients, self.hidden_states[0])

            with tf.variable_scope('rnn_linear'):
                self.W = tf.get_variable('softmax_w', [self.state_size, 1])
                self.b = tf.get_variable('softmax_b', [1])

        self.end_init()

    def network(self, inputs):
        with tf.variable_scope('optimizer_core/rnn_init', reuse=True):
            lstm_output, hidden_state = self.lstm(inputs['preprocessed_gradient'], inputs['hidden_state'])
        deltas = tf.add(tf.matmul(lstm_output, self.W, name='output_matmul'), self.b, name='add_bias')
        return [deltas, hidden_state]

    def step(self):
        def update(t, fx_array, params, hidden_states):
            rnn_inputs = self.get_preprocessed_gradients(params)
            for i, (rnn_input, hidden_state) in enumerate(zip(rnn_inputs, hidden_states)):
                deltas, hidden_states[i] = self.network({'preprocessed_gradient': rnn_input, 'hidden_state': hidden_state})
                # overwrite each iteration of the while loop, so you will end up with the last update
                # deltas_list[i] = deltas
                deltas = self.problem.set_shape(deltas, i, op_name='reshape_deltas')
                deltas = tf.multiply(deltas, self.learning_rate, 'multiply_deltas')
                params[i] = tf.add(params[i], deltas, 'add_deltas_params')
            fx_array = fx_array.write(t, self.problem.loss(params))
            t_next = t + 1
            return t_next, fx_array, params, hidden_states

        deltas_list = list(range(len(self.hidden_states)))

        _, self.fx_array, x_next, h_next = tf.while_loop(
            cond=lambda t, *_: t < self.unroll_len,
            body=update,
            loop_vars=([0, self.fx_array, self.problem.variables, self.hidden_states]),
            parallel_iterations=1,
            swap_memory=True,
            name="unroll")

        return {'x_next': x_next, 'h_next': h_next, 'deltas': deltas_list}

    def updates(self, args):
        update_list = list()
        update_list.append([tf.assign(variable, variable_final) for variable, variable_final in
                              zip(self.problem.variables, args['x_next'])])
        update_list.append([tf.assign(hidden_state, hidden_state_final) for hidden_state, hidden_state_final in
                              zip(nest.flatten(self.hidden_states), nest.flatten(args['h_next']))])
        return update_list

    def reset_problem(self):
        reset = super(l2l, self).reset_problem()
        reset.append(nest.flatten(self.hidden_states))
        reset.append(self.fx_array.close())
        return reset

    def loss(self, variables=None):
        return tf.divide(tf.reduce_sum(self.fx_array.stack()), self.unroll_len)

    def build(self):
        step = self.step()
        updates = self.updates(step)
        loss = self.loss(step['x_next'])
        meta_step = self.minimize(loss)
        reset = [self.reset_problem(), self.reset_optimizer()]
        self.ops_step = step
        self.ops_updates = updates
        self.ops_loss = loss
        self.ops_meta_step = meta_step
        self.ops_reset = reset

    def run(self, args=None):
        return super(l2l, self).run({'num_steps': self.num_step})

class MlpSimple(Meta_Optimizer):

    w_1, b_1, w_out, b_out = None, None, None, None
    layer_width = None
    hidden_layers = None

    def layer_fc(self, name, dims, inputs, initializers=None, activation=tf.nn.softplus, reuse=False):
        initializers = [tf.random_normal_initializer(mean=0.0, stddev=.1), tf.zeros_initializer] \
            if initializers is None else initializers
        with tf.name_scope('optimizer_fc_layer_' + name):
            with tf.variable_scope('optimizer_network', reuse=reuse):
                w = tf.get_variable('w_' + name, shape=dims, initializer=initializers[0])
                b = tf.get_variable('b_' + name, shape=[1, dims[-1]], initializer=initializers[1])
                linear = tf.add(tf.matmul(inputs, w), b, name='activations_' + 'layer_' + str(name))
                layer_output = linear if activation is None else activation(linear)
                # tf.summary.histogram('weights', w)
                # tf.summary.histogram('bias', b)
                if not reuse:
                    self.optimizer_variables.append([w, b])
        return layer_output

    def __init__(self, problem, path, args):
        if 'dims' not in args:
            input_dim, output_dim = (1, 1)
            if 'preprocess' in args and args['preprocess'] is not None:
                input_dim = 2
            args['dims'] = (input_dim, output_dim)
        super(MlpSimple, self).__init__(problem, path, args)
        self.num_layers = 2
        self.layer_width = self.global_args['layer_width'] if self.is_availble('layer_width') else 20
        self.end_init()

    def network(self, inputs):
        reuse = inputs['reuse']
        input_dim, output_dim = self.global_args['dims']
        layer_activations = inputs['preprocessed_gradient']
        layer_activations = self.layer_fc(name='in', dims=[input_dim, self.layer_width], inputs=layer_activations, reuse=reuse)
        if self.is_availble('hidden_layers') and self.global_args['hidden_layers']:
            for layer in range(self.global_args['hidden_layers']):
                layer_activations = self.layer_fc(str(layer + 1), dims=[self.layer_width, self.layer_width], inputs=layer_activations, reuse=reuse)
        output = self.layer_fc('out', dims=[self.layer_width, output_dim], inputs=layer_activations, activation=None, reuse=reuse)
        return [output]

    def step(self):
        with tf.name_scope('mlp_simple_optimizer_step'):
            x_next = list()
            deltas_list = []
            preprocessed_gradients = self.get_preprocessed_gradients()
            optimizer_inputs = preprocessed_gradients
            for i, (variable, optim_input) in enumerate(zip(self.problem.variables, optimizer_inputs)):
                deltas = self.network({'preprocessed_gradient': optim_input, 'reuse': i > 0})[0]
                deltas_list.append(deltas)
                deltas = tf.multiply(deltas, self.learning_rate, name='apply_learning_rate')
                deltas = self.problem.set_shape(deltas, like_variable=variable, op_name='reshape_deltas')
                x_next.append(tf.add(variable, deltas))
            return {'x_next': x_next, 'deltas': deltas_list}

    def updates(self, args):
        with tf.name_scope('mlp_simple_optimizer_updates'):
            update_list = [tf.assign(variable, updated_var) for variable, updated_var in zip(self.problem.variables, args['x_next'])]
            return update_list

    def loss(self, variables=None):
        with tf.name_scope('mlp_simple_optimizer_loss'):
            variables = variables if variables is not None else self.problem.variables
            return self.problem.loss(variables)

    def build(self):
        step = self.step()
        updates = self.updates(step)
        loss = self.loss(step['x_next'])
        meta_step = self.minimize(loss)
        reset = [self.reset_problem(), self.reset_optimizer()]
        self.ops_step = step
        self.ops_updates = updates
        self.ops_loss = loss
        self.ops_meta_step = meta_step
        self.ops_reset = reset

    def run(self, args=None):
        return super(MlpSimple, self).run({'num_step': 1})


class MlpMovingAverage(MlpSimple):

    avg_gradients = None
    def __init__(self, problem, path, args):
        args['dims'] = (4, 1) if self.is_availble('preprocess', args) else (2, 1)
        super(MlpMovingAverage, self).__init__(problem, path, args)
        self.avg_gradients = [
            tf.get_variable('avg_gradients_' + str(i), shape=[shape, 1], initializer=tf.zeros_initializer(),
                            trainable=False)
            for i, shape in enumerate(self.problem.variables_flattened_shape)]

    def step(self):
        x_next = list()
        deltas_list = []
        preprocessed_gradients = self.get_preprocessed_gradients()
        optimizer_inputs = [tf.concat([gradient, self.preprocess_input(avg_gradient)], 1)
                            for gradient, avg_gradient in zip(preprocessed_gradients, self.avg_gradients)]
        for i, (variable, optim_input) in enumerate(zip(self.problem.variables, optimizer_inputs)):
            deltas = self.network({'preprocessed_gradient': optim_input})[0]
            deltas_list.append(deltas)
            deltas = tf.multiply(deltas, self.learning_rate, name='apply_learning_rate')
            deltas = self.problem.set_shape(deltas, like_variable=variable, op_name='reshape_deltas')
            x_next.append(tf.add(variable, deltas))
        return {'x_next': x_next, 'deltas': deltas_list}

    def updates(self, args):
        update_list = super(MlpMovingAverage, self).updates(args)
        gradients = self.get_preprocessed_gradients(args['x_next'])
        if self.preprocessor is None:
            update_list.extend([tf.assign(avg_gradient, avg_gradient * .9 + .1 * gradient)
                                for gradient, avg_gradient in zip(gradients, self.avg_gradients)])
        else:
            for gradient, avg_gradient in zip(gradients, self.avg_gradients):
                mag_indices = [[row, 0] for row in range(gradient.get_shape()[0].value)]
                mag_updates = tf.slice(gradient, [0, 0], [-1, 1])
                sign_indices = [[row, 1] for row in range(gradient.get_shape()[0].value)]
                sign_updates = tf.slice(gradient, [0, 1], [-1, 1])
                tf.scatter_nd_update(avg_gradient, mag_indices, tf.squeeze(mag_updates))
                tf.scatter_nd_update(avg_gradient, sign_indices, tf.squeeze(sign_updates))
        return update_list

    def reset_optimizer(self):
        reset = super(MlpMovingAverage, self).reset_optimizer()
        reset.append(tf.variables_initializer(self.avg_gradients))
        return reset

    def reset_problem(self):
        reset = super(MlpMovingAverage, self).reset_problem()
        reset.append(tf.variables_initializer(self.avg_gradients))
        return reset

# class MlpGradHistory(MlpSimple):
#
#     gradient_history = None
#     gradient_history_ptr = None
#     adam_problem = None
#     adam_problem_step = None
#
#     def __init__(self, problem, path, args):
#         limit = args['limit']
#         args['dims'] = (limit * 2, 1) if self.is_availble('preprocess', args) else (limit, 1)
#         super(MlpGradHistory, self).__init__(problem, path, args)
#         self.gradient_history_ptr = tf.Variable(0, 'gradient_history_ptr')
#         self.adam_problem = tf.train.AdamOptimizer(.01)
#         self.adam_problem_step = self.adam_problem.minimize(self.problem.loss(self.problem.variables), var_list=self.problem.variables)
#         self.gradient_history = [tf.get_variable('gradients_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, 4], trainable=False)
#                                  for i, shape in enumerate(self.problem.variables_flattened_shape)]
#
#     def init_with_session(self, args=None):
#         for
#         return
#
#     def get_gradient_history(self):
#         if self.gradient_history is None:
#             gradient_history_tensor = [None for _ in self.problem.variables_flat]
#             for history_itr in range(self.global_args['limit']):
#                 initialized_values = [variable.initialized_value() for variable in self.problem.variables]
#                 gradients = self.get_preprocessed_gradients(initialized_values)
#                 for i, gradient in enumerate(gradients):
#                     if gradient_history_tensor[i] is None:
#                         gradient_history_tensor[i] = gradient
#                     else:
#                         gradient_history_tensor[i] = tf.concat([gradient_history_tensor[i], gradient], axis=1)
#             self.gradient_history = [tf.get_variable('gradients_history' + str(i), initializer=gradient_tensor, trainable=False)
#                                     for i, gradient_tensor in enumerate(gradient_history_tensor)]
#         return self.gradient_history
#
#     def core(self, inputs):
#         gradients = inputs['preprocessed_gradient']
#         cols = 2 if self.is_availble('preprocess') else 1
#         start_ptr = tf.multiply(self.gradient_history_ptr, cols)
#         start = tf.slice(gradients, [0, start_ptr], [-1, -1])
#         end = tf.slice(gradients, [0, 0], [-1, start_ptr])
#         final_input = tf.concat([start, end], 1, 'final_input')
#         activations = tf.nn.softplus(tf.add(tf.matmul(final_input, self.w_1), self.b_1))
#         if self.hidden_layers is not None:
#             for i, layer in enumerate(self.hidden_layers):
#                 activations = tf.nn.softplus(tf.add(tf.matmul(activations, layer[0]), layer[1]), name='layer_' + str(i))
#         output = tf.add(tf.matmul(activations, self.w_out), self.b_out, name='layer_final_activation')
#         return [output]
#
#     def step(self):
#         x_next = list()
#         deltas_list = []
#         for i, (variable, variable_gradient_history) in enumerate(zip(self.problem.variables, self.get_gradient_history())):
#             deltas = self.core({'preprocessed_gradient': variable_gradient_history})[0]
#             deltas_list.append(deltas)
#             deltas = tf.multiply(deltas, self.learning_rate, name='apply_learning_rate')
#             deltas = self.problem.set_shape(deltas, like_variable=variable, op_name='reshape_deltas')
#             x_next.append(tf.add(variable, deltas))
#         return {'x_next': x_next, 'deltas': deltas_list}
#
#     def update_gradient_history_ops(self, variable_ptr, gradients):
#         cols = 1
#         rows = gradients.shape[0].value
#         if len(gradients.shape) > 1:
#             cols = gradients.shape[1].value
#         write_ptr = tf.multiply(self.gradient_history_ptr, cols)
#         indices = []
#         for col in range(cols):
#             for row in range(rows):
#                 indices.append([row, write_ptr + col])
#         stacked_grads = tf.slice(gradients, [0, 0], [-1, 1])
#         for col in range(cols)[1:]:
#             stacked_grads = tf.concat([stacked_grads, tf.slice(gradients, [0, col], [-1, 1])], 0)
#         return tf.scatter_nd_update(self.gradient_history[variable_ptr], indices, tf.squeeze(stacked_grads))
#
#     def updates(self, args):
#         update_list = super(MlpGradHistory, self).updates(args)
#         gradients = self.get_preprocessed_gradients(args['x_next'])
#         for i, gradient in enumerate(gradients):
#             update_list.append(self.update_gradient_history_ops(i, gradient))
#         with tf.control_dependencies(update_list):
#             update_itr = tf.cond(self.gradient_history_ptr < self.global_args['limit'] - 1,
#                             lambda: tf.assign_add(self.gradient_history_ptr, 1),
#                             lambda: tf.assign(self.gradient_history_ptr, 0))
#         return update_list + [update_itr]
#
#     def reset_optimizer(self):
#         reset = super(MlpGradHistory, self).reset_optimizer()
#         reset.append(tf.variables_initializer(self.gradient_history))
#         return reset
#
#     def reset_problem(self):
#         reset = super(MlpGradHistory, self).reset_problem()
#         reset.append(tf.variables_initializer(self.gradient_history))
#         return reset

class MlpGradHistoryFAST(MlpSimple):
    gradient_history = None
    gradient_sign_history = None
    gradient_history_ptr = None
    history = None
    guide_optimizer = None
    adam_problem_step = None

    def __init__(self, problem, path, args):
        limit = args['limit']
        args['dims'] = (limit * 2, 1) if self.is_availble('preprocess', args) else (limit, 1)
        super(MlpGradHistoryFAST, self).__init__(problem, path, args)
        with tf.name_scope('optimizer_network'):
            self.gradient_history_ptr = tf.Variable(0, 'gradient_history_ptr')
            self.guide_optimizer = tf.train.AdamOptimizer(.01)
            self.adam_problem_step = self.guide_optimizer.minimize(self.problem.loss(self.problem.variables), var_list=self.problem.variables)
            self.gradient_history = [tf.get_variable('gradients_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, args['limit']], trainable=False)
                                     for i, shape in enumerate(self.problem.variables_flattened_shape)]
            if self.is_availble('preprocess'):
                self.gradient_sign_history = [tf.get_variable('gradients_sign_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, args['limit']], trainable=False)
                                          for i, shape in enumerate(self.problem.variables_flattened_shape)]

    def init_with_session(self, args=None):
        for col in range(4):
            for variable_ptr, gradient in enumerate(self.problem.get_gradients()):
                indices = [[row, col] for row in range(gradient.get_shape()[0].value)]
                update_ops = [tf.scatter_nd_update(self.gradient_history[variable_ptr], indices, tf.squeeze(gradient))]
                if self.is_availble('preprocess'):
                    update_ops.append(tf.scatter_nd_update(self.gradient_sign_history[variable_ptr], indices,
                                                   tf.squeeze(tf.sign(gradient))))
                self.session.run(update_ops)
                self.session.run(self.adam_problem_step)

    def network(self, inputs):
        input_list = []
        gradients, sign = inputs['preprocessed_gradient']
        start_gradients = tf.slice(gradients, [0, self.gradient_history_ptr], [-1, -1])
        end_gradients = tf.slice(gradients, [0, 0], [-1, self.gradient_history_ptr])
        start_sign = tf.slice(sign, [0, self.gradient_history_ptr], [-1, -1])
        end_sign = tf.slice(sign, [0, 0], [-1, self.gradient_history_ptr])
        gradients_input = tf.concat([start_gradients, end_gradients], 1)
        sign_inputs = tf.concat([start_sign, end_sign], 1)
        final_input = None
        for i in range(self.global_args['limit']):
            gradient_slice = tf.slice(gradients_input, [0, i], [-1, 1])
            sign_slice = tf.slice(sign_inputs, [0, i], [-1, 1])
            curr_input = tf.concat([gradient_slice, sign_slice], 1)
            if final_input is None:
                final_input = curr_input
            else:
                final_input = tf.concat([final_input, curr_input], 1)
        activations = tf.nn.softplus(tf.add(tf.matmul(final_input, self.w_1), self.b_1))
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                activations = tf.nn.softplus(tf.add(tf.matmul(activations, layer[0]), layer[1]), name='layer_' + str(i))
        output = tf.add(tf.matmul(activations, self.w_out), self.b_out, name='layer_final_activation')
        return [output]

    def step(self):
        x_next = list()
        deltas_list = []
        for i, (variable, variable_gradient_history, variable_gradient_sign_history) in enumerate(
                zip(self.problem.variables, self.gradient_history, self.gradient_sign_history)):
            deltas = self.network({'preprocessed_gradient': [variable_gradient_history, variable_gradient_sign_history]})[0]
            deltas_list.append(deltas)
            deltas = tf.multiply(deltas, self.learning_rate, name='apply_learning_rate')
            deltas = self.problem.set_shape(deltas, like_variable=variable, op_name='reshape_deltas')
            x_next.append(tf.add(variable, deltas))
        return {'x_next': x_next, 'deltas': deltas_list}

    def update_gradient_history_ops(self, variable_ptr, gradients):
        ops = []
        indices = [[i, self.gradient_history_ptr] for i in range(gradients.shape[0].value)]
        gradient_slice = tf.slice(gradients, [0, 0], [-1, 1])
        gradient_sign_slice = tf.slice(gradients, [0, 1], [-1, 1])
        ops.append(tf.scatter_nd_update(self.gradient_history[variable_ptr], indices, tf.squeeze(gradient_slice)))
        ops.append(tf.scatter_nd_update(self.gradient_sign_history[variable_ptr], indices, tf.squeeze(gradient_sign_slice)))
        return ops

    def updates(self, args):
        update_list = super(MlpGradHistoryFAST, self).updates(args)
        gradients = self.get_preprocessed_gradients(args['x_next'])
        for i, gradient in enumerate(gradients):
            update_list.extend(self.update_gradient_history_ops(i, gradient))
        with tf.control_dependencies(update_list):
            update_itr = tf.cond(self.gradient_history_ptr < self.global_args['limit'] - 1,
                                 lambda: tf.assign_add(self.gradient_history_ptr, 1),
                                 lambda: tf.assign(self.gradient_history_ptr, 0))
        return update_list + [update_itr]

    def reset_optimizer(self):
        reset = super(MlpGradHistoryFAST, self).reset_optimizer()
        reset.append(tf.variables_initializer(self.gradient_history))
        return reset

    def reset_problem(self):
        reset = super(MlpGradHistoryFAST, self).reset_problem()
        reset.append(tf.variables_initializer(self.gradient_history))
        return reset


class MlpXHistory(MlpSimple):

    variable_history = None
    grad_sign_history = None
    history_ptr = None
    update_window = None
    guide_optimizer = None
    guide_step = None

    def __init__(self, problem, path, args):
        limit = args['limit']
        args['dims'] = (limit * 2, 1)
        super(MlpXHistory, self).__init__(problem, path, args)
        with tf.name_scope('mlp_x_optimizer_input_init'):
            self.history_ptr = tf.Variable(0, 'history_ptr')
            self.guide_optimizer = tf.train.GradientDescentOptimizer(.01, name='guide_optimizer')
            self.guide_step = self.guide_optimizer.minimize(self.problem.loss(self.problem.variables),
                                                            var_list=self.problem.variables, name='guide_step')
            self.variable_history = [tf.get_variable('variable_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, args['limit']], trainable=False)
                                     for i, shape in enumerate(self.problem.variables_flattened_shape)]
            self.grad_sign_history = [tf.get_variable('gradients_sign_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, args['limit']], trainable=False)
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
    def normalize_values(history_tensor):
        with tf.name_scope('mlp_x_normalize_variable_history'):
            max_values = tf.reduce_max(history_tensor, 1)
            min_values = tf.reduce_min(history_tensor, 1)
            max_values = tf.reshape(max_values, [tf.shape(max_values)[0], 1])
            min_values = tf.reshape(min_values, [tf.shape(min_values)[0], 1])
            diff = max_values - min_values
            return 2 * (history_tensor - min_values) / diff - 1.0

    def sort_input(self, inputs):
        with tf.name_scope('mlp_x_sort_input'):
            start = tf.slice(inputs, [0, self.history_ptr], [-1, -1], name='start')
            end = tf.slice(inputs, [0, 0], [-1, self.history_ptr], name='end')
            return tf.concat([start, end], 1, name='sorted_input')

    def network(self, inputs):
        with tf.name_scope('mlp_x_optimizer_network'):
            variable_history, variable_grad_sign_history = inputs['preprocessed_gradient']
            normalized_variable_history = self.normalize_values(variable_history)
            final_var_history = self.sort_input(normalized_variable_history)
            final_var_grad_history = self.sort_input(variable_grad_sign_history)
            final_input = tf.concat([final_var_history, final_var_grad_history], 1, name='final_input')
            activations = final_input
            activations = super(MlpXHistory, self).network({'preprocessed_gradient': activations, 'reuse': inputs['reuse']})[0]
            output = tf.tanh(activations)
            # output = Preprocess.clamp(activations, {'min':-1, 'max':1})
            return [output]

    def step(self):
        with tf.name_scope('mlp_x_optimizer_step'):
            x_next = list()
            deltas_list = []
            for i, (variable, variable_history, variable_grad_sign_history) in enumerate(zip(self.problem.variables,
                                                                                             self.variable_history,
                                                                                             self.grad_sign_history)):
                deltas = self.network({'preprocessed_gradient': [variable_history, variable_grad_sign_history], 'reuse': i > 0})[0]
                deltas_list.append(deltas)
                max_values = tf.reduce_max(variable_history, 1)
                min_values = tf.reduce_min(variable_history, 1)
                max_values = tf.reshape(max_values, [tf.shape(max_values)[0], 1])
                min_values = tf.reshape(min_values, [tf.shape(min_values)[0], 1])
                diff = max_values - min_values
                ref_points = max_values + min_values
                new_points = tf.add(tf.divide(ref_points, 2.0), tf.multiply(deltas, diff), 'new_points')
                new_points = self.problem.set_shape(new_points, like_variable=variable, op_name='reshaped_new_points')
                x_next.append(new_points)
                tf.summary.histogram('deltas_' + str(i), deltas)
                tf.summary.histogram('new_x_' + str(i), new_points)
                tf.summary.scalar('deltas', tf.squeeze(deltas))
                tf.summary.scalar('new_x', tf.squeeze(new_points))
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
        with tf.name_scope('mlp_x_optimizer_updates'):
            update_list = super(MlpXHistory, self).updates(args)
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

    def reset_optimizer(self):
        reset = super(MlpXHistory, self).reset_optimizer()
        reset.append(tf.variables_initializer(self.variable_history))
        reset.append(tf.variables_initializer(self.grad_sign_history))
        reset.append(tf.variables_initializer([self.history_ptr]))
        return reset

    def reset_problem(self):
        reset = super(MlpXHistory, self).reset_problem()
        reset.append(tf.variables_initializer(self.variable_history))
        reset.append(tf.variables_initializer(self.grad_sign_history))
        reset.append(tf.variables_initializer([self.history_ptr]))
        return reset

