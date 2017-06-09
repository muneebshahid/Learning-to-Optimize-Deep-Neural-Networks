from __future__ import print_function
from abc import ABCMeta
import tensorflow as tf
from tensorflow.python.util import nest
import pickle
from preprocess import Preprocess


class Meta_Optimizer():
    __metaclass__ = ABCMeta

    global_args = None
    io_handle = None
    problem = None
    meta_optimizer = None
    second_derivatives = None
    preprocessor = None
    preprocessor_args = None
    debug_info = None
    trainable_variables = None

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
        self.second_derivatives = self.global_args['second_derivatives'] if 'second_derivatives' in self.global_args else False
        self.learning_rate = tf.get_variable('learning_rate', initializer=tf.constant(self.global_args['learning_rate']
                                                                                      if 'learning_rate' in self.global_args
                                                                                      else .0001,
                                                                                      dtype=tf.float32), trainable=False)
        self.meta_optimizer = tf.train.AdamOptimizer(self.global_args['meta_learning_rate'] if 'meta_learning_rate' in
                                                                                               self.global_args else .01)
        self.debug_info = []
        self.trainable_variables = []

    def __init_trainable_vars_list(self):
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def __init_io_handle(self):
        self.io_handle = tf.train.Saver(self.trainable_variables, max_to_keep=100)

    def end_init(self):
        self.__init_trainable_vars_list()
        self.__init_io_handle()

    def preprocess_input(self, inputs):
        if self.preprocessor is not None:
            return self.preprocessor(inputs, self.preprocessor_args)
        else:
            return inputs

    def flatten_input(self, i, inputs):
        return tf.reshape(inputs, [self.problem.variables_flattened_shape[i], 1])

    def get_gradients_raw(self, variables):
        return tf.gradients(self.problem.loss(variables), variables)

    def get_gradients(self, variables):
        gradients = self.get_gradients_raw(variables)
        if not self.second_derivatives:
            gradients = [tf.stop_gradient(gradient) for gradient in gradients]
        gradients = [self.flatten_input(i, gradient) for i, gradient in enumerate(gradients)]
        return gradients

    @property
    def optimizer_input_stack(self):
        variables = self.problem.variables
        gradients_raw = self.get_gradients_raw(variables)
        flat_gradients = [self.flatten_input(i, gradient) for i, gradient in enumerate(gradients_raw)]
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

    def core(self, inputs):
        pass

    def loss(self, variables):
        pass

    def step(self):
        pass

    def minimize(self, loss):
        return self.meta_optimizer.minimize(loss)

    def build(self):
        pass

    def reset_optimizer(self):
        return [tf.variables_initializer(self.trainable_variables)]

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



class l2l(Meta_Optimizer):

    state_size = None
    W, b = None, None
    lstm = None
    fx_array = None

    @property
    def optimizer_input_stack(self):
        inputs = super(l2l, self).optimizer_input_stack
        for (input, hidden_state) in zip(inputs, self.hidden_states):
            input['hidden_state'] = hidden_state
        return inputs

    def __init__(self, problem, path, args):
        super(l2l, self).__init__(problem, path, args)
        self.state_size = self.global_args['state_size']
        self.num_layers = self.global_args['num_layers']
        self.unroll_len = self.global_args['unroll_len']
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
            gradients = self.preprocess_input(self.get_gradients(self.problem.variables)[0])

            with tf.variable_scope('hidden_states'):
                self.hidden_states = [get_states(shape) for shape in self.problem.variables_flattened_shape]

            with tf.variable_scope('rnn_init'):
                self.lstm(gradients, self.hidden_states[0])

            with tf.variable_scope('rnn_linear'):
                self.W = tf.get_variable('softmax_w', [self.state_size, 1])
                self.b = tf.get_variable('softmax_b', [1])

        self.end_init()

    def core(self, inputs):
        with tf.variable_scope('optimizer_core/rnn_init', reuse=True):
            lstm_output, hidden_state = self.lstm(inputs['preprocessed_gradient'], inputs['hidden_state'])
        deltas = tf.add(tf.matmul(lstm_output, self.W, name='output_matmul'), self.b, name='add_bias')
        return [deltas, hidden_state]

    def step(self):
        def update(t, fx_array, params, hidden_states, deltas_list):
            rnn_inputs = self.preprocess_input(self.get_gradients(params))
            for i, (rnn_input, hidden_state) in enumerate(zip(rnn_inputs, hidden_states)):
                deltas, hidden_states[i] = self.core({'preprocessed_gradient': rnn_input, 'hidden_state': hidden_state})
                # overwrite each iteration of the while loop, so you will end up with the last update
                deltas_list[i] = deltas
                deltas = tf.reshape(deltas, self.problem.variables[i].get_shape(), name='reshape_deltas')
                deltas = tf.multiply(deltas, self.learning_rate, 'multiply_deltas')
                params[i] = tf.add(params[i], deltas, 'add_deltas_params')
            fx_array = fx_array.write(t, self.problem.loss(params))
            t_next = t + 1
            return t_next, fx_array, params, hidden_states, deltas_list

        deltas_list = list(range(len(self.hidden_states)))

        _, self.fx_array, x_next, h_next, deltas_list = tf.while_loop(
            cond=lambda t, *_: t < self.unroll_len,
            body=update,
            loop_vars=([0, self.fx_array, self.problem.variables, self.hidden_states, deltas_list]),
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

    def loss(self, variables):
        return tf.divide(tf.reduce_sum(self.fx_array.stack()), self.unroll_len)

    def build(self):
        step = self.step()
        updates = self.updates(step)
        loss = self.loss(step['x_next'])
        meta_step = self.minimize(loss)
        reset = [self.reset_problem(), self.reset_optimizer()]
        return step, updates, loss, meta_step, reset



class MlpSimple(Meta_Optimizer):

    w_1, b_1, w_out, b_out = None, None, None, None
    layer_width = None

    def __init__(self, problem, path, args):
        super(MlpSimple, self).__init__(problem, path, args)
        input_dim, output_dim = (2, 1) if 'dims' not in args else args['dims']
        self.num_layers = 2
        self.layer_width = self.global_args['layer_width'] if 'layer_width' in self.global_args else 20
        init = tf.random_normal_initializer(mean=0.0, stddev=.1)
        with tf.variable_scope('optimizer_core'):
            self.w_1 = tf.get_variable('w_1', shape=[input_dim, self.layer_width], initializer=init)
            self.b_1 = tf.get_variable('b_1', shape=[1, self.layer_width], initializer=tf.zeros_initializer)
            self.w_out = tf.get_variable('w_out', shape=[self.layer_width, output_dim], initializer=init)
            self.b_out = tf.get_variable('b_out', shape=[1, output_dim], initializer=tf.zeros_initializer)
        self.end_init()

    def core(self, inputs):
        layer_1_activations = tf.nn.softplus(tf.add(tf.matmul(inputs['preprocessed_gradient'], self.w_1), self.b_1))
        output = tf.add(tf.matmul(layer_1_activations, self.w_out), self.b_out, name='layer_final_activation')
        return [output]

    def step(self):
        x_next = list()
        deltas_list = []
        gradients = self.get_gradients(self.problem.variables)
        preprocessed_gradients = [self.preprocess_input(gradient) for gradient in gradients]
        optimizer_inputs = preprocessed_gradients
        for i, (variable, optim_input) in enumerate(zip(self.problem.variables, optimizer_inputs)):
            deltas = self.core({'preprocessed_gradient': optim_input})[0]
            deltas_list.append(deltas)
            deltas = tf.multiply(deltas, self.learning_rate, name='apply_learning_rate')
            deltas = tf.reshape(deltas, variable.get_shape(), name='reshape_deltas')
            x_next.append(tf.add(variable, deltas))
        return {'x_next': x_next, 'deltas': deltas_list}

    def updates(self, args):
        update_list = list()
        update_list.append(
            [tf.assign(variable, updated_var) for variable, updated_var in zip(self.problem.variables, args['x_next'])])
        return update_list

    def loss(self, variables):
        return self.problem.loss(variables)

    def build(self):
        step = self.step()
        updates = self.updates(step)
        loss = self.loss(step['x_next'])
        meta_step = self.minimize(loss)
        reset = [self.reset_problem(), self.reset_optimizer()]
        return step, updates, loss, meta_step, reset


class MlpMovingAverage(MlpSimple):

    avg_gradients = None
    def __init__(self, problem, path, args):
        args['dims'] = (4, 1)
        super(MlpMovingAverage, self).__init__(problem, path, args)
        self.avg_gradients = [
            tf.get_variable('avg_gradients_' + str(i), shape=[shape, 1], initializer=tf.zeros_initializer(),
                            trainable=False)
            for i, shape in enumerate(self.problem.variables_flattened_shape)]

    def updates(self, args):
        update_list = super(MlpMovingAverage, self).updates(args)
        gradients = self.get_gradients(args['x_next'])
        update_list.append([tf.assign(avg_gradient, avg_gradient * .9 + .1 * gradient)
                            for gradient, avg_gradient in zip(gradients, self.avg_gradients)])
        return update_list

    def reset_optimizer(self):
        reset = super(MlpMovingAverage, self).reset_optimizer()
        reset.append(tf.variables_initializer(self.avg_gradients))
        return reset

    def reset_problem(self):
        reset = super(MlpMovingAverage, self).reset_problem()
        reset.append(tf.variables_initializer(self.avg_gradients))
        return reset

