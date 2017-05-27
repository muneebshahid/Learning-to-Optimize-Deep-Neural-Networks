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
    optimizer = None
    hidden_states = None
    unroll_len = None
    second_derivatives = None
    preprocessor = None
    preprocessor_args = None
    debug_info = None
    best_eval = None
    variables = None
    constants = None

    def __init__(self, problem, path, args):
        if path is not None:
            print 'Loading optimizer args, ignoring provided args...'
            self.global_args = self.load_args(path)
            print 'Args Loaded, call load_optimizer with session to restore the optimizer graph.'
        else:
            self.global_args = args
        self.problem = problem
        if self.global_args.has_key('preprocess') and self.global_args['preprocess'] is not None:
            self.preprocessor = self.global_args['preprocess'][0]
            self.preprocessor_args = self.global_args['preprocess'][1]
        self.second_derivatives = self.global_args['second_derivatives']
        self.debug_info = []
        self.variables = []
        self.constants = []

    def meta_loss(self):
        pass

    def meta_minimize(self):
        pass

    def preprocess_input(self, inputs):
        if self.preprocessor is not None:
            return self.preprocessor(inputs, self.preprocessor_args)
        else:
            return inputs
    
    def flatten_input(self, i, inputs):
        return tf.reshape(inputs, [self.problem.variables_flattened_shape[i], 1])

    def get_gradients(self, variables):
        return tf.gradients(self.problem.loss(variables), variables)

    def get_processed_gradients(self, variables):
        gradients = self.get_gradients(variables)
        if not self.second_derivatives:
            gradients = [tf.stop_gradient(gradient) for gradient in gradients]
        for i, gradient in enumerate(gradients):
            gradients[i] = self.preprocess_input(self.flatten_input(i, gradient))
        return gradients

    def create_variable(self, name,  shape, initializer=tf.random_normal_initializer(mean=0, stddev=0.01), constant=False):
        x = tf.get_variable(name, shape=shape, dtype=tf.float32,
                                   initializer=initializer, trainable=not constant)
        if constant:
            self.constants.append(x)
        else:
            self.variables.append(x)
        return x

    def load_args(self, path):
        pickle_ready_args = pickle.load(open(path + '_config.p', 'rb'))
        pickle_ready_args['preprocess'][0] = getattr(Preprocess, pickle_ready_args['preprocess'][0])
        return pickle_ready_args

    def save_args(self, path):
        pickle_ready_args = dict(self.global_args)
        pickle_ready_args['preprocess'][0] = pickle_ready_args['preprocess'][0].func_name
        pickle.dump(pickle_ready_args, open(path + '_config.p', 'wb'))

    def load(self, sess, path):
        self.io_handle.restore(sess, path)
        print 'Optimizer Restored'

    def save(self, sess, path):
        print 'Saving optimizer'
        self.io_handle.save(sess, path)
        self.save_args(path)



class l2l(Meta_Optimizer):

    state_size = None
    num_layers = None
    learning_rate = None
    W, b = None, None

    def __init__(self, problem, path, args):
        super(l2l, self).__init__(problem, path, args)
        self.state_size = self.global_args['state_size']
        self.num_layers = self.global_args['num_layers']
        self.unroll_len = self.global_args['unroll_len']
        self.learning_rate = self.global_args['learning_rate']
        self.meta_optimizer = tf.train.AdamOptimizer(self.global_args['meta_learning_rate'])


        # initialize for later use.
        with tf.variable_scope('rnn_core'):

            # Formulate variables for all states as it allows to use tf.assign() for states
            def get_states(batch_size):
                state_variable = []
                for state_c, state_h in self.optimizer.zero_state(batch_size, tf.float32):
                    state_variable.append(tf.contrib.rnn.LSTMStateTuple(tf.Variable(state_c, trainable=False),
                                                                        tf.Variable(state_h, trainable=False)))
                return tuple(state_variable)

            self.optimizer = tf.contrib.rnn.BasicLSTMCell(self.state_size)
            self.optimizer = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.state_size) for _ in range(self.num_layers)])
            gradients = self.get_processed_gradients(self.problem.variables)[0]

            with tf.variable_scope('hidden_states'):
                self.hidden_states = [get_states(shape) for shape in self.problem.variables_flattened_shape]

            with tf.variable_scope('rnn_init'):
                self.optimizer(gradients, self.hidden_states[0])

            with tf.variable_scope('rnn_linear'):
                self.W = tf.get_variable('softmax_w', [self.state_size, 1])
                self.b = tf.get_variable('softmax_b', [1])

    def meta_loss(self):
        def update(t, fx_array, params, hidden_states):
            rnn_inputs = self.get_processed_gradients(params)
            with tf.variable_scope('rnn_core'):
                for i, (rnn_input, hidden_state) in enumerate(zip(rnn_inputs, hidden_states)):
                    with tf.variable_scope('rnn_init', reuse=True):
                        output, hidden_states[i] = self.optimizer(rnn_input, hidden_state)
                    deltas = tf.add(tf.matmul(output, self.W, name='output_matmul'), self.b, name='add_bias')
                    deltas = tf.reshape(deltas, self.problem.variables[i].get_shape(), name='reshape_deltas')
                    deltas = tf.multiply(deltas, self.learning_rate, 'multiply_deltas')
                    params[i] = tf.add(params[i], deltas, 'add_deltas_params')
            fx_array = fx_array.write(t, self.problem.loss(params))
            t_next = t + 1
            return t_next, fx_array, params, hidden_states

        fx_array = tf.TensorArray(tf.float32, size=self.unroll_len,
                              clear_after_read=False)
        _, fx_array, vars_final, hidden_states_final = tf.while_loop(
            cond=lambda t, *_ : t < self.unroll_len,
            body=update,
            loop_vars=([0, fx_array, self.problem.variables, self.hidden_states]),
            parallel_iterations=1,
            swap_memory=True,
            name="unroll")

        with tf.name_scope('update_params'):
            update_params = list()
            update_params.append([tf.assign(variable, variable_final) for variable, variable_final in zip(self.problem.variables, vars_final)])
            update_params.append([tf.assign(hidden_state, hidden_state_final) for hidden_state, hidden_state_final in zip(nest.flatten(self.hidden_states), nest.flatten(hidden_states_final))])

        with tf.name_scope('reset_variables'):
            reset = tf.variables_initializer(self.problem.variables + self.problem.constants + nest.flatten(self.hidden_states))

        loss_sum = tf.divide(tf.reduce_sum(fx_array.stack()), self.unroll_len)
        return [loss_sum, update_params, reset]

    def meta_minimize(self):
        info = self.meta_loss()
        step = self.meta_optimizer.minimize(info[0])
        info.append(step)
        return info



class mlp(Meta_Optimizer):

    w_1, b_1, w_out, b_out = None, None, None, None

    enable_momentum, avg_gradients, beta_1 = None, None, None

    layer_width = None

    def optimizer(self, input):
        layer_1_activations = tf.sigmoid(tf.add(tf.matmul(input, self.w_1), self.b_1), name='layer_1_activation')
        output = tf.add(tf.matmul(layer_1_activations, self.w_out), self.b_out, name='layer_final_activation')
        return tf.multiply(output, self.learning_rate, name='apply_learning_rate')

    def __init__(self, problem, path, args):
        super(mlp, self).__init__(problem, path, args)
        self.num_layers = self.global_args['num_layers']
        self.learning_rate = self.global_args['learning_rate']
        self.layer_width = self.global_args['layer_width']
        self.enable_momentum = self.global_args.has_key('momentum') and self.global_args['momentum']
        self.meta_optimizer = tf.train.AdamOptimizer(self.global_args['meta_learning_rate'])

        with tf.variable_scope('meta_optimizer_core'):
            init = tf.contrib.layers.xavier_initializer()
            input_dim, output_dim = (4, 2) if self.enable_momentum else (2, 1)
            self.w_1 = self.create_variable('w_1', shape=[input_dim, self.layer_width], initializer=init)
            self.b_1 = self.create_variable('b_1', shape=[1, self.layer_width], initializer=init)
            self.w_out = self.create_variable('w_out', shape=[self.layer_width, output_dim], initializer=init)
            self.b_out = self.create_variable('b_out', shape=[1, output_dim], initializer=init)
            # self.learning_rate = tf.get_variable('learning_rate', initializer=tf.constant(.0001, dtype=tf.float32))

            if self.enable_momentum:
                self.beta_1 = [tf.get_variable('beta_1' + str(i), shape=[shape, 1], initializer=tf.zeros_initializer(), trainable=False)
                               for i, shape in enumerate(self.problem.variables_flattened_shape)]
                self.avg_gradients = [tf.get_variable('avg_gradients_' + str(i), shape=[shape, 1], initializer=tf.zeros_initializer(), trainable=False)
                                      for i, shape in enumerate(self.problem.variables_flattened_shape)]

            # trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.io_handle = tf.train.Saver(self.variables, max_to_keep=100)


    def meta_loss(self):
        update_params = list()
        updated_vars = list()
        if self.enable_momentum:
            beta_1_new_list = list()

        # prepare optimizer inputs
        flat_gradients = [self.flatten_input(i, gradient) for i, gradient in enumerate(self.get_gradients(self.problem.variables))]
        preprocessed_gradients = [self.preprocess_input(flat_gradient) for flat_gradient in flat_gradients]
        if self.enable_momentum:
            avg_gradients_pre_processed = [self.preprocess_input(avg_gradient) for avg_gradient in self.avg_gradients]
            optimizer_inputs = [tf.concat([gradient_pre_processed, avg_gradient_pre_processed], 1)
                                for gradient_pre_processed, avg_gradient_pre_processed in
                                zip(preprocessed_gradients, avg_gradients_pre_processed)]
        else:
            optimizer_inputs = preprocessed_gradients
        deltas_list = list()
        for i, (variable, optim_input) in enumerate(zip(self.problem.variables, optimizer_inputs)):
            output = self.optimizer(optim_input)
            if self.enable_momentum:
                deltas_output = tf.slice(output, [0, 0], [-1, 1], name='deltas')
                beta_1_new = tf.slice(output, [0, 1], [-1, 1], name='beta_1_new')
                beta_1_new_list.append(beta_1_new)
            else:
                deltas_output = output
            deltas_list.append(deltas_output / self.learning_rate)
            deltas = tf.reshape(deltas_output, variable.get_shape(), name='reshape_deltas')
            updated_vars.append(tf.add(variable, deltas))
        loss = self.problem.loss(updated_vars)
        reset = tf.variables_initializer(self.problem.variables + self.problem.constants)
        # reset = None
        update_params.append([tf.assign(variable, updated_var) for variable, updated_var in zip(self.problem.variables, updated_vars)])
        if self.enable_momentum:
            update_params.append([tf.assign(avg_gradient, avg_gradient * tf.sigmoid(beta_1_t) + (1 - tf.sigmoid(beta_1_t)) * gradient)
                                  for gradient, avg_gradient, beta_1_t in
                                  zip(flat_gradients, self.avg_gradients, self.beta_1)])
            update_params.append([tf.assign(beta_1_old, beta_1_new) for beta_1_old, beta_1_new in zip(self.beta_1, beta_1_new_list)])
        self.debug_info = [flat_gradients, preprocessed_gradients, deltas_list]
        return [loss, update_params, reset]

    def meta_minimize(self):
        info = self.meta_loss()
        step = self.meta_optimizer.minimize(info[0])
        info.append(step)
        return info








