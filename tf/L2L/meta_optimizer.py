from abc import ABCMeta
import tensorflow as tf
from tensorflow.python.util import nest
import preprocess

class Meta_Optimizer():

    __metaclass__ = ABCMeta

    problem = None
    meta_optimizer = None
    optimizer = None
    hidden_states = None
    unroll_len = None
    second_derivatives = None
    preprocessor = None

    def __init__(self, problem, processing_constant=None, second_derivatives=False):
        self.problem = problem
        if processing_constant is not None:
            self.preprocessor = preprocess.LogAndSign(processing_constant)
        self.second_derivatives = second_derivatives

    def meta_loss(self):
        pass

    def meta_minimize(self):
        pass

    def preprocess_input(self, inputs):
        if self.preprocessor is not None:
            return self.preprocessor.process(inputs)
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

class l2l(Meta_Optimizer):

    state_size = None
    num_layers = None
    learning_rate = None
    W, b = None, None

    def __init__(self, problem, processing_constant, second_derivatives, args):
        super(l2l, self).__init__(problem, processing_constant, second_derivatives)
        self.state_size = args['state_size']
        self.num_layers = args['num_layers']
        self.unroll_len = args['unroll_len']
        self.learning_rate = args['learning_rate']
        self.optimizer = tf.train.AdamOptimizer(args['meta_learning_rate'])

        # initialize for later use.
        with tf.variable_scope('rnn_core'):

            # Formulate variables for all states as it allows to use tf.assign() for states
            def get_states(batch_size):
                state_variable = []
                for state_c, state_h in self.meta_optimizer.zero_state(batch_size, tf.float32):
                    state_variable.append(tf.contrib.rnn.LSTMStateTuple(tf.Variable(state_c, trainable=False),
                                                                        tf.Variable(state_h, trainable=False)))
                return tuple(state_variable)

            self.meta_optimizer = tf.contrib.rnn.BasicLSTMCell(self.state_size)
            self.meta_optimizer = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.state_size) for _ in range(self.num_layers)])
            gradients = self.get_processed_gradients(self.problem.variables)[0]

            with tf.variable_scope('hidden_states'):
                self.hidden_states = [get_states(shape) for shape in self.problem.variables_flattened_shape]

            with tf.variable_scope('rnn_init'):
                self.meta_optimizer(gradients, self.hidden_states[0])

            with tf.variable_scope('rnn_linear'):
                self.W = tf.get_variable('softmax_w', [self.state_size, 1])
                self.b = tf.get_variable('softmax_b', [1])

    def meta_loss(self):
        def update(t, fx_array, params, hidden_states):
            rnn_inputs = self.get_processed_gradients(params)
            with tf.variable_scope('rnn_core'):
                for i, (rnn_input, hidden_state) in enumerate(zip(rnn_inputs, hidden_states)):
                    with tf.variable_scope('rnn_init', reuse=True):
                        output, hidden_states[i] = self.meta_optimizer(rnn_input, hidden_state)
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
        step = self.optimizer.minimize(info[0])
        info.append(step)
        return info



class mlp(Meta_Optimizer):

    w_1, b_1, w_out, b_out = None, None, None, None
    avg_gradients, beta_1 = None, None
    def meta_optimizer(self, input):
        layer_activations = tf.sigmoid(tf.add(tf.matmul(input, self.w_1), self.b_1))
        output = tf.tanh(tf.add(tf.matmul(layer_activations, self.w_out), self.b_out))
        return output * self.learning_rate

    def __init__(self, problem, processing_constant, second_derivatives, args):
        super(mlp, self).__init__(problem, processing_constant, second_derivatives)
        self.num_layers = args['num_layers']
        self.learning_rate = args['learning_rate']
        self.optimizer = tf.train.AdamOptimizer(args['meta_learning_rate'])
        # init = tf.contrib.layers.xavier_initializer()
        init = tf.random_normal_initializer(mean=0)
        self.w_1 = tf.get_variable('w_1', shape=[4, 20], initializer=init)
        self.b_1 = tf.get_variable('b_1', shape=[1, 20], initializer=init)
        self.w_out = tf.get_variable('w_out', shape=[20, 2], initializer=init)
        self.b_out = tf.get_variable('b_out', shape=[1, 2], initializer=init)
        self.beta_1 = [tf.get_variable('beta_1' + str(i), shape=[shape, 1], initializer=tf.zeros_initializer())
                       for i, shape in enumerate(self.problem.variables_flattened_shape)]
        self.avg_gradients = [tf.get_variable('avg_gradients_' + str(i), shape=[shape, 1], initializer=tf.zeros_initializer())
                              for i, shape in enumerate(self.problem.variables_flattened_shape)]

    def meta_loss(self):
        update_params = list()
        updated_vars = list()
        beta_1_new_list = list()

        # prepare optimizer inputs
        flat_gradients = [self.flatten_input(i, gradient) for i, gradient in enumerate(self.get_gradients(self.problem.variables))]
        preprocessed_gradients = [self.preprocess_input(flat_gradient) for flat_gradient in flat_gradients]
        avg_gradients_pre_processed = [self.preprocess_input(avg_gradient) for avg_gradient in self.avg_gradients]
        optimizer_inputs = [tf.concat([gradient_pre_processed, avg_gradient_pre_processed], 1)
                        for gradient_pre_processed, avg_gradient_pre_processed in
                         zip(preprocessed_gradients, avg_gradients_pre_processed)]
        
        for i, (variable, optim_input) in enumerate(zip(self.problem.variables, optimizer_inputs)):
            output = self.meta_optimizer(optim_input)
            deltas = tf.slice(output, [0, 0], [-1, 1], name='deltas')
            beta_1_new = tf.slice(output, [0, 1], [-1, 1], name='beta_1_new')
            beta_1_new_list.append(beta_1_new)
            deltas = tf.reshape(deltas, variable.get_shape(), name='reshape_deltas')
            updated_vars.append(tf.add(variable, deltas))
        loss = self.problem.loss(updated_vars)
        # reset = tf.variables_initializer(self.problem.variable + self.problem.constants)
        reset = None
        update_params.append([tf.assign(variable, updated_var) for variable, updated_var in zip(self.problem.variables, updated_vars)])
        update_params.append([tf.assign(avg_gradient, avg_gradient * tf.sigmoid(beta_1_t) + (1 - tf.sigmoid(beta_1_t)) * gradient)
                              for gradient, avg_gradient, beta_1_t in
                              zip(flat_gradients, self.avg_gradients, self.beta_1)])
        update_params.append([tf.assign(beta_1_old, beta_1_new) for beta_1_old, beta_1_new in zip(self.beta_1, beta_1_new_list)])
        return [loss, update_params, reset]

    def meta_minimize(self):
        info = self.meta_loss()
        step = self.optimizer.minimize(info[0])
        info.append(step)
        return info








