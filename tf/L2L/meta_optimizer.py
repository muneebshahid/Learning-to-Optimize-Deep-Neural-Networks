from abc import ABCMeta
import tensorflow as tf
from tensorflow.python.util import nest

class Meta_Optimizer():

    __metaclass__ = ABCMeta

    problem = None
    meta_optimizer = None
    optimizer = None
    hidden_states = None
    unroll_len = None

    def __init__(self, problem):
        self.problem = problem

    def step(self):
        pass

    def optimize(self):
        pass


class l2l(Meta_Optimizer):

    state_size = None
    num_layers = None
    learning_rate = None
    W, b = None, None

    def __init__(self, problem, args):
        super(l2l, self).__init__(problem)
        self.state_size = args['state_size']
        self.num_layers = args['num_layers']
        self.unroll_len = args['unroll_len']
        self.learning_rate = args['learning_rate']
        self.optimizer = tf.train.AdamOptimizer(args['meta_learning_rate'])

        # initialize for later use.
        with tf.variable_scope('rnn'):
            def get_states(batch_size):
                state_variable = []
                for state_c, state_h in self.meta_optimizer.zero_state(batch_size, tf.float32):
                    state_variable.append(tf.contrib.rnn.LSTMStateTuple(tf.Variable(state_c, trainable=False),
                                                                        tf.Variable(state_h, trainable=False)))
                return tuple(state_variable)
            self.W = tf.get_variable('softmax_w', [self.state_size, 1])
            self.b = tf.get_variable('softmax_b', [1])
            self.meta_optimizer = tf.contrib.rnn.BasicLSTMCell(self.state_size)
            self.meta_optimizer = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.state_size) for _ in range(self.num_layers)])
            self.hidden_states = [get_states(shape) for shape in self.problem.variables_flattened_shape]
            gradients = self.problem.get_gradients(self.problem.variables)[0]
            self.meta_optimizer(gradients, self.hidden_states[0])

    def step(self):
        def update(t, fx_array, params, hidden_states):
            rnn_inputs = self.problem.get_gradients(params)
            with tf.variable_scope('rnn', reuse=True):
                for i, (rnn_input, hidden_state) in enumerate(zip(rnn_inputs, hidden_states)):
                    output, hidden_states[i] = self.meta_optimizer(rnn_input, hidden_state)
                    deltas = tf.add(tf.matmul(output, self.W), self.b)
                    deltas = tf.reshape(deltas, self.problem.variables[i].get_shape())
                    deltas = tf.multiply(deltas, self.learning_rate)
                    params[i] = tf.add(params[i], deltas)
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
            # update_params.append([tf.assign(self.problem.variables[i], vars_final[i]) for i, (_, _) in enumerate(zip(self.problem.variables, vars_final))])
            # update_hidden_states = list()
            # for i, _ in enumerate(self.hidden_states):
            #     for j in range(self.num_layers):
            #         for k in range(len(self.hidden_states[i][j])):
            #             update_hidden_states.append(tf.assign(self.hidden_states[i][j][k], hidden_states_final[i][j][k]))
            # update_params.append(update_hidden_states)

        with tf.name_scope('reset_variables'):
            reset = tf.variables_initializer(self.problem.variables + self.problem.constants + nest.flatten(self.hidden_states))
            # reset.append(self.problem.constants)
            # for i, _ in enumerate(self.hidden_states):
            #     for j in range(self.num_layers):
            #         reset.append(tf.variables_initializer(self.hidden_states[i][j]))

        loss_sum = tf.divide(tf.reduce_sum(fx_array.stack()), self.unroll_len)
        # loss_sum = self.problem.loss(self.problem.variables)
        step = self.optimizer.minimize(loss_sum)
        # step = None
        return loss_sum, step, update_params, reset

    def optimize(self):
        print 'optimize'






