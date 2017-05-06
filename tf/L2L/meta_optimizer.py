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
    W, b = None, None

    def __init__(self, problem, args):
        super(l2l, self).__init__(problem)
        self.state_size = args['state_size']
        self.num_layers = args['num_layers']
        self.unroll_len = args['unroll_len']
        self.optimizer = tf.train.AdamOptimizer(.01)

        # intialize for later use.
        with tf.variable_scope('rnn'):
            self.W = tf.get_variable('softmax_w', [self.state_size, 1])
            self.b = tf.get_variable('softmax_b', [1])
            self.meta_optimizer = tf.contrib.rnn.BasicLSTMCell(self.state_size)
            self.meta_optimizer = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.state_size) for _ in range(self.num_layers)])
            self.hidden_states = [self.meta_optimizer.zero_state(shape, tf.float32) for shape in self.problem.variables_flattened_shape]
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
                    params[i] = tf.add(params[i], deltas)
            fx_array = fx_array.write(t, self.problem.loss(params))
            t_next = t + 1
            return t_next, fx_array, params, hidden_states

        fx_array = tf.TensorArray(tf.float32, size=self.unroll_len,
                              clear_after_read=False)
        _, fx_array, vars_final, self.hidden_states = tf.while_loop(
            cond=lambda t, *_ : t < self.unroll_len,
            body=update,
            loop_vars=([0, fx_array, self.problem.variables, self.hidden_states]),
            parallel_iterations=1,
            swap_memory=True,
            name="unroll")

        with tf.variable_scope('update_params'):
            update_params = [tf.assign(self.problem.variables[i], vars_final[i]) for i, (_, _) in enumerate(zip(self.problem.variables, vars_final))]
        reset = fx_array.close()
        loss_sum = tf.divide(tf.reduce_sum(fx_array.stack()), self.unroll_len)
        # loss_sum = self.problem.loss(self.problem.variables)
        step = self.optimizer.minimize(loss_sum)
        # step = None
        return loss_sum, step, update_params, reset

    def optimize(self):
        print 'optimize'






