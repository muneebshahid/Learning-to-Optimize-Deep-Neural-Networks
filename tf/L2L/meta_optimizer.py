from abc import ABCMeta
import tensorflow as tf

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
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        self.meta_optimizer = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers)
        self.hidden_states = [self.meta_optimizer.zero_state(shape, tf.float32) for shape in self.problem.variables_flattened_shape]

        # intialize for later use
        with tf.variable_scope('rnn'):
            self.meta_optimizer(self.problem.get_gradients(self.problem.variables)[0], self.hidden_states[0])
            self.W = tf.get_variable('softmax_w', [self.state_size, self.problem.dims])
            self.b = tf.get_variable('softmax_b', [self.problem.dims])

    def step(self):
        def update(t, loss, params, hidden_states):
            gradients = self.problem.get_gradients(params)
            with tf.variable_scope('rnn', reuse=True):
                for i, (gradient, hidden_state) in enumerate(zip(gradients, hidden_states)):
                    output, hidden_states[i] = self.meta_optimizer(gradient, hidden_state)
                    deltas = tf.add(tf.matmul(output, self.W), self.b)
                    params[i] = tf.add(params[i], deltas)
            loss += self.problem.loss(params)
            t_next = t + 1
            return t_next, loss, params, hidden_states

        _, loss_final, vars_final, self.hidden_states = tf.while_loop(
            cond=lambda t, *_ : t < self.unroll_len,
            body=update,
            loop_vars=([0, tf.zeros([self.problem.batch_size, 1]), self.problem.variables, self.hidden_states]),
            parallel_iterations=1,
            swap_memory=True,
            name="unroll")
        # update_params = [tf.assign(self.problem.variables[0], vars_final[0])]
        # _, loss_final, vars_final, self.hidden_states = update(0, 0, self.problem.vars, self.hidden_states)
        # self.problem.variables = vars_final
        with tf.variable_scope('update_params'):
            update_params = [tf.assign(self.problem.variables[i], vars_final[i]) for i, (_, _) in enumerate(zip(self.problem.variables, vars_final))]
            

        loss_sum = tf.divide(tf.reduce_sum(loss_final), self.unroll_len)
        step = self.optimizer.minimize(loss_sum)
        return loss_sum, step, update_params

    def optimize(self):
        print 'optimize'






