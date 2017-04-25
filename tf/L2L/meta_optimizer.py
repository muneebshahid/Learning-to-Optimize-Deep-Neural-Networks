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
        self.hidden_states = self.meta_optimizer.zero_state(problem.batch_size, tf.float32)
        with tf.variable_scope('rnn'):
            self.meta_optimizer(problem.gradients(problem.vars), self.hidden_states)
            self.W = tf.get_variable('softmax_w', [self.state_size, self.problem.dim])
            self.b = tf.get_variable('softmax_b', [self.problem.dim])

    def step(self):
        def update(t, loss, params, hidden_states):
            with tf.variable_scope('rnn', reuse=True):
                output, hidden_states_next = self.meta_optimizer(self.problem.gradients(params), hidden_states)
                deltas = tf.matmul(output, self.W) + self.b
            params_next = params + deltas
            loss += self.problem.loss(params_next)
            t_next = t + 1
            return t_next, loss, params_next, hidden_states_next

        _, loss_final, self.problem.vars, self.hidden_states = tf.while_loop(
            cond=lambda t, *_ : t < self.unroll_len,
            body=update,
            loop_vars=([0, tf.zeros([self.problem.batch_size, 1]), self.problem.vars, self.hidden_states]),
            parallel_iterations=1,
            swap_memory=True,
            name="unroll")

        loss_sum = tf.divide(tf.reduce_sum(loss_final), self.unroll_len)
        step = self.optimizer.minimize(loss_sum)
        return loss_sum, step

    def optimize(self):
        print 'optimize'






