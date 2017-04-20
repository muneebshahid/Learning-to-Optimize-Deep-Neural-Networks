from abc import ABCMeta, abstractmethod
import tensorflow as tf
class Meta_Optimizer():

    __metaclass__ = ABCMeta

    params = None
    func = None
    meta_optimizer = None
    optimizer = None
    hidden_states = None
    unroll_len = None

    def __init__(self, params, func):
        self.params = params
        self.func = func
    
    def get_gradients(self, func, params):
		return tf.gradients(func(params), params)[0]

    def step(self):
        pass

    def optimize(self):
        pass

class l2l(Meta_Optimizer):

    state_size = None
    num_layers = None
    W, b = None, None

    def __init__(self, params, func, args):
        super(l2l, self).__init__(params, func)
        self.state_size = args['state_size']
        self.num_layers = args['num_layers']
        self.unroll_len = args['unroll_len']
        self.optimizer = tf.train.AdamOptimizer(.01)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        self.meta_optimizer = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers)
        self.hidden_state = self.meta_optimizer.zero_state(1, tf.float32)
        with tf.variable_scope('rnn'):
            self.meta_optimizer(self.get_gradients(self.func, self.params), self.hidden_state)
            self.W = tf.get_variable('softmax_w', [self.state_size, 1])
            self.b = tf.get_variable('softmax_b', [1])


    def step(self):
        def update(t, loss):
            with tf.variable_scope('rnn', reuse=True):				
                output, self.hidden_state = self.meta_optimizer(self.get_gradients(self.func, self.params), self.hidden_state)
                deltas = tf.matmul(output, self.W) + self.b
            self.params = self.params + deltas
            loss += self.func(self.params)
            t_next = t + 1
            return t_next, loss

        _, loss_final =  tf.while_loop(
            cond=lambda t, *_ : t < 20,
            body=update,
            loop_vars=([0, tf.zeros([1, 1])]),
            parallel_iterations=1,
            swap_memory=True,
            name="unroll")

        step = self.optimizer.minimize(loss_final)

        return loss_final, step


    def optimize(self):
        print 'optimize'






