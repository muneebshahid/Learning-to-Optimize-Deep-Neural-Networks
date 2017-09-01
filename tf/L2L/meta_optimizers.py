from __future__ import print_function
from abc import ABCMeta
import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
import pickle
from preprocess import Preprocess
from timeit import default_timer as timer



class Meta_Optimizer():
    __metaclass__ = ABCMeta

    io_handle = None
    problems = None
    meta_optimizer_optimizer = None
    preprocessor = None
    preprocessor_args = None
    optimizer_variables = None
    session = None

    ops_step = None
    ops_updates = None
    ops_loss = None
    ops_meta_step = None
    ops_reset_problem = None

    def __init__(self, problems, path, args):
        if path is not None:
            print('Loading optimizer args, ignoring provided args...')
            args = self.load_args(path)
            print('Args Loaded, call load_optimizer with session to restore the optimizer graph.')
        self.problems = problems
        if self.is_availble('preprocess', args):
            self.preprocessor = args['preprocess'][0]
            self.preprocessor_args = args['preprocess'][1]
        self.meta_optimizer_optimizer = tf.train.RMSPropOptimizer(args['meta_learning_rate'] if
                                                                  self.is_availble('meta_learning_rate', args) else .01,
                                                                  name='meta_optimizer_optimizer')
        self.optimizer_variables = []

    def init_saver_handle(self):
        self.io_handle = tf.train.Saver([variable for variable in self.optimizer_variables], max_to_keep=100)

    def preprocess_input(self, inputs):
        if self.preprocessor is not None:
            return self.preprocessor(inputs, self.preprocessor_args)
        else:
            return inputs

    def is_availble(self, param, args):
        return param in args and args[param] is not None

    def get_preprocessed_gradients(self, problem, variables=None):
        return [self.preprocess_input(gradient) for gradient in problem.get_gradients(variables)]

    @property
    def meta_optimizer_input_stack(self):
        variables = self.problems.variables_flat
        gradients_raw = self.problems.get_gradients_raw(variables)
        flat_gradients = [self.problems.flatten_input(i, gradient) for i, gradient in enumerate(gradients_raw)]
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

    def updates(self, args=None):
        pass

    def network(self, args=None):
        pass

    def loss(self, args=None):
        pass

    def step(self, args=None):
        pass

    def minimize(self, loss):
        return self.meta_optimizer_optimizer.minimize(loss, var_list=self.optimizer_variables)

    def build(self):
        self.init_saver_handle()
        pass

    def reset_optimizer(self):
        return [tf.variables_initializer(self.optimizer_variables, name='reset_optimizer')]

    def reset_problem(self, problem):
        return [tf.variables_initializer(problem.variables + problem.constants, name='reset_' + problem.__class__.__name__)]

    def reset_problems(self):
        reset_problem_ops = []
        for problem in self.problems:
            reset_problem_ops.extend(self.reset_problem(problem))
        return reset_problem_ops

    def restore_problem(self, index, path):
        self.problems[index].restore(self.session, path)

    @staticmethod
    def load_args(path):
        pickled_args = pickle.load(open(path + '_config.p', 'rb'))
        pickled_args['preprocess'][0] = getattr(Preprocess, pickled_args['preprocess'][0])
        return pickled_args

    def save_args(self, path):
        dump_args = dict(self.g_args)
        dump_args['preprocess'][0] = dump_args['preprocess'][0].func_name
        pickle.dump(dump_args, open(path + '_config.p', 'wb'))

    def load(self, path):
        self.io_handle.restore(self.session, path)
        print('Optimizer Restored')

    def save(self, path):
        print('Saving optimizer')
        self.io_handle.save(self.session, path)
        # self.save_args(path)

    def run_init(self, args=None):
        return

    def set_session(self, session):
        self.session = session

    def run_reset(self, index=None, optimizer=False):
        pass

    def run(self, args=None):
        set_arg = lambda op, op_key: op if op_key in args and args[op_key] else []
        num_steps = 1 if 'num_steps' not in args else args['num_steps']
        ops_reset = set_arg(self.ops_reset_problem, 'ops_reset')
        ops_loss = set_arg(self.ops_loss, 'ops_loss')
        ops_meta_step = set_arg(self.ops_meta_step, 'ops_meta_step')
        ops_updates = set_arg(self.ops_updates, 'ops_updates')
        if ops_reset:
            self.run_reset()
        loss_array = 0
        start = timer()
        for _ in range(num_steps):
            loss = self.session.run([ops_loss, ops_meta_step, ops_updates])[0]
            loss_array += np.array(loss)
        return timer() - start, loss_array / num_steps


def layer_fc(name, dims, inputs, variable_list, initializers=None, activation=None):
    initializers = [tf.random_normal_initializer(mean=0.0, stddev=.1), tf.zeros_initializer] \
        if initializers is None else initializers
    # initializers = [tf.contrib.layers.variance_scaling_initializer()]
    reuse = False
    with tf.name_scope('optimizer_fc_layer_' + name):
        with tf.variable_scope('optimizer_network') as scope:
            try:
                w = tf.get_variable('w_' + name, shape=dims, initializer=initializers[0])
            except ValueError:
                scope.reuse_variables()
                reuse = True
                w = tf.get_variable('w_' + name, shape=dims, initializer=initializers[0])
            b = tf.get_variable('b_' + name, shape=[1, dims[-1]], initializer=initializers[1])
            linear = tf.add(tf.matmul(inputs, w), b, name='activations_' + 'layer_' + str(name))
            layer_output = linear if activation is None else activation(linear)

            if not reuse:
                variable_list.extend([w, b])
                tf.summary.histogram('weights', w)
                tf.summary.histogram('bias', b)
                tf.summary.histogram('activation', layer_output)
    return layer_output

class l2l(Meta_Optimizer):

    state_size = None
    unroll_len = None
    optim_per_epoch = None
    W, b = None, None
    lstm = None
    fx_array = None
    learning_rate = None

    @property
    def meta_optimizer_input_stack(self):
        inputs = super(l2l, self).meta_optimizer_input_stack
        for (input, hidden_state) in zip(inputs, self.hidden_states):
            input['hidden_state'] = hidden_state
        return inputs

    def __init__(self, problems, path, args):
        super(l2l, self).__init__(problems, path, args)
        self.state_size = args['state_size']
        self.num_layers = args['num_layers']
        self.unroll_len = args['unroll_len']
        self.num_step = args['optim_per_epoch'] // self.unroll_len
        self.learning_rate = tf.get_variable('learning_rate', initializer=tf.constant(args['learning_rate'], dtype=tf.float32))
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
            gradients = self.preprocess_input(self.problems.get_gradients()[0])

            with tf.variable_scope('hidden_states'):
                self.hidden_states = [get_states(self.problems.get_shape(variable=variable)) for variable in self.problems.variables_flat]

            with tf.variable_scope('rnn_init'):
                self.lstm(gradients, self.hidden_states[0])

            with tf.variable_scope('rnn_linear'):
                self.W = tf.get_variable('softmax_w', [self.state_size, 1])
                self.b = tf.get_variable('softmax_b', [1])

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
                deltas = self.problems.set_shape(deltas, i, op_name='reshape_deltas')
                deltas = tf.multiply(deltas, self.learning_rate, 'multiply_deltas')
                params[i] = tf.add(params[i], deltas, 'add_deltas_params')
            fx_array = fx_array.write(t, self.problems.loss(params))
            t_next = t + 1
            return t_next, fx_array, params, hidden_states

        deltas_list = list(range(len(self.hidden_states)))

        _, self.fx_array, x_next, h_next = tf.while_loop(
            cond=lambda t, *_: t < self.unroll_len,
            body=update,
            loop_vars=([0, self.fx_array, self.problems.variables, self.hidden_states]),
            parallel_iterations=1,
            swap_memory=True,
            name="unroll")

        return {'x_next': x_next, 'h_next': h_next, 'deltas': deltas_list}

    def updates(self, args):
        update_list = list()
        update_list.append([tf.assign(variable, variable_final) for variable, variable_final in
                            zip(self.problems.variables, args['x_next'])])
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
        return super(l2l, self).run(args)

class MlpSimple(Meta_Optimizer):

    w_1, b_1, w_out, b_out = None, None, None, None
    layer_width = None
    hidden_layers = None
    network_in_dims = None
    network_out_dims = None

    def __init__(self, problems, path, args):
        super(MlpSimple, self).__init__(problems, path, args)
        self.layer_width = args['layer_width']
        self.network_in_dims = args['network_in_dims']
        self.network_out_dims = args['network_out_dims']
        self.hidden_layers = args['hidden_layers']
        self.learning_rate = tf.get_variable('learning_rate',
                                             initializer=tf.constant(args['learning_rate'], dtype=tf.float32))

    def network(self, args=None):
        hidden_activation = args['h_act'] if 'h_act' in args else tf.nn.relu
        output_activation = args['o_act'] if 'o_act' in args else None
        activations = args['preprocessed_gradient']
        activations = layer_fc(name='in', dims=[self.network_in_dims, self.layer_width], inputs=activations,
                               variable_list=self.optimizer_variables, activation=hidden_activation)
        for layer in range(self.hidden_layers):
            activations = layer_fc(str(layer + 1), dims=[self.layer_width, self.layer_width], inputs=activations,
                                   variable_list=self.optimizer_variables, activation=hidden_activation)
        output = layer_fc('out', dims=[self.layer_width, self.network_out_dims], inputs=activations,
                          variable_list=self.optimizer_variables, activation=output_activation)
        return [output]

    def step(self, args=None):
        with tf.name_scope('mlp_simple_optimizer_step'):
            problem = args['problem']
            x_next = list()
            deltas_list = []
            preprocessed_gradients = self.get_preprocessed_gradients(problem)
            optimizer_inputs = preprocessed_gradients
            for i, (variable, optim_input) in enumerate(zip(problem.variables, optimizer_inputs)):
                deltas = self.network({'preprocessed_gradient': optim_input})[0]
                deltas_list.append(deltas)
                deltas = tf.multiply(deltas, self.learning_rate, name='apply_learning_rate')
                deltas = problem.set_shape(deltas, like_variable=variable, op_name='reshape_deltas')
                x_next.append(tf.add(variable, deltas))
            return {'x_next': x_next, 'deltas': deltas_list}

    def updates(self, args=None):
        with tf.name_scope('mlp_simple_optimizer_updates'):
            problem = args['problem']
            update_list = [tf.assign(variable, updated_var) for variable, updated_var in zip(problem.variables, args['x_next'])]
            return update_list

    def loss(self, args=None):
        with tf.name_scope('mlp_simple_optimizer_loss'):
            problem = args['problem']
            variables = args['x_next'] if 'x_next' in args else problem.variables
            return problem.loss(variables)

    def build(self):
        self.ops_step = []
        self.ops_updates = []
        self.ops_loss = []
        self.ops_meta_step = []
        self.ops_final_loss = 0
        self.ops_reset = [self.reset_optimizer()]
        for problem in self.problems:
            step = self.step(problem)
            args = {'problem': problem, 'x_next': step['x_next']}
            updates = self.updates(args)
            loss = self.loss(args)
            reset = self.reset_problem(problem)
            self.ops_step.append(step)
            self.ops_updates.append(updates)
            self.ops_loss.append(loss)
            self.ops_reset.append(reset)
        for op_loss in self.ops_loss:
            self.ops_final_loss += op_loss
        self.ops_final_loss /= len(self.ops_loss)
        self.ops_meta_step = self.minimize(self.ops_final_loss)

    def run(self, args=None):
        return super(MlpSimple, self).run(args)


class MlpMovingAverage(MlpSimple):

    avg_gradients = None
    def __init__(self, problems, path, args):
        args['dims'] = (4, 1) if self.is_availble('preprocess', args) else (2, 1)
        super(MlpMovingAverage, self).__init__(problems, path, args)
        self.avg_gradients = [
            tf.get_variable('avg_gradients_' + str(i), shape=[shape, 1], initializer=tf.zeros_initializer(),
                            trainable=False)
            for i, shape in enumerate(self.problems.variables_flattened_shape)]

    def step(self):
        x_next = list()
        deltas_list = []
        preprocessed_gradients = self.get_preprocessed_gradients()
        optimizer_inputs = [tf.concat([gradient, self.preprocess_input(avg_gradient)], 1)
                            for gradient, avg_gradient in zip(preprocessed_gradients, self.avg_gradients)]
        for i, (variable, optim_input) in enumerate(zip(self.problems.variables, optimizer_inputs)):
            deltas = self.network({'preprocessed_gradient': optim_input})[0]
            deltas_list.append(deltas)
            deltas = tf.multiply(deltas, self.learning_rate, name='apply_learning_rate')
            deltas = self.problems.set_shape(deltas, like_variable=variable, op_name='reshape_deltas')
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

class MlpNormHistory(Meta_Optimizer):

    network_in_dims = None
    network_out_dims = None
    layer_width = None
    hidden_layers = None
    limit = None
    use_momentums = None
    use_guide_step = None
    gradients_only = None
    gradient_sign_only = None
    vari_hist = None
    grad_hist = None
    history_ptr = None
    update_window = None
    guide_optimizer = None
    guide_step = None
    network_activation = None
    li, lr = None, None
    sign_dist = None
    lr_dist = None
    history_range = None
    min_lr = None
    min_step_max = None
    momentum_alpha = None
    momentum_alpha_inv = None
    momentum_base = None
    grad_mom = None
    vari_mom = None
    normalize_with_sq_grad = None
    sq_vari_hist = None
    sq_grad_hist = None
    use_dist_mv_avg = None
    dist_mv_avg = None
    use_noise_est = None
    use_log_noise = None
    use_delta_mv_avg = None
    step_dist_max_step = None
    delta_mv_avg = None
    learn_lr = None
    learn_lr_delta = None
    lr_delta_dist = None


    def __init__(self, problems, path, args):
        super(MlpNormHistory, self).__init__(problems, path, args)
        self.gradients_only = args['grad_only']
        self.gradient_sign_only = args['grad_sign_only']
        self.layer_width = args['layer_width']
        self.hidden_layers = args['hidden_layers']
        self.network_activation = tf.nn.relu
        self.limit = args['limit']
        self.network_in_dims =  args['network_in_dims']
        self.network_out_dims = args['network_out_dims']
        self.use_momentums = args['use_momentum']
        self.history_range = args['history_range']
        self.min_lr = args['min_step']
        self.min_step_max = args['min_step_max']
        self.momentum_base = args['momentum_base']
        self.normalize_with_sq_grad = args['normalize_with_sq_grad']
        self.use_delta_mv_avg  = args['use_delta_mv_avg']
        self.use_dist_mv_avg = args['use_dist_mv_avg']
        self.use_guide_step = args['use_guide_step']
        self.learn_momentum_base = args['learn_momentum_base']
        self.use_noise_est = args['enable_noise_est']
        self.use_log_noise = args['use_log_noise']
        self.step_dist_max_step = args['step_dist_max_step']
        self.learn_lr = args['learn_lr']
        self.learn_lr_delta = args['learn_lr_delta']


        with tf.name_scope('Optim_Init'):
            self.step_dist = tf.Variable(tf.constant(np.linspace(0.0, self.step_dist_max_step, 10), shape=[10, 1], dtype=tf.float32),
                                         name='step_dist')
            self.sign_dist = tf.Variable(tf.constant([-1.0, 1.0], shape=[2, 1], dtype=tf.float32),
                                         name='sign_dist')
            self.lr_dist = tf.Variable(tf.constant([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.0], shape=[9, 1], dtype=tf.float32),
                                   name='grad_dist')
            self.lr_delta_dist = tf.Variable(tf.constant([0.5, 0.75, 1.0, 1.25, 1.5], shape=[5, 1], dtype=tf.float32), name='delta_lr_dist')
            self.guide_optimizer = tf.train.AdamOptimizer(1, name='guide_optimizer')

            if self.learn_momentum_base:
                self.momentum_alpha = []
            else:
                alpha = []
                for i in np.linspace(1, 17, self.limit, dtype=np.int32):
                    alpha.append(1 / np.power(self.momentum_base, i))
                self.momentum_alpha = tf.constant(np.array(alpha), shape=[1, self.limit], dtype=tf.float32)
                self.momentum_alpha_inv = tf.subtract(1.0, self.momentum_alpha)

            (self.guide_step, self.vari_hist, self.grad_hist,
             self.grad_mom, self.vari_mom, self.sq_vari_hist,
             self.sq_grad_hist, self.dist_mv_avg, self.delta_mv_avg, self.lr_mv_avg) = [], [], [], [], [], [], [], [], [], []

            for i, problem in enumerate(self.problems):
                with tf.variable_scope('problem_' + str(i)):
                    if self.use_guide_step:
                        self.guide_step.append(self.guide_optimizer.minimize(problem.loss(problem.variables),
                                                                             var_list=problem.variables,
                                                                             name='guide_step'))
                    else:
                        self.guide_step.append([])
                    self.vari_hist.append([tf.get_variable('vari_hist' + str(i), initializer=tf.zeros_initializer,
                                                           shape=[shape, self.limit], trainable=False)
                                           for i, shape in enumerate(problem.variables_flattened_shape)])
                    self.grad_hist.append([tf.get_variable('grad_mom' + str(i), initializer=tf.zeros_initializer,
                                                           shape=[shape, self.limit], trainable=False)
                                           for i, shape in enumerate(problem.variables_flattened_shape)])
                    if self.normalize_with_sq_grad or self.use_noise_est:
                        self.sq_vari_hist.append([tf.get_variable('sq_vari_mom' + str(i), initializer=tf.zeros_initializer,
                                                                  shape=[shape, self.limit], trainable=False)
                                                  for i, shape in enumerate(problem.variables_flattened_shape)])
                        self.sq_grad_hist.append([tf.get_variable('sq_grad_mom' + str(i), initializer=tf.zeros_initializer,
                                                                  shape=[shape, self.limit], trainable=False)
                                                  for i, shape in enumerate(problem.variables_flattened_shape)])
                    else:
                        self.sq_vari_hist.append([0.0 for variable in problem.variables])
                        self.sq_grad_hist.append([0.0 for variable in problem.variables])

                    if self.use_dist_mv_avg:
                        self.dist_mv_avg.append([tf.get_variable('dist_mv_avg' + str(i), initializer=tf.zeros_initializer,
                                                                  shape=[shape, self.limit], trainable=False)
                                                  for i, shape in enumerate(problem.variables_flattened_shape)])
                    else:
                        self.dist_mv_avg.append([0.0 for variable in problem.variables])

                    if self.use_delta_mv_avg:
                        self.delta_mv_avg.append([tf.get_variable('delta_mv_avg' + str(i),
                                                                  initializer=tf.ones(shape=[shape, self.limit],
                                                                                      dtype=tf.float32) * 0.5,
                                                                  trainable=False)
                                                  for i, shape in enumerate(problem.variables_flattened_shape)])
                    else:
                        self.delta_mv_avg.append([0.0 for variable in problem.variables])
                    if self.learn_lr_delta:
                        self.lr_mv_avg.append([tf.get_variable('min_step_mv_avg' + str(i),
                                                               initializer=tf.random_uniform(shape=[shape, 1], minval=tf.log(1e-6), maxval=tf.log(1e-3)),
                                                               trainable=False) for i, shape in enumerate(problem.variables_flattened_shape)])
                    else:
                        self.lr_mv_avg.append([0.0 for variable in problem.variables])
                    if self.learn_momentum_base:
                        self.momentum_alpha.append([tf.get_variable('mom_alpha' + str(i), initializer=tf.zeros_initializer,
                                                              shape=[shape, 1], trainable=False)
                                                    for i, shape in enumerate(problem.variables_flattened_shape)])
    def run_init(self, args=None):
        if args is None or args['problem_index'] is None:
            problems = self.problems
            guide_steps = self.guide_step
            ops_init = self.ops_init
        else:
            index = args['problem_index']
            problems = [self.problems[index]]
            guide_steps = [self.guide_step[index]]
            ops_init = [self.ops_init[index]]
        with tf.name_scope('Init_With_Session'):
            for problem, guide_step, op_init in zip(problems, guide_steps, ops_init):
                for update in range(self.limit):
                    if self.use_guide_step:
                        if update == 0:
                            self.session.run(op_init[0])
                        else:
                            self.session.run(op_init[1])
                    else:
                        self.session.run(ops_init)
                    self.session.run(guide_step)

    def run_reset(self, index=None, optimizer=False):
        reset_ops = self.ops_reset_problem[index] if index is not None else self.ops_reset_problem
        self.session.run(reset_ops)
        if optimizer:
            self.session.run(self.ops_reset_optim)
        self.run_init({'problem_index': index})


    def normalize_values(self, history_tensor, squared_history=None, switch=0):
        epsilon = 1e-15
        with tf.name_scope('Input_Normalizer'):
            if self.normalize_with_sq_grad and squared_history is not None:
                normalized_values = tf.divide(history_tensor, tf.sqrt(squared_history) + epsilon)
            else:
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
                    diff = (max_values - min_values) + epsilon
                    # normalized_values = 2 * (history_tensor - min_values) / diff - 1.0
                    normalized_values = (history_tensor - min_values) / diff
            return normalized_values

    def sort_input(self, args):
        with tf.name_scope('Sort_Input'):
            inputs = args['inputs']
            history_ptr = args['history_ptr']
            read_ptr = history_ptr + 1
            start = tf.slice(inputs, [0, 0], [-1, read_ptr], name='start')
            end = tf.slice(inputs, [0, read_ptr], [-1, self.limit - read_ptr], name='end')
            rev_start = tf.reverse(start, [1])
            rev_end = tf.reverse(end, [1])
            return tf.concat([rev_start, rev_end], 1, name='sorted_input')

    def network(self, args=None):
        with tf.name_scope('Optimizer_network'):
            delta_lr = None
            activations = args['inputs']
            activations = layer_fc(name='in', dims=[self.network_in_dims, self.layer_width], inputs=activations,
                                   variable_list=self.optimizer_variables, activation=self.network_activation)
            for layer in range(self.hidden_layers):
                activations = layer_fc(str(layer + 1), dims=[self.layer_width, self.layer_width], inputs=activations,
                                       variable_list=self.optimizer_variables, activation=self.network_activation)
            activations = layer_fc('out', dims=[self.layer_width, self.network_out_dims], inputs=activations,
                              variable_list=self.optimizer_variables)

            lr_x_step_magnitude = tf.slice(activations, [0, 0], [-1, 10], 'x_step_mag')
            lr_x_step_magnitude = tf.nn.softmax(lr_x_step_magnitude, 1)
            lr_x_step_magnitude = tf.matmul(lr_x_step_magnitude, self.step_dist)

            lr_x_step_sign = tf.slice(activations, [0, 10], [-1, 2], 'x_step_sign')
            lr_x_step_sign = tf.nn.softmax(lr_x_step_sign, 1)
            lr_x_step_sign = tf.matmul(lr_x_step_sign, self.sign_dist)
            delta_x_step = lr_x_step_magnitude * lr_x_step_sign
            if self.min_lr is None:
                lr_grad_step_magnitude = tf.slice(activations, [0, 12], [-1, 9], 'grad_step_mag')
                lr_grad_step_magnitude = tf.nn.softmax(lr_grad_step_magnitude, 1)
                lr_grad_step_magnitude = tf.matmul(lr_grad_step_magnitude, self.lr_delta_dist)

                lr_grad_step_sign = tf.slice(activations, [0, 17], [-1, -1], 'grad_step_sign')
                lr_grad_step_sign = tf.nn.softmax(lr_grad_step_sign, 1)
                lr_grad_step_sign = tf.matmul(lr_grad_step_sign, self.sign_dist)
                delta_lr = lr_grad_step_magnitude * lr_grad_step_sign

            if self.learn_lr:
                lr_minstep = tf.slice(activations, [0, 12], [-1, 9], 'lr_min_step')
                lr_minstep = tf.nn.softmax(lr_minstep, 1)
                delta_lr = tf.matmul(lr_minstep, self.lr_dist)

            if self.learn_lr_delta:
                delta_lr = tf.sigmoid(tf.slice(activations, [0, 12], [-1, 1], 'lr_delta')) * 2.0
                # delta_lr = tf.slice(activations, [0, 12], [-1, 5], 'lr_delta')
                # delta_lr = tf.nn.softmax(delta_lr, 1)
                # delta_lr = tf.matmul(delta_lr, self.lr_delta_dist)

            if self.learn_momentum_base:
                delta_lr = tf.slice(activations, [0, 12], [-1, 1])

            if delta_lr is None:
                delta_lr = tf.constant(0.0)
            # rows = tf.shape(lr_grad_step_sign)[0]
            # max_values = tf.expand_dims(tf.reduce_max(lr_grad_step_sign, 1), 1)
            # flags = tf.equal(max_values, lr_grad_step_sign)
            # max_sign = tf.where(flags, tf.ones([rows, 2]), tf.zeros([rows, 2]))

            return [delta_x_step, delta_lr]

    def step(self, args=None):
        with tf.name_scope('mlp_x_optimizer_step'):

            problem = args['problem']
            problem_vari_hist = args['vari_hist']
            problem_grad_hist = args['grad_hist']
            problem_sq_vari_hist = args['sq_vari_hist']
            problem_sq_grad_hist = args['sq_grad_hist']
            problem_dist_mv_avg = args['dist_mv_avg']
            problem_delta_mv_avg = args['delta_mv_avg']
            problem_lr_mv_avg = args['lr_mv_avg']
            x_next = list()
            deltas_list = []
            deltas_mv_avg_next = []
            lr_mv_avg_next = []
            for (variable, variable_flat, batch_vari_hist, batch_grad_hist,
                 batch_sq_vari_hist, batch_sq_grad_hist, batch_dist_mv_avg,
                 batch_delta_mv_avg, batch_lr_mv_avg) in zip(problem.variables, problem.variables_flat,
                                                                                   problem_vari_hist, problem_grad_hist,
                                                                                   problem_sq_vari_hist, problem_sq_grad_hist,
                                                                                   problem_dist_mv_avg, problem_delta_mv_avg, problem_lr_mv_avg):

                normalized_variable_history = self.normalize_values(batch_vari_hist, batch_sq_vari_hist)
                normalized_grad_history = self.normalize_values(batch_grad_hist, batch_sq_grad_hist)
                if self.use_noise_est:
                    def noise_measure(inputs):
                        epsilon = 1e-15
                        if self.use_log_noise:
                            log_inputs = tf.log(inputs + epsilon)
                            mean_input = tf.reduce_mean(log_inputs, 1, keep_dims=True)
                            rel_noise = log_inputs - mean_input
                        else:
                            mean_input = tf.reduce_mean(inputs, 1, keep_dims=True)
                            rel_noise = inputs / (mean_input + epsilon)
                        return rel_noise
                    noise_vari_hist = noise_measure(batch_sq_vari_hist)
                    noise_grad_hist = noise_measure(batch_sq_grad_hist)
                    if self.use_log_noise:
                        normalized_noise_vari_hist = noise_vari_hist
                        normalized_noise_grad_hist = noise_grad_hist
                    else:
                        normalized_noise_vari_hist = self.normalize_values(noise_vari_hist)
                        normalized_noise_grad_hist = self.normalize_values(noise_grad_hist)
                    normalized_variable_history = tf.concat([normalized_variable_history, normalized_noise_vari_hist], 1)
                    normalized_grad_history = tf.concat([normalized_grad_history, normalized_noise_grad_hist], 1)

                if self.gradient_sign_only:
                    normalized_grad_history = tf.sign(normalized_grad_history)

                if self.gradients_only:
                    network_input = normalized_grad_history
                else:
                    network_input = tf.concat([normalized_variable_history, normalized_grad_history], 1, name='final_input')

                deltas_x, delta_lr = self.network({'inputs': network_input})
                deltas_list.append([deltas_x])

                if self.history_range is not None and self.history_range:
                    batch_variable_history_range = tf.slice(batch_vari_hist, [0, 0], [-1, self.history_range])
                else:
                    batch_variable_history_range = batch_vari_hist
                max_values = tf.reduce_max(batch_variable_history_range, axis=1, keep_dims=True)
                min_values = tf.reduce_min(batch_variable_history_range, axis=1, keep_dims=True)
                if self.use_dist_mv_avg:
                    diff = tf.reduce_mean(batch_dist_mv_avg, axis=1, keep_dims=True)
                else:
                    diff = max_values - min_values

                if self.use_delta_mv_avg:
                    delta_mv_avg_next = batch_delta_mv_avg * self.momentum_alpha + deltas_x * (1 - self.momentum_alpha)
                    deltas_mv_avg_next.append(delta_mv_avg_next)
                    deltas_x = tf.reduce_mean(delta_mv_avg_next, axis=1, keep_dims=True)

                ref = (max_values + min_values) / 2.0
                # deterministic
                # ref_points = (max_values + min_values) / 2.0
                # new_points = tf.add(variable_flat, tf.multiply(deltas, diff), 'new_points')

                # tf.where(tf.equal(tf.squeeze(x), -.47270381), tf.zeros(10), tf.ones(10))
                # random prev
                # mean = tf.multiply(deltas, diff)
                # noise = tf.random_normal([mean.shape[0].value, 1], 0, .001)
                # noisey_mean = mean * (1 + noise)

                # for same effect use .001 as multiplier for mean.
                # noisey_mean = tf.random_normal([1, 1], mean, .000001 + tf.abs(mean) * .00001)
                default_step = diff
                if self.min_lr is not None:
                    if self.min_step_max:
                        default_step = tf.maximum(diff, self.min_lr)
                    else:
                        if self.learn_lr:
                            lr = delta_lr
                        elif self.learn_lr_delta:
                            delta_lr = batch_lr_mv_avg + tf.log(delta_lr)
                            lr_mv_avg_next.append(batch_lr_mv_avg * 0.7 + delta_lr * 0.3)
                            lr = tf.exp(delta_lr) + 1e-8
                        else:
                            lr = self.min_lr
                        default_step = diff + lr
                mean = tf.multiply(deltas_x, default_step)
                new_points = tf.add(ref, mean, 'new_points')
                new_points = problem.set_shape(new_points, like_variable=variable, op_name='reshaped_new_points')
                x_next.append(new_points)
            return {'x_next': x_next, 'deltas_list': deltas_list,
                    'delta_mv_avg_next': deltas_mv_avg_next,
                    'lr_mv_avg_next': lr_mv_avg_next}

    def update_history_ops(self, args):
        problem_no = args['problem_no']
        batch_no = args['batch_no']
        batch_variables = args['batch_variable']
        batch_gradients = args['batch_gradients']
        batch_vari_hist = args['batch_vari_hist']
        batch_grad_hist = args['batch_grad_hist']
        batch_sq_vari_hist = args['batch_sq_vari_hist']
        batch_sq_grad_hist = args['batch_sq_grad_hist']
        batch_dist_mv_avg = args['batch_dist_mv_avg']
        batch_delta_mv_avg = args['batch_delta_mv_avg']
        batch_delta_mv_avg_next = args['batch_delta_mv_avg_next']
        batch_lr_mv_avg = args['batch_lr_mv_avg']
        batch_lr_mv_avg_next = args['batch_lr_mv_avg_next']
        init_ops = args['init_ops']
        history_ops = []
        momentum_alpha = self.momentum_alpha[problem_no][batch_no] if self.learn_momentum_base else self.momentum_alpha

        if init_ops:
            tiled_batch_variables = tf.tile(batch_variables, [1, self.limit])
            tiled_batch_grads = tf.tile(batch_gradients, [1, self.limit])
            history_ops.append(tf.assign(batch_vari_hist, tiled_batch_variables))
            history_ops.append(tf.assign(batch_grad_hist, tiled_batch_grads))
            if self.normalize_with_sq_grad or self.use_noise_est:
                history_ops.append(tf.assign(batch_sq_vari_hist, tf.square(tiled_batch_variables)))
                history_ops.append(tf.assign(batch_sq_grad_hist, tf.square(tiled_batch_grads)))
            if self.use_dist_mv_avg:
                with tf.control_dependencies(history_ops):
                    max = tf.reduce_max(batch_vari_hist, axis=1, keep_dims=True)
                    min = tf.reduce_min(batch_vari_hist, axis=1, keep_dims=True)
                    diff = max - min
                    tiled_diff = tf.tile(diff, [1, self.limit])
                    history_ops.append(tf.assign(batch_dist_mv_avg, tiled_diff))
        else:
            updated_vari_hist = None
            if self.use_momentums:
                # oldest_history_index = tf.cond(tf.equal(history_ptr, self.limit - 1), lambda: 0, lambda: history_ptr + 1)
                # oldest_history_slice = tf.slice(batch_grad_history, [0, oldest_history_index], [-1, 1])
                oldest_history_slice = batch_variables
                updated_vari_hist = batch_vari_hist * momentum_alpha + batch_variables * (1 - momentum_alpha)
                updated_grad_hist= batch_grad_hist * momentum_alpha + batch_gradients * (1 - momentum_alpha)
                history_ops.append(tf.assign(batch_vari_hist, updated_vari_hist))
                history_ops.append(tf.assign(batch_grad_hist, updated_grad_hist))
            else:
                updated_vari_hist = tf.concat([batch_vari_hist[:, 1:], batch_variables], axis=1)
                updated_grad_hist = tf.concat([batch_grad_hist[:, 1:], batch_gradients], axis=1)
                history_ops.append(tf.assign(batch_vari_hist, updated_vari_hist))
                history_ops.append(tf.assign(batch_grad_hist, updated_grad_hist))
            if self.normalize_with_sq_grad or self.use_noise_est:
                with tf.control_dependencies(history_ops):
                    updated_sq_vari_hist = batch_sq_vari_hist * momentum_alpha + tf.square(updated_vari_hist) * (1 - momentum_alpha)
                    updated_sq_grad_hist = batch_sq_grad_hist * momentum_alpha + tf.square(updated_grad_hist) * (1 - momentum_alpha)
                    history_ops.append(tf.assign(batch_sq_vari_hist, updated_sq_vari_hist))
                    history_ops.append(tf.assign(batch_sq_grad_hist, updated_sq_grad_hist))
            if self.use_dist_mv_avg:
                with tf.control_dependencies(history_ops):
                    max = tf.reduce_max(updated_vari_hist, axis=1, keep_dims=True)
                    min = tf.reduce_min(updated_vari_hist, axis=1, keep_dims=True)
                    diff = max - min
                    updated_batch_dist_moving_avg = batch_dist_mv_avg * momentum_alpha + diff * (1 - momentum_alpha)
                    history_ops.append(tf.assign(batch_dist_mv_avg, updated_batch_dist_moving_avg))
            if self.use_delta_mv_avg:
                history_ops.append(tf.assign(batch_delta_mv_avg, batch_delta_mv_avg_next))
            if self.learn_lr_delta:
                history_ops.append(tf.assign(batch_lr_mv_avg, batch_lr_mv_avg_next))
            # history_ops.append(tf.scatter_nd_update(batch_variables_history, indices, tf.reshape(batch_variables, [shape])))
            # history_ops.append(tf.scatter_nd_update(batch_grad_history, indices, tf.reshape(batch_gradients, [shape])))
        return history_ops

    def updates(self, args=None):
        with tf.name_scope('mlp_x_optimizer_updates'):
            x_next = args['x_next']
            problem_no = args['problem_no']
            problem = args['problem']
            problem_vari_hist = args['vari_hist']
            problem_grad_hist = args['grad_hist']
            problem_sq_vari_hist = args['sq_vari_hist']
            problem_sq_grad_hist = args['sq_grad_hist']
            problem_dist_mvg_avg = args['dist_mv_avg']
            problem_delta_mv_avg = args['delta_mv_avg']
            problem_lr_mv_avg = args['lr_mv_avg']
            update_problem_vars = args['update_problem_vars']
            init_ops = args['init_ops']

            problem_delta_mv_avg_next = [None for variable in problem.variables]
            problem_lr_mv_avg_next = [None for variable in problem.variables]
            if not init_ops:
                if self.use_delta_mv_avg:
                    problem_delta_mv_avg_next = args['delta_mv_avg_next']
                if self.learn_lr_delta:
                    problem_lr_mv_avg_next = args['lr_mv_avg_next']


            update_list = []
            if update_problem_vars:
                update_list.extend([tf.assign(variable, updated_var) for variable, updated_var in
                               zip(problem.variables, x_next)])

            flat_gradients = problem.get_gradients(x_next)
            flat_variables = [problem.flatten_input(i, variable) for i, variable in enumerate(x_next)]
            for batch_no, (variable, grads, batch_vari_hist, batch_grad_hist,
                 batch_sq_vari_hist, batch_sq_grad_hist,
                 batch_dist_mvg_avg, batch_delta_mv_avg, batch_delta_mv_avg_next, batch_lr_mv_avg, batch_lr_mv_avg_next) in enumerate(zip(flat_variables,
                                                                                                   flat_gradients,
                                                                                                   problem_vari_hist,
                                                                                                   problem_grad_hist,
                                                                                                   problem_sq_vari_hist,
                                                                                                   problem_sq_grad_hist,
                                                                                                   problem_dist_mvg_avg,
                                                                                                   problem_delta_mv_avg,
                                                                                                   problem_delta_mv_avg_next, problem_lr_mv_avg,
                                                                                                   problem_lr_mv_avg_next)):
                update_list.extend(self.update_history_ops({'problem_no': problem_no, 'batch_no': batch_no,
                                                            'batch_variable': variable, 'batch_gradients': grads,
                                                            'batch_vari_hist': batch_vari_hist, 'batch_grad_hist': batch_grad_hist,
                                                            'batch_sq_vari_hist': batch_sq_vari_hist,
                                                            'batch_sq_grad_hist': batch_sq_grad_hist,
                                                            'batch_dist_mv_avg': batch_dist_mvg_avg,
                                                            'batch_delta_mv_avg': batch_delta_mv_avg,
                                                            'batch_delta_mv_avg_next': batch_delta_mv_avg_next,
                                                            'batch_lr_mv_avg': batch_lr_mv_avg,
                                                            'batch_lr_mv_avg_next': batch_lr_mv_avg_next,
                                                            'init_ops': init_ops}))
            return update_list

    def reset_optimizer(self):
        reset = super(MlpNormHistory, self).reset_optimizer()
        return reset

    def reset_problem(self, args):
        problem = args['problem']
        problem_vari_hist = args['vari_hist']
        problem_grad_hist = args['grad_hist']
        problem_sq_vari_hist = args['sq_vari_hist']
        problem_sq_grad_hist = args['sq_grad_hist']
        problem_delta_mv_avg = args['delta_mv_avg']
        problem_lr_mv_avg = args['lr_mv_avg']
        reset = []
        reset.append(super(MlpNormHistory, self).reset_problem(problem))
        reset.append(tf.variables_initializer(problem_vari_hist, name='reset_vari_hist'))
        reset.append(tf.variables_initializer(problem_grad_hist, name='reset_grad_hist'))
        if self.normalize_with_sq_grad or self.use_noise_est:
            reset.append(tf.variables_initializer(problem_sq_vari_hist, name='reset_sq_vari_hist'))
            reset.append(tf.variables_initializer(problem_sq_grad_hist, name='reset_sq_grad_hist'))
        if self.use_delta_mv_avg:
            reset.append(tf.variables_initializer(problem_delta_mv_avg, name='reset_delta_mv_avg'))
        if self.learn_lr_delta:
            reset.append(tf.variables_initializer(problem_lr_mv_avg, name='reset_lr_mv_avg'))
        return reset

    def loss(self, args=None):
        with tf.name_scope('Problem_Loss'):
            problem = args['problem']
            variables = args['x_next'] if 'x_next' in args else problem.variables
            return problem.loss(variables)

    def run(self, args=None):
        if args['train']:
            ops_meta_step = self.ops_meta_step
        else:
            ops_meta_step = []
        start = timer()
        op_loss, pr_loss, _, _ = self.session.run([self.ops_loss, self.ops_loss_problem, ops_meta_step, self.ops_updates])
        return timer() - start, np.array(op_loss), np.array(pr_loss)

    def build(self):
        self.ops_step = []
        self.ops_updates = []
        self.ops_loss = []
        self.ops_meta_step = []
        self.ops_final_loss = 0
        self.ops_reset_problem = []
        self.ops_reset_optim = None
        self.ops_init = []
        self.ops_loss_problem = [tf.squeeze(self.loss({'problem': problem})) for problem in self.problems]
        for problem_no, (problem, vari_hist, grad_hist, sq_vari_hist,
                         sq_grad_hist, dist_mv_avg, delta_mv_avg, lr_mv_avg) in enumerate(zip(self.problems, self.vari_hist,
                                                                                   self.grad_hist, self.sq_vari_hist,
                                                                                   self.sq_grad_hist, self.dist_mv_avg,
                                                                                   self.delta_mv_avg, self.lr_mv_avg)):
            args = {'problem_no': problem_no, 'problem': problem, 'vari_hist': vari_hist, 'grad_hist': grad_hist,
                    'sq_vari_hist': sq_vari_hist, 'sq_grad_hist': sq_grad_hist,
                    'x_next': [variable.initialized_value() for variable in problem.variables],
                    'dist_mv_avg': dist_mv_avg, 'delta_mv_avg': delta_mv_avg, 'lr_mv_avg': lr_mv_avg,
                    'update_problem_vars': False, 'init_ops': True}

            init_ops = [self.updates(args)]
            args['init_ops'] = False
            if self.use_guide_step:
                init_ops.append(self.updates(args))
            self.ops_init.append(init_ops)

            loss_curr = tf.log(self.loss(args) + 1e-20)
            step = self.step(args)
            args['x_next'] = step['x_next']
            args['update_problem_vars'] = True
            args['delta_mv_avg_next'] = step['delta_mv_avg_next']
            args['lr_mv_avg_next'] = step['lr_mv_avg_next']
            updates = self.updates(args)
            loss_next = tf.log(self.loss(args) + 1e-20)
            reset = self.reset_problem(args)
            self.ops_step.append(step)
            self.ops_updates.append(updates)
            loss = tf.squeeze(loss_next - loss_curr)
            self.ops_loss.append(loss)
            self.ops_meta_step.append(self.minimize(loss))
            self.ops_reset_problem.append(reset)
        self.ops_reset_optim = self.reset_optimizer()
        self.init_saver_handle()


class GRUNormHistory(MlpNormHistory):
    unroll_len = None
    state_size = None
    hidden_states = None
    rnn = None
    gru = None
    rnn_w, rnn_b = None, None

    def __init__(self, problems, path, args):
        super(GRUNormHistory, self).__init__(problems, path, args)
        self.state_size = args['state_size']
        self.unroll_len = args['unroll_len']
        self.gru = args['gru']
        self.hidden_states = []

        # initialize for later use.
        with tf.variable_scope('optimizer_core'):
            # Formulate variables for all states as it allows to use tf.assign() for states
            def get_states(batch_size):
                state_variable = []
                if self.gru:
                    for state in self.rnn.zero_state(batch_size, tf.float32):
                        state_variable.append(tf.Variable(state, trainable=False))
                else:
                    for state_c, state_h in self.rnn.zero_state(batch_size, tf.float32):
                        state_variable.append(tf.contrib.rnn.LSTMStateTuple(tf.Variable(state_c, trainable=False),
                                                                            tf.Variable(state_h, trainable=False)))
                return tuple(state_variable)
                # return tf.Variable(self.rnn.zero_state(batch_size, tf.float32), trainable=False)
            # self.rnn = tf.contrib.rnn.GRUCell(self.state_size)
            if self.gru:
                self.rnn = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.GRUCell(self.state_size) for _ in range(2)])
            else:
                self.rnn = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LayerNormBasicLSTMCell(self.state_size) for _ in range(2)])
                # [tf.contrib.rnn.GRUCell(self.state_size) for _ in range(2)])



            with tf.variable_scope('hidden_states'):
                for problem in self.problems:
                    self.hidden_states.append([get_states(problem.get_shape(variable=variable)) for variable in
                                          problem.variables_flat])

            with tf.variable_scope('rnn_linear'):
                self.rnn_w = tf.get_variable('softmax_w', [self.state_size, self.network_out_dims])
                self.rnn_b = tf.get_variable('softmax_b', [self.network_out_dims])

            network_input = self.get_network_input(self.vari_hist[0][0], self.grad_hist[0][0], 0, self.vari_mom[0][0], self.grad_mom[0][0])
            self.network({'inputs': network_input, 'hidden_states': self.hidden_states[0][0], 'init': True})

    def network(self, args=None):
        with tf.name_scope('Optimizer_Network'):
            activations = args['inputs']
            hidden_states = args['hidden_states']
            init = args['init'] if 'init' in args else False
            if init:
                with tf.variable_scope('network'):
                    activations, hidden_states = self.rnn(activations, hidden_states)
                    rnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer_core/network')
                    self.optimizer_variables.extend(rnn_variables)
                    self.optimizer_variables.append(self.rnn_w)
                    self.optimizer_variables.append(self.rnn_b)
            else:
                with tf.variable_scope('optimizer_core/network', reuse=True):
                    activations, hidden_states = self.rnn(activations, hidden_states)

            activations = tf.add(tf.matmul(activations, self.rnn_w), self.rnn_b)

            lr_x_step_magnitude = tf.slice(activations, [0, 0], [-1, 10], 'x_step_mag')
            lr_x_step_magnitude = tf.nn.softmax(lr_x_step_magnitude, 1)
            lr_x_step_magnitude = tf.matmul(lr_x_step_magnitude, self.step_dist)

            lr_x_step_sign = tf.slice(activations, [0, 10], [-1, 2], 'x_step_sign')
            lr_x_step_sign = tf.nn.softmax(lr_x_step_sign, 1)
            lr_x_step_sign = tf.matmul(lr_x_step_sign, self.sign_dist)
            delta_x_step = lr_x_step_magnitude * lr_x_step_sign
            if self.min_lr is None:
                lr_grad_step_magnitude = tf.slice(activations, [0, 12], [-1, 5], 'grad_step_mag')
                lr_grad_step_magnitude = tf.nn.softmax(lr_grad_step_magnitude, 1)
                lr_grad_step_magnitude = tf.matmul(lr_grad_step_magnitude, self.lr_dist)

                lr_grad_step_sign = tf.slice(activations, [0, 17], [-1, -1], 'grad_step_sign')
                lr_grad_step_sign = tf.nn.softmax(lr_grad_step_sign, 1)
                lr_grad_step_sign = tf.matmul(lr_grad_step_sign, self.sign_dist)
                delta_lr = lr_grad_step_magnitude * lr_grad_step_sign
            else:
                delta_lr = tf.constant(0.0)
            return [delta_x_step, delta_lr, hidden_states]

    def get_network_input(self, variable_history, grad_history, history_ptr, vari_mom=None, grad_mom=None):

        sorted_vari_history = variable_history#self.sort_input({'inputs': variable_history,
                                               #            'history_ptr': history_ptr})
        sorted_vari_history.set_shape(shape=variable_history.get_shape())
        sorted_grad_history = grad_history#self.sort_input({'inputs': grad_history,
                                          #             'history_ptr': history_ptr})
        sorted_grad_history.set_shape(shape=grad_history.get_shape())

        if self.use_momentums:
            sorted_vari_history = tf.concat([vari_mom, sorted_vari_history], 1,
                                            name='concat_vari_mom')
            sorted_grad_history = tf.concat([grad_mom, sorted_grad_history], 1,
                                            name='concat_grad_mom')

        normalized_variable_history = self.normalize_values(sorted_vari_history)
        normalized_grad_history = self.normalize_values(sorted_grad_history)

        if self.gradient_sign_only:
            normalized_grad_history = tf.sign(normalized_grad_history)

        if self.gradients_only:
            network_input = normalized_grad_history
        else:
            network_input = tf.concat([normalized_variable_history, normalized_grad_history], 1, name='final_input')
        return network_input

    def step(self, args=None):
        with tf.name_scope('mlp_x_optimizer_step'):
            problem_no = args['problem_no']
            problem = args['problem']
            problem_variable_history = args['variable_history']
            problem_grad_history = args['grad_history']
            history_ptr = args['history_ptr']
            problem_vari_mom = args['vari_mom']
            problem_grad_mom = args['grad_mom']
            problem_hidden_states = args['hidden_states']
            x_next = list()
            deltas_list = []
            def update(itr, loss, problem_variables, problem_variable_history, problem_grad_history, problem_vari_mom,
                       problem_grad_mom, problem_hidden_states):
                for batch_no, (variable, batch_variable_history, batch_variable_grad_history, batch_vari_mom, batch_grad_mom,
                               batch_hidden_states) in enumerate(zip(problem_variables,
                                                                           problem_variable_history, problem_grad_history,
                                                                           problem_vari_mom, problem_grad_mom, problem_hidden_states)):
                    network_input = self.get_network_input(batch_variable_history, batch_variable_grad_history,
                                                           history_ptr, batch_vari_mom, batch_grad_mom)
                    deltas_x, deltas_g, problem_hidden_states[batch_no] = self.network({'inputs': network_input,
                                                                            'hidden_states': batch_hidden_states})
                    # if self.history_range is not None and self.history_range:
                    #     batch_variable_history_range = tf.slice(sorted_variable_history, [0, 0],
                    #                                             [-1, self.history_range])
                    # else:
                    if self.use_momentums:
                        batch_variable_history_range = tf.concat([batch_vari_mom, batch_variable_history], axis=1)
                    else:
                        batch_variable_history_range = batch_variable_history
                    max_values = tf.reduce_max(batch_variable_history_range, 1)
                    min_values = tf.reduce_min(batch_variable_history_range, 1)
                    max_values = tf.expand_dims(max_values, 1)
                    min_values = tf.expand_dims(min_values, 1)
                    diff = max_values - min_values
                    ref = (max_values + min_values) / 2.0

                    default_step = diff
                    if self.min_lr is not None:
                        if self.min_step_max:
                            default_step = tf.maximum(diff, self.min_lr)
                        else:
                            default_step = diff + self.min_lr
                    mean = tf.multiply(deltas_x, default_step) + deltas_g
                    problem_variables[batch_no] = tf.add(ref, mean, 'new_points')
                original_shaped_variables = [problem.set_shape(flat_variable, i=variable_no, op_name='reshape_variable') for variable_no, flat_variable in enumerate(problem_variables)]
                gradients = self.get_preprocessed_gradients(problem, original_shaped_variables)
                for i, (batch_variable, batch_variable_history, batch_variable_grad_history, batch_vari_mom, batch_grad_mom, gradient) \
                        in enumerate(zip(problem_variables, problem_variable_history, problem_grad_history, problem_vari_mom, problem_grad_mom, gradients)):
                    problem_variable_history[i] = tf.concat([batch_variable_history[:, 1:], batch_variable], axis=1)
                    problem_grad_history[i] = tf.concat([batch_variable_grad_history[:, 1:], gradient], axis=1)
                    if self.use_momentums:
                        problem_vari_mom[i] = batch_vari_mom * self.momentum_alpha + batch_variable * self.momentum_alpha_inv
                        problem_grad_mom[i] = batch_grad_mom * self.momentum_alpha + gradient * self.momentum_alpha_inv
                loss = tf.squeeze(loss + self.loss({'problem': problem, 'x_next': original_shaped_variables}))
                return itr + 1, loss, problem_variables, problem_variable_history, problem_grad_history, problem_vari_mom, problem_grad_mom, problem_hidden_states


            _, _, variables_next, variable_history_next, grad_history_next, problem_vari_mom_next, \
            problem_grad_mom_next, hidden_states_next = tf.while_loop(
                cond=lambda t, *_: t < self.unroll_len,
                body=update,
                loop_vars=([0, tf.constant(0.0), problem.variables_flat, problem_variable_history, problem_grad_history,
                            problem_vari_mom, problem_grad_mom, problem_hidden_states]),
                parallel_iterations=1,
                swap_memory=True,
                name="unroll")
            # _, _, variables_next, variable_history_next, grad_history_next, problem_vari_mom_next, problem_grad_mom_next, hidden_states_next = \
            #     update(0, 0, problem.variables_flat, problem_variable_history, problem_grad_history,
            #                 problem_vari_mom, problem_grad_mom, problem_hidden_states)
            for variable_next_flat, variable_orig in zip(variables_next, problem.variables):
                new_points = problem.set_shape(variable_next_flat, like_variable=variable_orig, op_name='reshaped_new_points')
                x_next.append(new_points)
            return {'x_next': x_next, 'deltas': deltas_list, 'variable_history_next': variable_history_next,
                    'grad_history_next': grad_history_next, 'vari_mom_next': problem_vari_mom_next,
                    'grad_mom_next': problem_grad_mom_next, 'hidden_states_next': hidden_states_next}

    def update_history_ops(self, args):
        batch_variables = args['batch_variable']
        batch_gradients = args['batch_gradients']
        batch_variables_history = args['batch_variables_hitory']
        batch_grad_history = args['batch_grad_history']
        history_ptr = args['history_ptr']
        batch_vari_mom = args['batch_vari_mom']
        batch_grad_mom = args['batch_grad_mom']
        init_ops = args['init_ops']
        variable_history_next = args['variable_history_next']
        grad_history_next = args['grad_history_next']
        vari_mom_next = args['vari_mom_next']
        grad_mom_next = args['grad_mom_next']
        mom_ops = []
        history_ops = []
        shape = batch_variables.shape[0].value
        indices = [[i, history_ptr] for i in range(shape)]

        if self.use_momentums:
            # oldest_history_index = tf.cond(tf.equal(history_ptr, self.limit - 1), lambda: 0, lambda: history_ptr + 1)
            # oldest_history_slice = tf.slice(batch_grad_history, [0, oldest_history_index], [-1, 1])
            oldest_history_slice = batch_variables
            if init_ops:
                updated_grad_mom = tf.tile(batch_gradients, [1, self.limit])
                updated_vari_mom = tf.tile(batch_variables, [1, self.limit])
            else:
                updated_grad_mom = grad_mom_next
                updated_vari_mom = vari_mom_next
            mom_ops.append(tf.assign(batch_grad_mom, updated_grad_mom))
            mom_ops.append(tf.assign(batch_vari_mom, updated_vari_mom))

        with tf.control_dependencies(mom_ops):
            if init_ops:
                updated_batch_variables_history = tf.concat([batch_variables_history[:, 1:], batch_variables], axis=1)
                updated_batch_grad_history = tf.concat([batch_grad_history[:, 1:], batch_gradients], axis=1)
                history_ops.append(tf.assign(batch_variables_history, updated_batch_variables_history))
                history_ops.append(tf.assign(batch_grad_history, updated_batch_grad_history))
            else:
                history_ops.append(tf.assign(batch_variables_history, variable_history_next))
                history_ops.append(tf.assign(batch_grad_history, grad_history_next))
        return history_ops

    def updates(self, args=None):
        with tf.name_scope('mlp_x_optimizer_updates'):
            x_next = args['x_next']
            problem = args['problem']
            problem_variables_history = args['variable_history']
            problem_grad_history = args['grad_history']
            problem_vari_mom = args['vari_mom']
            problem_grad_mom = args['grad_mom']
            history_ptr = args['history_ptr']
            init_ops = args['init_ops']
            problem_hidden_states = args['hidden_states']

            # Since we use *_next variables for ops other than init, no need to increment history_ptr
            if init_ops:
                update_list = [tf.cond(history_ptr < self.limit - 1,
                                       lambda: tf.assign_add(history_ptr, 1),
                                       lambda: tf.assign(history_ptr, 0))]
                problem_variable_history_next = [None for variable in args['variable_history']]
                problem_grad_history_next = [None for variable in args['grad_history']]
                problem_vari_mom_next = [None for variable in args['vari_mom']]
                problem_grad_mom_next = [None for variable in args['grad_mom']]
            else:
                problem_hidden_states_next = args['hidden_states_next']
                problem_variable_history_next = args['variable_history_next']
                problem_grad_history_next = args['grad_history_next']
                problem_vari_mom_next = args['vari_mom_next']
                problem_grad_mom_next = args['grad_mom_next']
                update_list = [tf.assign(hidden_state, hidden_state_next) for hidden_state, hidden_state_next in
                               # zip(self.hidden_states[0], problem_hidden_states_next)]
                               zip(nest.flatten(problem_hidden_states), nest.flatten(problem_hidden_states_next))]

            with tf.control_dependencies(update_list):
                if not init_ops:
                    update_list.extend([tf.assign(variable, updated_var) for variable, updated_var in
                                   zip(problem.variables, x_next)])
                flat_gradients = problem.get_gradients(x_next)
                flat_variables = [problem.flatten_input(i, variable) for i, variable in enumerate(x_next)]
                for variable, grads, batch_variable_history, batch_grad_history, batch_vari_mom, \
                    batch_grad_mom, batch_variable_history_next, batch_grad_history_next, \
                    batch_vari_mom_next, batch_grad_mom_next in zip(flat_variables, flat_gradients, problem_variables_history,
                                                                                              problem_grad_history,
                                                                                              problem_vari_mom,
                                                                                              problem_grad_mom,
                                                                                              problem_variable_history_next,
                                                                                              problem_grad_history_next,
                                                                                              problem_vari_mom_next, problem_grad_mom_next,
                                                                                              ):
                    update_list.extend(self.update_history_ops({'batch_variable': variable, 'batch_gradients': grads,
                                                                'batch_variables_hitory': batch_variable_history,
                                                                'batch_grad_history': batch_grad_history,
                                                                'history_ptr': history_ptr, 'batch_vari_mom': batch_vari_mom,
                                                                'batch_grad_mom': batch_grad_mom, 'init_ops': init_ops,
                                                                'variable_history_next': batch_variable_history_next,
                                                                'grad_history_next': batch_grad_history_next,
                                                                'vari_mom_next': batch_vari_mom_next,
                                                                'grad_mom_next': batch_grad_mom_next}))
            return update_list

    def reset_problem(self, args):
        reset_ops = super(GRUNormHistory, self).reset_problem(args)
        hidden_states = args['hidden_states']
        reset_ops.append(tf.variables_initializer(nest.flatten(hidden_states), name='reset_states'))
        return reset_ops

    def build(self):
        self.ops_step = []
        self.ops_updates = []
        self.ops_loss = []
        self.ops_meta_step = []
        self.ops_final_loss = 0
        self.ops_reset_problem = []
        self.ops_reset_optim = None
        self.ops_init = []
        self.ops_loss_problem = [tf.squeeze(self.loss({'problem': problem})) for problem in self.problems]
        for problem_no, (problem, variable_history, grad_sign_history, history_ptr, vari_mom, grad_mom, hidden_states) in enumerate(zip(self.problems,
                                                                                                 self.vari_hist,
                                                                                       self.grad_hist,
                                                                                       self.history_ptr,
                                                                                       self.vari_mom,
                                                                                       self.grad_mom,
                                                                                        self.hidden_states)):
            args = {'problem_no': problem_no, 'problem': problem, 'variable_history': variable_history,
                    'grad_history': grad_sign_history, 'history_ptr': history_ptr,
                    'x_next': [variable.initialized_value() for variable in problem.variables],
                    'init_ops': True, 'vari_mom': vari_mom, 'grad_mom': grad_mom, 'hidden_states': hidden_states}
            self.ops_init.append(self.updates(args))
            loss_curr = tf.log(self.loss(args) + 1e-20)
            step = self.step(args)
            args['x_next'] = step['x_next']
            args['variable_history_next'] = step['variable_history_next']
            args['grad_history_next'] = step['grad_history_next']
            args['vari_mom_next'] = step['vari_mom_next']
            args['grad_mom_next'] = step['grad_mom_next']
            args['hidden_states_next'] = step['hidden_states_next']
            args['init_ops'] = False
            updates = self.updates(args)
            loss_next = tf.log(self.loss(args) + 1e-20)
            reset = self.reset_problem(args)
            self.ops_step.append(step)
            self.ops_updates.append(updates)
            loss = tf.squeeze(loss_next - loss_curr)
            self.ops_loss.append(loss)
            self.ops_meta_step.append(self.minimize(loss))
            self.ops_reset_problem.append(reset)
        self.ops_reset_optim = self.reset_optimizer()
        self.init_saver_handle()

class L2L2(Meta_Optimizer):

    grad_moving_avg = None
    norm_grad_moving_avg = None
    avg_sq_grad_moving_avg = None
    relative_log_grad_mag = None
    relative_learning_rate = None
    learning_rate_moving_avg = None

    network_in_dims = None
    num_time_scales = None
    hidden_states = None

    def __init__(self, problems, path, args):
        super(L2L2, self).__init__(problems, path, args)
        self.network_in_dims = args['network_in_dims']
        self.network_out_dims = args['network_out_dims']
        self.state_size = args['state_size']
        self.unroll_len = args['unroll_len']
        self.num_time_scales = args['num_time_scales']
        self.grad_moving_avg = []
        self.norm_grad_moving_avg = []
        self.avg_sq_grad_moving_avg = []
        self.relative_log_grad_mag = []
        self.relative_learning_rate = []
        self.learning_rate_moving_avg = []
        self.hidden_states = []

        # initialize for later use.
        with tf.variable_scope('optimizer_core'):
            # Formulate variables for all states as it allows to use tf.assign() for states
            def get_states(batch_size):
                state_variable = []
                for state in self.rnn.zero_state(batch_size, tf.float32):
                    state_variable.append(tf.Variable(state, trainable=False))
                return tuple(state_variable)
                # return tf.Variable(self.rnn.zero_state(batch_size, tf.float32), trainable=False)

            # self.rnn = tf.contrib.rnn.GRUCell(self.state_size)
            self.rnn = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.state_size) for _ in range(2)])
            # [tf.contrib.rnn.GRUCell(self.state_size) for _ in range(2)])

            with tf.variable_scope('hidden_states'):
                for problem in self.problems:
                    self.hidden_states.append([get_states(problem.get_shape(variable=variable)) for variable in
                                               problem.variables_flat])

            with tf.variable_scope('rnn_linear'):
                self.rnn_w = tf.get_variable('softmax_w', [self.state_size, self.network_out_dims])
                self.rnn_b = tf.get_variable('softmax_b', [self.network_out_dims])

            network_input = tf.zeros([self.problems[0].variables[0].get_shape().as_list()[0], self.network_in_dims], dtype=tf.float32)
            self.network({'inputs': network_input, 'hidden_states': self.hidden_states[0][0], 'init': True})

        for i, problem in enumerate(problems):
            init_problem_variables = [variable.initialized_value() for variable in problem.variables]
            gradients = self.get_preprocessed_gradients(problem, variables=init_problem_variables)
            grad_moving_avg = []
            avg_sq_grad_moving_avg = []
            norm_grad_moving_avg = []
            relative_log_grad_mag = []
            relative_learning_rate = []
            learning_rate_moving_avg = []
            for j, variable_gradient in enumerate(gradients):
                variables_tiled = tf.tile(variable_gradient, [1, self.num_time_scales])
                variables_tiled_sq = tf.square(variables_tiled)
                variables_normalized = tf.divide(variables_tiled, tf.sqrt(variables_tiled_sq))
                grad_moving_avg.append(tf.get_variable(name='norm_grad_avg_' + str(i) + '_' + str(j), initializer=variables_tiled))
                avg_sq_grad_moving_avg.append(tf.get_variable(name='avg_sq_grad_moving_avg_' + str(i) + '_' + str(j), initializer=variables_tiled_sq))
                norm_grad_moving_avg.append(tf.get_variable(name='norm_grad_moving_avg_' + str(i) + '_' + str(j), initializer=variables_normalized))
                relative_log_grad_mag.append(tf.get_variable(name='relative_log_grad_mag_' + str(i) + '_' + str(j), initializer=tf.zeros_initializer, shape=[variable_gradient.get_shape()[0], self.num_time_scales]))
                relative_learning_rate.append(tf.get_variable(name='realtive_learning_rate' + str(i) + '_' + str(j), initializer=tf.zeros_initializer, shape=[variable_gradient.get_shape()[0], 1]))
                learning_rate_moving_avg.append(tf.get_variable(name='variable_learning_rate' + str(i) + '_' + str(j), initializer=tf.ones([variable_gradient.get_shape()[0], 1]) * tf.log(1e-6)))
            self.grad_moving_avg.append(grad_moving_avg)
            self.avg_sq_grad_moving_avg.append(avg_sq_grad_moving_avg)
            self.norm_grad_moving_avg.append(norm_grad_moving_avg)
            self.relative_log_grad_mag.append(relative_log_grad_mag)
            self.relative_learning_rate.append(relative_learning_rate)
            self.learning_rate_moving_avg.append(learning_rate_moving_avg)


    def network(self, args=None):
        with tf.name_scope('Optimizer_Network'):
            activations = args['inputs']
            hidden_states = args['hidden_states']
            init = args['init'] if 'init' in args else False
            if init:
                with tf.variable_scope('network'):
                    activations, hidden_states = self.rnn(activations, hidden_states)
                    rnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer_core/network')
                    self.optimizer_variables.extend(rnn_variables)
                    self.optimizer_variables.append(self.rnn_w)
                    self.optimizer_variables.append(self.rnn_b)
            else:
                with tf.variable_scope('optimizer_core/network', reuse=True):
                    activations, hidden_states = self.rnn(activations, hidden_states)

            activations = tf.add(tf.matmul(activations, self.rnn_w), self.rnn_b)
            variable_direction, attention_direction, delta_learning_rate, beta_grad, beta_scale = tf.expand_dims(activations[:, 0], 1), \
                                                                                                  tf.expand_dims(activations[:, 1], 1), \
                                                                                                  tf.expand_dims(activations[:, 2], 1), \
                                                                                                  tf.expand_dims(activations[:, 3], 1), \
                                                                                                  tf.expand_dims(activations[:, 4], 1)
        return variable_direction, attention_direction, delta_learning_rate, beta_grad, beta_scale, hidden_states

    def step(self, args=None):
        with tf.name_scope('mlp_x_optimizer_step'):
            problem_no = args['problem_no']
            problem = args['problem']
            problem_grad_moving_avg = args['grad_moving_avg']
            problem_avg_sq_grad_moving_avg = args['avg_sq_grad_moving_avg']
            problem_norm_grad_moving_avg = args['norm_grad_moving_avg']
            problem_relative_log_grad_mag = args['relative_log_grad_mag']
            problem_relative_learning_rate = args['relative_learning_rate']
            problem_learning_rate_moving_avg = args['learning_rate_moving_avg']
            problem_hidden_states = args['hidden_states']
            total_varialbes = 0
            for variable in problem.variables_flat:
                total_varialbes += variable.get_shape().as_list()[0]
            x_next = list()

            def get_beta_matrices(beta):
                beta_sigmoid = tf.reshape(tf.sigmoid(beta), shape=[beta.get_shape().as_list()[0], 1])
                def time_scale_entry(time_scale):
                    return tf.pow(beta_sigmoid, tf.pow(2.0, time_scale))
                beta_matrix = time_scale_entry(0)
                for time_scale in range(self.num_time_scales)[1:]:
                    beta_matrix = tf.concat([beta_matrix, time_scale_entry(time_scale)], axis=1)
                #tf_beta_matrix = tf.convert_to_tensor(beta_matrix, dtype=tf.float32)#.constant(beta_matrix, shape=[tf.shape(beta)[0], self.num_time_scales], dtype=tf.float32)
                return [beta_matrix, 1.0 - beta_matrix]

            def update(itr, loss, problem_variables, problem_grad_moving_avg, problem_avg_sq_grad_moving_avg,
                       problem_norm_grad_moving_avg, problem_relative_log_grad_mag,
                       problem_relative_learning_rate, problem_learning_rate_moving_avg,
                       problem_hidden_states):
                attention_variables = []
                betas_grad, betas_scale = [], []
                learning_rates = []
                learning_rates_sum = 0.0
                for i, (variable, batch_norm_grad_moving_avg, batch_relative_log_grad_mag, batch_relative_learning_rate,
                               batch_learning_rate_moving_avg, batch_hidden_states) \
                                                    in enumerate(zip(problem_variables, problem_norm_grad_moving_avg,
                                                                     problem_relative_log_grad_mag,
                                                                     problem_relative_learning_rate,
                                                                     problem_learning_rate_moving_avg,
                                                                     problem_hidden_states)):
                    network_inputs = tf.concat([batch_norm_grad_moving_avg, batch_relative_log_grad_mag, batch_relative_learning_rate], axis=1)
                    variable_direction, attention_direction, delta_learning_rate, \
                    beta_grad, beta_scale, problem_hidden_states[i] = self.network({'inputs': network_inputs, 'hidden_states': batch_hidden_states})
                    number_variables = variable.get_shape().as_list()[0]

                    learning_rate = batch_learning_rate_moving_avg + delta_learning_rate
                    problem_learning_rate_moving_avg[i] = .7 * batch_learning_rate_moving_avg + .3 * learning_rate
                    learning_rates.append(learning_rate)
                    learning_rates_sum += tf.reduce_sum(learning_rate, axis=0)

                    variable_delta = tf.exp(learning_rate) * variable_direction / tf.divide(
                        tf.norm(variable_direction, ord='euclidean'), number_variables)
                    attention_delta = tf.exp(learning_rate) * variable_direction / tf.divide(
                        tf.norm(attention_direction, ord='euclidean'), number_variables)

                    attention_variables.append(variable + variable_delta + attention_delta)
                    problem_variables[i] = variable + variable_delta
                    betas_grad.append(get_beta_matrices(beta_grad))
                    betas_scale.append(get_beta_matrices(beta_scale))

                original_shaped_variables = [problem.set_shape(flat_variable, i=variable_no, op_name='reshape_variable') for variable_no, flat_variable in enumerate(attention_variables)]
                gradients = self.get_preprocessed_gradients(problem, original_shaped_variables)
                average_learning_rate = learning_rates_sum / total_varialbes

                for i, (batch_variable, batch_grad_moving_avg, batch_avg_sq_grad_moving_avg,
                        batch_norm_grad_moving_avg, batch_gradient, beta_grad, beta_scale, learning_rate) \
                        in enumerate(zip(problem_variables, problem_grad_moving_avg, problem_avg_sq_grad_moving_avg,
                                         problem_norm_grad_moving_avg, gradients, betas_grad, betas_scale, learning_rates)):
                    problem_grad_moving_avg[i] = batch_grad_moving_avg * beta_grad[0] + batch_gradient * beta_grad[1]
                    problem_avg_sq_grad_moving_avg[i] = batch_avg_sq_grad_moving_avg * beta_scale[0] + tf.square(problem_grad_moving_avg[i]) * beta_scale[1]
                    problem_norm_grad_moving_avg[i] = problem_grad_moving_avg[i] / tf.sqrt(problem_avg_sq_grad_moving_avg[i])
                    log_avg_sq_grad_moving_avg = tf.log(problem_avg_sq_grad_moving_avg[i])
                    mean_avg_sq_grad_moving_avg = tf.reduce_mean(log_avg_sq_grad_moving_avg, axis=1, keep_dims=True)
                    problem_relative_log_grad_mag[i] = log_avg_sq_grad_moving_avg - mean_avg_sq_grad_moving_avg
                    problem_relative_learning_rate[i] = learning_rate - average_learning_rate
                loss = tf.squeeze(loss + self.loss({'problem': problem, 'x_next': original_shaped_variables}))

                return itr + 1, loss, problem_variables, problem_grad_moving_avg, problem_avg_sq_grad_moving_avg, \
                       problem_norm_grad_moving_avg, problem_relative_log_grad_mag, problem_relative_learning_rate, \
                       problem_learning_rate_moving_avg, problem_hidden_states


            _, _, problem_variables_next, problem_grad_moving_avg_next, \
            problem_avg_sq_grad_moving_avg_next, problem_norm_grad_moving_avg_next,\
            problem_relative_log_grad_mag_next, problem_relative_learning_rate_next,\
            problem_learning_rate_moving_avg_next, problem_hidden_states_next = tf.while_loop(
                cond=lambda t, *_: t < self.unroll_len,
                body=update,
                loop_vars=([0, tf.constant(0.0), problem.variables_flat, problem_grad_moving_avg,
                            problem_avg_sq_grad_moving_avg, problem_norm_grad_moving_avg,
                            problem_relative_log_grad_mag, problem_relative_learning_rate,
                            problem_learning_rate_moving_avg, problem_hidden_states]),
                parallel_iterations=1,
                swap_memory=True,
                name="unroll")
            # _, _, problem_variables_next, problem_grad_moving_avg_next, \
            # problem_avg_sq_grad_moving_avg_next, problem_norm_grad_moving_avg_next, \
            # problem_relative_log_grad_mag_next, problem_relative_learning_rate_next, \
            # problem_learning_rate_moving_avg_next, problem_hidden_states_next = update(0, tf.constant(0.0), problem.variables_flat, problem_grad_moving_avg,
            #             problem_avg_sq_grad_moving_avg, problem_norm_grad_moving_avg,
            #             problem_relative_log_grad_mag, problem_relative_learning_rate,
            #             problem_learning_rate_moving_avg, problem_hidden_states)
            for variable_next_flat, variable_orig in zip(problem_variables_next, problem.variables):
                new_points = problem.set_shape(variable_next_flat, like_variable=variable_orig, op_name='reshaped_new_points')
                x_next.append(new_points)
            return {'x_next': x_next,
                    'grad_moving_avg_next': problem_grad_moving_avg_next,
                    'avg_sq_grad_moving_avg_next': problem_avg_sq_grad_moving_avg_next,
                    'norm_grad_moving_avg_next': problem_norm_grad_moving_avg_next,
                    'relative_log_grad_mag_next': problem_relative_log_grad_mag_next,
                    'relative_learning_rate_next': problem_relative_learning_rate_next,
                    'learning_rate_moving_avg_next': problem_learning_rate_moving_avg_next,
                    'hidden_states_next': problem_hidden_states_next}

    def run_reset(self, index=None, optimizer=False):
        reset_ops = self.ops_reset_problem[index] if index is not None else self.ops_reset_problem
        self.session.run(reset_ops)
        if optimizer:
            self.session.run(self.ops_reset_optim)

    def run_init(self, args=None):
        return

    def updates(self, args=None):
        with tf.name_scope('mlp_x_optimizer_updates'):
            problem = args['problem']
            problem_grad_moving_avg = args['grad_moving_avg']
            problem_avg_sq_grad_moving_avg = args['avg_sq_grad_moving_avg']
            problem_norm_grad_moving_avg = args['norm_grad_moving_avg']
            problem_relative_log_grad_mag = args['relative_log_grad_mag']
            problem_relative_learning_rate = args['relative_learning_rate']
            problem_learning_rate_moving_avg = args['learning_rate_moving_avg']
            problem_hidden_states = args['hidden_states']

            x_next = args['x_next']
            problem_grad_moving_avg_next = args['grad_moving_avg_next']
            problem_avg_sq_grad_moving_avg_next = args['avg_sq_grad_moving_avg_next']
            problem_norm_grad_moving_avg_next = args['norm_grad_moving_avg_next']
            problem_relative_log_grad_mag_next = args['relative_log_grad_mag_next']
            problem_relative_learning_rate_next = args['relative_learning_rate_next']
            problem_learning_rate_moving_avg_next = args['learning_rate_moving_avg_next']
            problem_hidden_states_next = args['hidden_states_next']

            update_list = [tf.assign(hidden_state, hidden_state_next) for hidden_state, hidden_state_next in
                           # zip(self.hidden_states[0], problem_hidden_states_next)]
                           zip(nest.flatten(problem_hidden_states), nest.flatten(problem_hidden_states_next))]

            with tf.control_dependencies(update_list):
                update_list.extend([tf.assign(variable, updated_var) for variable, updated_var in
                               zip(problem.variables, x_next)])
                flat_gradients = problem.get_gradients(x_next)
                flat_variables = [problem.flatten_input(i, variable) for i, variable in enumerate(x_next)]
                for (variable, grads, batch_grad_moving_avg, batch_grad_moving_avg_next,
                    batch_avg_sq_grad_moving_avg, batch_avg_sq_grad_moving_avg_next,
                    batch_norm_grad_moving_avg, batch_norm_grad_moving_avg_next,
                    batch_relative_log_grad_mag, batch_relative_log_grad_mag_next,
                    batch_relative_learning_rate, batch_relative_learning_rate_next,
                    batch_learning_rate_moving_avg, batch_learning_rate_moving_avg_next) in zip(flat_variables, flat_gradients,
                                                                    problem_grad_moving_avg, problem_grad_moving_avg_next,
                                                                    problem_avg_sq_grad_moving_avg, problem_avg_sq_grad_moving_avg_next,
                                                                    problem_norm_grad_moving_avg, problem_norm_grad_moving_avg_next,
                                                                    problem_relative_log_grad_mag, problem_relative_log_grad_mag_next,
                                                                    problem_relative_learning_rate, problem_relative_learning_rate_next,
                                                                    problem_learning_rate_moving_avg, problem_learning_rate_moving_avg_next):
                    update_list.append(tf.assign(batch_grad_moving_avg, batch_grad_moving_avg_next))
                    update_list.append(tf.assign(batch_avg_sq_grad_moving_avg, batch_avg_sq_grad_moving_avg_next))
                    update_list.append(tf.assign(batch_norm_grad_moving_avg, batch_norm_grad_moving_avg_next))
                    update_list.append(tf.assign(batch_relative_log_grad_mag, batch_relative_log_grad_mag_next))
                    update_list.append(tf.assign(batch_relative_learning_rate, batch_relative_learning_rate_next))
                    update_list.append(tf.assign(batch_learning_rate_moving_avg, batch_learning_rate_moving_avg_next))
            return update_list

    def reset_problem(self, args):
        reset_ops = super(L2L2, self).reset_problem(args['problem'])
        hidden_states = args['hidden_states']
        avg_sq_grad_moving_avg = args['avg_sq_grad_moving_avg']
        norm_grad_moving_avg = args['norm_grad_moving_avg']
        relative_log_grad_mag = args['relative_log_grad_mag']
        relative_learning_rate = args['relative_learning_rate']
        learning_rate_moving_avg = args['learning_rate_moving_avg']
        reset_ops.append(tf.variables_initializer(nest.flatten(hidden_states), name='reset_states'))
        reset_ops.append(tf.variables_initializer(avg_sq_grad_moving_avg))
        reset_ops.append(tf.variables_initializer(norm_grad_moving_avg))
        reset_ops.append(tf.variables_initializer(relative_log_grad_mag))
        reset_ops.append(tf.variables_initializer(relative_learning_rate))
        reset_ops.append(tf.variables_initializer(learning_rate_moving_avg))
        return reset_ops

    def run(self, args=None):
        if args['train']:
            ops_meta_step = self.ops_meta_step
        else:
            ops_meta_step = []
        start = timer()
        op_loss, pr_loss, _, _ = self.session.run([self.ops_loss, self.ops_loss_problem, ops_meta_step, self.ops_updates])
        return timer() - start, np.array(op_loss), np.array(pr_loss)

    def loss(self, args=None):
        with tf.name_scope('Problem_Loss'):
            problem = args['problem']
            variables = args['x_next'] if 'x_next' in args else problem.variables
            return problem.loss(variables)

    def build(self):
        self.ops_step = []
        self.ops_updates = []
        self.ops_loss = []
        self.ops_meta_step = []
        self.ops_final_loss = 0
        self.ops_reset_problem = []
        self.ops_reset_optim = None
        self.ops_init = []
        self.ops_loss_problem = [tf.squeeze(self.loss({'problem': problem})) for problem in self.problems]
        for problem_no, (problem, problem_grad_moving_avg, problem_avg_sq_grad_moving_avg, problem_norm_grad_moving_avg,
                         problem_relative_log_grad_mag, problem_relative_learning_rate,
                         problem_learning_rate_moving_avg, problem_hidden_states) in enumerate(zip(self.problems,
                                                                    self.grad_moving_avg,
                                                                    self.avg_sq_grad_moving_avg,
                                                                    self.norm_grad_moving_avg,
                                                                    self.relative_log_grad_mag,
                                                                    self.relative_learning_rate,
                                                                    self.learning_rate_moving_avg,
                                                                            self.hidden_states)):

            args = {'problem_no': problem_no, 'problem': problem, 'grad_moving_avg': problem_grad_moving_avg,
                    'avg_sq_grad_moving_avg': problem_avg_sq_grad_moving_avg,
                    'norm_grad_moving_avg': problem_norm_grad_moving_avg,
                    'relative_log_grad_mag': problem_relative_log_grad_mag,
                    'relative_learning_rate': problem_relative_learning_rate,
                    'learning_rate_moving_avg': problem_learning_rate_moving_avg,
                    'hidden_states': problem_hidden_states}
            loss_curr = tf.log(self.loss(args) + 1e-20)
            step = self.step(args)
            args['x_next'] = step['x_next']
            args['grad_moving_avg_next'] = step['grad_moving_avg_next']
            args['avg_sq_grad_moving_avg_next'] = step['avg_sq_grad_moving_avg_next']
            args['norm_grad_moving_avg_next'] = step['norm_grad_moving_avg_next']
            args['relative_log_grad_mag_next'] = step['relative_log_grad_mag_next']
            args['relative_learning_rate_next'] = step['relative_learning_rate_next']
            args['learning_rate_moving_avg_next'] = step['learning_rate_moving_avg_next']
            args['hidden_states_next'] = step['hidden_states_next']
            updates = self.updates(args)
            loss_next = tf.log(self.loss(args) + 1e-20)
            reset = self.reset_problem(args)
            self.ops_step.append(step)
            self.ops_updates.append(updates)
            loss = tf.squeeze(loss_next - loss_curr)
            self.ops_loss.append(loss)
            self.ops_meta_step.append(self.minimize(loss))
            self.ops_reset_problem.append(reset)
        self.ops_reset_optim = self.reset_optimizer()
        self.init_saver_handle()


class MlpHistoryGradNormMinStep(MlpNormHistory):

    sign_dist = None
    lr_dist = None

    def __init__(self, problems, path, args):
        self.sign_dist = tf.Variable(tf.constant([-1.0, 1.0], shape=[2, 1], dtype=tf.float32),
                                     name='sign_dist')
        self.lr_dist = tf.Variable(tf.constant([.1, .05, .001, .0005, 0], shape=[5, 1], dtype=tf.float32),
                                   name='grad_dist')
        args['input_dim'] = args['limit']
        args['output_dim'] = 19
        args['step_dist_dims'] = 10
        args['step_dist_minval'] = 0
        args['step_dist_maxval'] = 1.0
        super(MlpHistoryGradNormMinStep, self).__init__(problems, path, args)


    def network(self, args=None):
        with tf.name_scope('mlp_x_optimizer_network'):
            _, variable_grad_history = args['inputs']
            normalized_grad_history = self.normalize_values(variable_grad_history)
            final_var_grad_history = self.sort_input({'inputs': normalized_grad_history,
                                                      'history_ptr': args['history_ptr']})
            final_input = final_var_grad_history
            activations = final_input
            activations = super(MlpNormHistory, self).network({'preprocessed_gradient': activations})[0]

            lr_x_step_magnitude = tf.slice(activations, [0, 0], [-1, 10], 'x_step_mag')
            lr_x_step_magnitude = tf.nn.softmax(lr_x_step_magnitude, 1)
            lr_x_step_magnitude = tf.matmul(lr_x_step_magnitude, self.step_dist)

            lr_x_step_sign = tf.slice(activations, [0, 10], [-1, 2], 'x_step_sign')
            lr_x_step_sign = tf.nn.softmax(lr_x_step_sign, 1)
            lr_x_step_sign = tf.matmul(lr_x_step_sign, self.sign_dist)
            delta_x_step = lr_x_step_magnitude * lr_x_step_sign

            lr_grad_step_magnitude = tf.slice(activations, [0, 12], [-1, 5], 'grad_step_mag')
            lr_grad_step_magnitude = tf.nn.softmax(lr_grad_step_magnitude, 1)
            lr_grad_step_magnitude = tf.matmul(lr_grad_step_magnitude, self.lr_dist)

            lr_grad_step_sign = tf.slice(activations, [0, 17], [-1, -1], 'grad_step_sign')
            lr_grad_step_sign = tf.nn.softmax(lr_grad_step_sign, 1)
            lr_grad_step_sign = tf.matmul(lr_grad_step_sign, self.sign_dist)
            delta_lr = lr_grad_step_magnitude * lr_grad_step_sign

            rows = tf.shape(lr_grad_step_sign)[0]
            max_values = tf.expand_dims(tf.reduce_max(lr_grad_step_sign, 1), 1)
            flags = tf.equal(max_values, lr_grad_step_sign)
            max_sign = tf.where(flags, tf.ones([rows, 2]), tf.zeros([rows, 2]))

            return [delta_x_step, delta_lr]

    def step(self, args=None):
        with tf.name_scope('mlp_x_optimizer_step'):
            problem = args['problem']
            problem_variable_history = args['variable_history']
            problem_grad_history = args['grad_history']
            history_ptr = args['history_ptr']
            x_next = list()
            deltas_list = []
            for variable, variable_flat, batch_variable_history, batch_variable_grad_history in zip(problem.variables,
                                                                                             problem.variables_flat,
                                                                                             problem_variable_history,
                                                                                             problem_grad_history):
                deltas_x, deltas_lr = self.network({'inputs': [batch_variable_history, batch_variable_grad_history], 'history_ptr': history_ptr})
                deltas_list.append(deltas_x)
                max_values = tf.reduce_max(batch_variable_history, 1)
                min_values = tf.reduce_min(batch_variable_history, 1)
                max_values = tf.expand_dims(max_values, 1)
                min_values = tf.expand_dims(min_values, 1)
                diff = max_values - min_values
                ref = (max_values + min_values) / 2.0

                step_mean = deltas_lr + tf.multiply(deltas_x, diff)
                new_points = tf.add(ref, step_mean, 'new_points')
                new_points = problem.set_shape(new_points, like_variable=variable, op_name='reshaped_new_points')
                x_next.append(new_points)
            return {'x_next': x_next, 'deltas': deltas_list}


class MlpXHistoryGradSign(MlpNormHistory):

    def network(self, args=None):
        with tf.name_scope('mlp_x_optimizer_network'):
            args['inputs'][1] = tf.sign(args['inputs'][1])
            return super(MlpXHistoryGradSign, self).network(args)


class MlpXHistoryCont(MlpSimple):

    variable_history = None
    grad_sign_history = None
    history_ptr = None
    update_window = None
    guide_optimizer = None
    guide_step = None

    def __init__(self, problems, path, args):
        limit = args['limit']
        args['dims'] = (limit * 2, 1)
        super(MlpXHistoryCont, self).__init__(problems, path, args)
        with tf.name_scope('mlp_x_optimizer_input_init'):
            self.history_ptr = tf.Variable(0, 'history_ptr')
            self.guide_optimizer = tf.train.AdamOptimizer(.01, name='guide_optimizer')
            self.guide_step = self.guide_optimizer.minimize(self.problems.loss(self.problems.variables),
                                                            var_list=self.problems.variables, name='guide_step')
            self.variable_history = [tf.get_variable('variable_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, args['limit']], trainable=False)
                                     for i, shape in enumerate(self.problems.variables_flattened_shape)]
            self.grad_sign_history = [tf.get_variable('gradients_sign_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, args['limit']], trainable=False)
                                      for i, shape in enumerate(self.problems.variables_flattened_shape)]
            # for i, variable in enumerate(self.variable_history):
            #     tf.summary.histogram('variable_history_' + str(i), variable)

    def run_init(self, args=None):
        with tf.name_scope('mlp_x_init_with_session'):
            for col in range(self.global_args['limit']):
                for variable_ptr, (variable, gradient) in enumerate(zip(self.problems.variables_flat, self.problems.get_gradients())):
                    update_ops = self.update_history_ops(variable_ptr, (variable, tf.sign(gradient)))
                    self.session.run(update_ops)
                if col < self.global_args['limit'] - 1:
                    self.session.run(self.guide_step)
                    self.session.run(tf.assign_add(self.history_ptr, 1))
            self.session.run(tf.assign(self.history_ptr, 0))

    @staticmethod
    def normalize_values(history_tensor, switch=0):
        with tf.name_scope('mlp_x_normalize_variable_history'):
            if switch == 0:
                norm = tf.norm(history_tensor, ord=np.inf, axis=1, keep_dims=True)
                normalized_values =  tf.cond(tf.equal(norm, 0.0), history_tensor, tf.divide(history_tensor, norm))
                # normalized_values = tf.divide(history_tensor, tf.norm(history_tensor, ord=np.inf, axis=1, keep_dims=True))
            else:
                max_values = tf.reduce_max(history_tensor, 1)
                min_values = tf.reduce_min(history_tensor, 1)
                max_values = tf.reshape(max_values, [tf.shape(max_values)[0], 1])
                min_values = tf.reshape(min_values, [tf.shape(min_values)[0], 1])
                diff = max_values - min_values
                normalized_values = 2 * (history_tensor - min_values) / diff - 1.0
            return normalized_values

    def sort_input(self, inputs):
        with tf.name_scope('mlp_x_sort_input'):
            start = tf.slice(inputs, [0, self.history_ptr], [-1, -1], name='start')
            end = tf.slice(inputs, [0, 0], [-1, self.history_ptr], name='end')
            return tf.concat([start, end], 1, name='sorted_input')

    def network(self, args=None):
        with tf.name_scope('mlp_x_optimizer_network'):
            variable_history, variable_grad_sign_history = args['preprocessed_gradient']
            normalized_variable_history = self.normalize_values(variable_history)
            final_var_history = self.sort_input(normalized_variable_history)
            final_var_grad_history = self.sort_input(variable_grad_sign_history)
            final_input = tf.concat([final_var_history, final_var_grad_history], 1, name='final_input')
            activations = final_input
            activations = super(MlpXHistoryCont, self).network({'preprocessed_gradient': activations, 'reuse': args['reuse']})[0]
            output = tf.tanh(activations)
            # output = Preprocess.clamp(activations, {'min':-1, 'max':1})
            return [output]

    def step(self, args=None):
        with tf.name_scope('mlp_x_optimizer_step'):
            x_next = list()
            deltas_list = []
            for i, (variable, variable_history, variable_grad_sign_history) in enumerate(zip(self.problems.variables,
                                                                                             self.variable_history,
                                                                                             self.grad_sign_history)):
                deltas = self.network({'preprocessed_gradient': [variable_history, variable_grad_sign_history], 'reuse': i > 0})[0]
                deltas_list.append([deltas])
                max_values = tf.reduce_max(variable_history, 1)
                min_values = tf.reduce_min(variable_history, 1)
                max_values = tf.reshape(max_values, [tf.shape(max_values)[0], 1])
                min_values = tf.reshape(min_values, [tf.shape(min_values)[0], 1])
                diff = max_values - min_values
                ref_points = max_values + min_values
                new_points = tf.add(tf.divide(ref_points, 2.0), tf.multiply(deltas, diff), 'new_points')
                new_points = self.problems.set_shape(new_points, like_variable=variable, op_name='reshaped_new_points')
                x_next.append(new_points)
                # tf.summary.histogram('deltas_' + str(i), deltas)
                # tf.summary.histogram('new_x_' + str(i), new_points)
                # tf.summary.scalar('deltas', tf.squeeze(deltas))
                # tf.summary.scalar('new_x', tf.squeeze(new_points))
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
            update_list = super(MlpXHistoryCont, self).updates(args)
            flat_gradients = self.problems.get_gradients(args['x_next'])
            flat_variables = [self.problems.flatten_input(i, variable) for i, variable in enumerate(args['x_next'])]
            for i, (variable, grads) in enumerate(zip(flat_variables, flat_gradients)):
                new_input = [variable, tf.sign(grads)]
                update_list.extend(self.update_history_ops(i, new_input))
            with tf.control_dependencies(update_list):
                update_itr = tf.cond(self.history_ptr < self.global_args['limit'] - 1,
                                lambda: tf.assign_add(self.history_ptr, 1),
                                lambda: tf.assign(self.history_ptr, 0))
            return update_list + [update_itr]

    def reset_optimizer(self):
        reset = super(MlpXHistoryCont, self).reset_optimizer()
        reset.append(tf.variables_initializer(self.variable_history))
        reset.append(tf.variables_initializer(self.grad_sign_history))
        reset.append(tf.variables_initializer([self.history_ptr]))
        return reset

    def reset_problem(self):
        reset = super(MlpXHistoryCont, self).reset_problem()
        reset.append(tf.variables_initializer(self.variable_history))
        reset.append(tf.variables_initializer(self.grad_sign_history))
        reset.append(tf.variables_initializer([self.history_ptr]))
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

    def __init__(self, problems, path, args):
        limit = args['limit']
        args['dims'] = (limit * 2, 1) if self.is_availble('preprocess', args) else (limit, 1)
        super(MlpGradHistoryFAST, self).__init__(problems, path, args)
        with tf.name_scope('optimizer_network'):
            self.gradient_history_ptr = tf.Variable(0, 'gradient_history_ptr')
            self.guide_optimizer = tf.train.AdamOptimizer(.01)
            self.adam_problem_step = self.guide_optimizer.minimize(self.problems.loss(self.problems.variables), var_list=self.problems.variables)
            self.gradient_history = [tf.get_variable('gradients_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, args['limit']], trainable=False)
                                     for i, shape in enumerate(self.problems.variables_flattened_shape)]
            if self.is_availble('preprocess'):
                self.gradient_sign_history = [tf.get_variable('gradients_sign_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, args['limit']], trainable=False)
                                              for i, shape in enumerate(self.problems.variables_flattened_shape)]

    def run_init(self, args=None):
        for col in range(4):
            for variable_ptr, gradient in enumerate(self.problems.get_gradients()):
                indices = [[row, col] for row in range(gradient.get_shape()[0].value)]
                update_ops = [tf.scatter_nd_update(self.gradient_history[variable_ptr], indices, tf.squeeze(gradient))]
                if self.is_availble('preprocess'):
                    update_ops.append(tf.scatter_nd_update(self.gradient_sign_history[variable_ptr], indices,
                                                   tf.squeeze(tf.sign(gradient))))
                self.session.run(update_ops)
                self.session.run(self.adam_problem_step)

    def network(self, args):
        input_list = []
        gradients, sign = args['preprocessed_gradient']
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
                zip(self.problems.variables, self.gradient_history, self.gradient_sign_history)):
            deltas = self.network({'preprocessed_gradient': [variable_gradient_history, variable_gradient_sign_history]})[0]
            deltas_list.append(deltas)
            deltas = tf.multiply(deltas, self.learning_rate, name='apply_learning_rate')
            deltas = self.problems.set_shape(deltas, like_variable=variable, op_name='reshape_deltas')
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
