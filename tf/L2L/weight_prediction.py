from abc import ABCMeta
import tensorflow as tf

class WeightPredictor():
    __metaclass__ = ABCMeta

    problem = None
    optimizer_problem = None
    optimizer_weight_predictor = None
    network = None

    def __init__(self, args):
        self.problem = args['problem']
        self.optimizer_problem = tf.train.AdamOptimizer(0.01)
        self.optimizer_weight_predictor = tf.train.AdadeltaOptimizer(1e-3)

    def init_history(self, args=None):
        pass

    def core(self, args=None):
        pass

    def predict(self, args=None):
        pass

    def optim_step_problem(self, args=None):
        pass

    def loss_problem(self, args=None):
        pass

    def optim_step_pred(self, args=None):
        pass

    def loss_pred(self, args=None):
        pass

    def update_history(self, args=None):
        pass

    def build(self):
        pass

class mlp(WeightPredictor):

    layer_width = None
    variable_history = None
    variable_update_ptr = None
    prediction_pointer = None

    def __init__(self, args):
        super(mlp, self).__init__(args)
        self.layer_width = 40
        self.network = {}
        with tf.variable_scope('optimizer_core'):
            layer_1_w = tf.get_variable('w_in', shape=[4, self.layer_width], initializer=tf.random_normal_initializer(mean=0.0, stddev=.1))
            layer_1_b = tf.get_variable('b_in', shape=[1, self.layer_width], initializer=tf.zeros_initializer)
            self.network['l_1'] = {'w': layer_1_w, 'b': layer_1_b}
            layer_o_w = tf.get_variable('w_out', shape=[self.layer_width, 1], initializer=tf.random_normal_initializer(mean=0.0, stddev=.1))
            layer_o_b = tf.get_variable('b_out', shape=[1, 1], initializer=tf.zeros_initializer)
            self.network['l_out'] = {'w': layer_o_w, 'b': layer_o_b}
        self.variable_history = [tf.get_variable('var_history' + str(i), initializer=tf.zeros_initializer, shape=[shape, 4], trainable=False)
                                 for i, shape in enumerate(self.problem.variables_flattened_shape)]
    
    # Couldnt figure out how to run an op more than once in the same sess call, hence this ugliness.
    def init_history(self, args=None):
        sess = args['sess']
        optim_step_problem_op = args['optim_prob_op']
        col = 0
        for iter in range(101)[1:]:
            if iter == 1 or iter == 40 or iter == 70  or iter == 100:
                for variable_ptr, variable in enumerate(self.problem.variables_flat):
                    indices = [[row, col] for row in range(variable.get_shape()[0].value)]
                    sess.run(tf.scatter_nd_update(self.variable_history[variable_ptr], indices, tf.squeeze(variable)))
                    sess.run(optim_step_problem_op)
                col += 1

    def predict(self, args=None):
        predictions = []
        for variable_history in self.variable_history:
            prediction = self.core({'input': variable_history})
            predictions.append(prediction)
        return predictions


    def loss_problem(self, args=None):
        return self.problem.loss(self.problem.variables)

    def optim_step_problem(self, args=None):
        return self.optimizer_problem.minimize(self.loss_problem(), var_list=self.problem.variables)

    def loss_pred(self, args=None):
        loss = 0
        predictions = self.predict()
        for prediction, variable in zip(predictions, self.problem.variables_flat):
            loss += tf.reduce_sum(tf.abs(prediction - variable))
        return loss

    def optim_step_pred(self, args=None):
        trainable_vars = [param for layer in self.network.values() for param in layer.values()]
        optim_step_pred_ops = self.optimizer_weight_predictor.minimize(self.loss_pred(), var_list=trainable_vars)
        return optim_step_pred_ops

    def core(self, args=None):
        inputs = args['input']
        activations = tf.nn.relu(tf.add(tf.matmul(inputs, self.network['l_1']['w']), self.network['l_1']['b']))
        output = tf.add(tf.matmul(activations, self.network['l_out']['w']), self.network['l_out']['b'], name='layer_final_activation')
        return output

    def update_history(self, args=None):
        return

    def build(self):
        optim_step_problem_ops = self.optim_step_problem()
        optim_step_pred_ops = self.optim_step_pred({'dep': optim_step_problem_ops})
        update_history_ops = self.update_history()
        return optim_step_problem_ops, optim_step_pred_ops, self.loss_pred(), self.loss_problem(), update_history_ops





