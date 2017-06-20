import problems
import weight_prediction as wp
import tensorflow as tf
import six.moves
import os
mnist = problems.Mnist({})
pred = wp.mlp({'problem': mnist})
optim_step_problem_ops, optim_step_pred_ops, loss_pred, loss_problem, update_history_ops = pred.build()

iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())
tf.train.start_queue_runners(iis)
pred.init_history({'sess': iis, 'optim_prob_op': optim_step_problem_ops})