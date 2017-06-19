import problems
import weight_prediction as wp
import tensorflow as tf

mnist = problems.Mnist({})
pred = wp.mlp({'problem': mnist})
init_history_ops, step_optim_problem_ops, optim_step_pred_ops, update_history_ops = pred.build()

iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())
