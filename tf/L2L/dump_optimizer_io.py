import meta_optimizer
import problems
from preprocess import Preprocess
import tensorflow as tf
import numpy as np
import util
from matplotlib.pyplot import scatter, ion, show

second_derivatives = False
meta = True

flag_optimizer = 'MLP'
preprocess = [Preprocess.log_sign, {'k': 10}]
problem = problems.Mnist(args={'gog': second_derivatives, 'meta': meta, 'mode': 'test'})

model_id = '1000000'
model_id += '_FINAL'
load_path = None#util.get_model_path(flag_optimizer=flag_optimizer, model_id=model_id)

epochs = 1
num_optim_steps_per_epoch = 1
unroll_len = 1
mean_optim_variables = None
if flag_optimizer == 'MLP':
    optimizer = meta_optimizer.MlpSimple(problem, path=load_path, args={'preprocess': preprocess})
    mean_optim_variables = [tf.reduce_mean(optimizer.w_1), tf.reduce_mean(optimizer.w_out),
                            tf.reduce_mean(optimizer.b_1), optimizer.b_out[0][0]]
else:
    optimizer = meta_optimizer.l2l(problem, path=load_path, args={})

optimizer_inputs = optimizer.optimizer_input_stack
mean_problem_variables = [tf.reduce_mean(variable) for variable in problem.variables]

flat_gradients, preprocessed_gradients, deltas_list = [], [], []
iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())
# optimizer.load(iis, load_path)

print('Init Problem Vars: ', iis.run(mean_problem_variables))
if mean_optim_variables is not None:
    print('Init Optim Vars: ', iis.run(mean_optim_variables))

flatten = lambda mat_array: [element for mat in mat_array for element in mat]

for gradient in optimizer_inputs:
    output = optimizer.core(gradient)[0]
    deltas_list.append(output)
    flat_gradients.append(gradient['flat_gradient'])
    preprocessed_gradients.append(gradient['preprocessed_gradient'])
flat_gradients, preprocessed_gradients, deltas = iis.run([flat_gradients, preprocessed_gradients, deltas_list])
final_array = np.hstack((np.hstack((flat_gradients, preprocessed_gradients)), deltas))
np.savetxt(load_path + '_optim_io.txt', final_array, fmt='%7f')
print('Results Dumped')