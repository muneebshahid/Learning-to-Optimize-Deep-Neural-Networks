import meta_optimizer
import problems
from preprocess import Preprocess
import tensorflow as tf

second_derivatives = False
meta = True


preprocess = [Preprocess.log_sign, {'k': 10}]
problem = problems.Mnist(args={'gog': second_derivatives, 'meta': meta, 'mode': 'test'})
load_path = 'trained_models/model_10000'

epochs = 1
num_optim_steps_per_epoch = 1
unroll_len = 1



choice_optimizer = 'l2l'

if choice_optimizer == 'mlp':
    optimizer = meta_optimizer.mlp(problem, path=load_path, args={})

else:
    optimizer = meta_optimizer.l2l(problem, path=load_path, args={})

inputs = optimizer.stacked_optimizer_inputs

deltas_list = []
iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())
optimizer.load(iis, load_path)
for input in optimizer.stacked_optimizer_inputs:
    output = optimizer.step(input)[0]


