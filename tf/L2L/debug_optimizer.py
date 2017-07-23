import tensorflow as tf
import numpy as np
from optimizers import *
from problems import ElementwiseSquare, FitX, Mnist, Rosenbrock, RosenbrockMulti, DifferentPowers
tf.set_random_seed(0)
prob = Rosenbrock(args={'minval':-10, 'maxval':10, 'dims': 2})

optim_pre = tf.train.AdamOptimizer(.01)
optim_pre_loss = prob.loss(prob.variables)
optim_pre_step = optim_pre.minimize(optim_pre_loss, var_list=prob.variables)


optim_self = XHistorySign(prob, args={'limit': 5, 'beta': 0.8})
optim_self.build()
iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())
optim_self.set_session(iis)
optim_self.run_init()
p = optim_self.problem.variables
hist = optim_self.variable_history
gs = optim_self.grad_history
x_n = optim_self.ops_step['x_next']
d_n = optim_self.ops_step['deltas']

def itr(itera, x_s=False, g_s=False, print_itr=1):

    loss = 0

    def print_loss(name, i, loss):
        if (i + 1) % print_itr == 0:
            print(name + ' :', np.log10(loss / print_itr))
            return 0
        return loss

    for i in range(itera):
        if x_s:
            s, u, l = iis.run([optim_self.ops_updates, optim_self.ops_step, optim_self.ops_loss])
            loss += l
            loss = print_loss('Xloss', i, loss)
        if g_s:
            s, l = iis.run([optim_pre_step, optim_pre_loss])
            loss += l
            loss = print_loss('Aloss', i, loss)

itr(1000, x_s=True, print_itr=100)
