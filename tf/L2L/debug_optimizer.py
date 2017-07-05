import tensorflow as tf
import numpy as np
from optimizers import XHistorySign, XSign
from problems import ElementwiseSquare, FitX, Mnist, Rosenbrock, RosenbrockMulti, DifferentPowers

prob = DifferentPowers(args={'meta': False, 'minval':-10000, 'maxval':10000, 'dims': 2})

optim = XHistorySign(prob, {'limit': 5, 'beta': 0.9})

optim.build()

iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())

optim.set_session(iis)
optim.init_with_session()

def itr(itera, x_s=True, g_s=False):
    updates = optim.ops_updates
    step = optim.ops_step
    loss = optim.ops_loss
    # if g_s:
    updates_adam = []
    step_adam = optim.guide_step
    loss_adam = optim.loss()
    for i in range(itera):
        if x_s:
            s, u, l = iis.run([updates, step, loss])
            print('Xloss', l)
        if g_s:
            s, u, l = iis.run([updates_adam, step_adam, loss_adam])
            print('Aloss', l)