import tensorflow as tf
import numpy as np
from optimizers import XhistorySign, XSign

from problems import ElementwiseSquare, FitX, Mnist

prob = Mnist(args={'meta': False, 'minval':-10000, 'maxval':10000, 'dims': 100})

optim = XSign(prob, {'limit': 5, 'beta': 0.8})

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
            print('Xloss', np.log10(l))
        if g_s:
            s, u, l = iis.run([updates_adam, step_adam, loss_adam])
            print('Aloss', np.log10(l))