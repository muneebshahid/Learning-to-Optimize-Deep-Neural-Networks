import tensorflow as tf
from optimizers import XhistorySign

from problems import ElementwiseSquare, FitX, Mnist

prob = ElementwiseSquare(args={'meta': True, 'minval':-1000, 'maxval':1000, 'dims':100})

optim = XhistorySign(prob, {'limit': 5})

optim.build()

iis = tf.InteractiveSession()
iis.run(tf.global_variables_initializer())

optim.set_session(iis)
optim.init_with_session()

def itr(itera):
    for i in range(itera):
        step, up, loss = iis.run([optim.ops_step, optim.ops_updates, optim.ops_loss])
        print('step', step)
        print('up', up)
        print('loss', loss)