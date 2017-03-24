import theano
import theano.tensor as T
from theano.ifelse import ifelse
from collections import OrderedDict
import numpy as np

def update_1(test_var, updates):
	updates[test_var] = np.int64(1)
	return np.int64(1), np.int64(0), np.int64(3)

def update_2(test_var, updates):
	updates[test_var] = np.int64(2)
	return np.int64(2), np.int64(54), np.int64(666)

def get_updates():
	t = theano.shared(0., name='t')
	test_var = theano.shared(0, name='test_var')	
	updates = OrderedDict()
	updates[t] = t + 1	
	ifelse(T.lt(t, 5), update_2(test_var, updates), update_1(test_var, updates))
	return updates, t, test_var

updates, t, test_var = get_updates()
print t.get_value(), test_var.get_value()
func = theano.function([], [], updates=updates)
func()
