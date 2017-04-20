import tensorflow as tf



# l2l = tf.Graph()
# with l2l.as_default():
# 	def problem():
# 		x = tf.get_variable('x', shape=[1,1], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)
# 		def func(var):
# 			return tf.square(var, name='x_squared_sta')
# 		return x, func
# 	state_size = 20
# 	num_layers = 1
#
# 	x, func = problem()
#
# 	def get_gradients(func, x):
# 		return tf.gradients(func(x), x)[0]
#
# 	lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
# 	hidden_state = lstm_cell.zero_state(1, tf.float32)
#
# 	with tf.variable_scope('rnn'):
# 		lstm_cell(get_gradients(func, x), hidden_state)
# 		W = tf.get_variable('softmax_w', [state_size, 1])
# 		b = tf.get_variable('softmax_b', [1])
#
# 	def update(t, loss, x, hidden_state):
# 		with tf.variable_scope('rnn', reuse=True):
# 			output, hidden_state_next = lstm_cell(get_gradients(func, x), hidden_state)
# 			deltas = tf.matmul(output, W) + b
# 		x_next = x + deltas
# 		loss += func(x)
# 		t_next = t + 1
# 		return t_next, loss, x_next, hidden_state_next
#
# 	_, loss_final, x_final, hidden_state =  tf.while_loop(
# 		cond=lambda t, *_ : t < 2,
# 		body=update,
# 		loop_vars=([0, tf.zeros([1, 1]), x, hidden_state]),
# 		parallel_iterations=1,
# 		swap_memory=True,
# 		name="unroll")
#
# 	update_x = x.assign(x_final)
#
# 	optimizer = tf.train.AdamOptimizer(.01)
# 	step_optim = optimizer.minimize(loss_final)
#
# 	with tf.Session() as sess:
# 		sess.run(tf.global_variables_initializer())
# 		for i in range(100):
# 			print sess.run([update_x, func(x), get_gradients(func, x), loss_final, step_optim]


class test():
	x = None
	t = None
	func = None
	lstm_cell = None
	hidden_state = None

	def __init__(self):
		self.x = tf.get_variable('x', shape=[2,1], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)
		self.t = 0
		self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(1)
		self.hidden_state = self.lstm_cell.zero_state(2, tf.float32)

	def func(self, x):
		return tf.square(x, name='x_squared_sta')
	def update(self, t, fx_array, x,hidden_state):
		output, hidden_state = self.lstm_cell(tf.gradients(self.func(x), x)[0], hidden_state)
		x_next = x + output
		fx_array = fx_array.write(t, x_next)
		t = t + 1
		return t, fx_array, x_next, hidden_state


	def loop(self):
		fx_array = tf.TensorArray(tf.float32, size=10,
                              clear_after_read=False)
		t_f, fx_array, x_final, self.hidden_state =  tf.while_loop(
			cond=lambda t, *_ : t < 10,
			body=self.update,
			loop_vars=([0, fx_array, self.x, self.hidden_state]),
			parallel_iterations=1,
			swap_memory=True,
			name="unroll")

		self.t = t_f

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			# print sess.run(self.x)
			print sess.run(self.t)
			print sess.run([fx_array.stack(), x_final, self.x, tf.gradients(self.func(self.x), self.x)])
			# print sess.run(self.hidden_state)
			# print sess.run(self.x)

T = test()
T.loop()
