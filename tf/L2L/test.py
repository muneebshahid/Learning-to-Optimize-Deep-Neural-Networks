import tensorflow as tf
'''
l2l = tf.Graph()
with l2l.as_default():
	def problem():
		x = tf.get_variable('x', shape=[1,1], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)
		def func():
			return tf.square(x, name='x_squared_sta')
		return x, func		
	state_size = 20
	num_layers = 1
	
	x, func = problem()
	grads = tf.gradients(func(), x)[0]
	grads = tf.reshape(grads, [1, 1])
	
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size)	
	hidden_state = lstm_cell.zero_state(1, tf.float32)
	with tf.variable_scope('rnn'):
		lstm_cell(grads, hidden_state)
		W = tf.get_variable('softmax_w', [state_size, 1])
		b = tf.get_variable('softmax_b', [1])

	loss = 0
	def update(t):
		global hidden_state, loss
		with tf.variable_scope('rnn', reuse=True):				
			output, hidden_state = lstm_cell(grads, hidden_state)
			deltas = tf.matmul(output, W) + b
		step = x.assign_add(deltas)		
		with l2l.control_dependencies([step]):			
			loss += func()
		t_next = t + 1
		return t_next
	
	_ =  tf.while_loop(
			cond=lambda t : t < 20,
			body=update,
			loop_vars=([0]),
			parallel_iterations=1,
			swap_memory=True,
			name="unroll")

	optimizer = tf.train.AdamOptimizer(.01)
	step = optimizer.minimize(func())
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())		
		print sess.run([x, func()])
		print sess.run([loss])		
		print sess.run([loss])		
		print sess.run([x, func()])
		print sess.run(tf.gradients(func(), x))
'''


l2l = tf.Graph()
with l2l.as_default():
	def problem():
		x = tf.get_variable('x', shape=[1,1], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)
		def func(var):
			return tf.square(var, name='x_squared_sta')
		return x, func		
	state_size = 20
	num_layers = 1

	x, func = problem()

	def get_gradients(func, x):
		return tf.gradients(func(x), x)[0]

	lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size)	
	hidden_state = lstm_cell.zero_state(1, tf.float32)

	with tf.variable_scope('rnn'):
		lstm_cell(get_gradients(func, x), hidden_state)
		W = tf.get_variable('softmax_w', [state_size, 1])
		b = tf.get_variable('softmax_b', [1])

	def update(t, loss, x):
		global hidden_state
		with tf.variable_scope('rnn', reuse=True):				
			output, hidden_state = lstm_cell(get_gradients(func, x), hidden_state)
			deltas = tf.matmul(output, W) + b
		x_next = x + deltas				
		loss += func(x)
		t_next = t + 1
		return t_next, loss, x_next

	_, loss_final, x_final =  tf.while_loop(
		cond=lambda t, *_ : t < 20,
		body=update,
		loop_vars=([0, tf.zeros([1, 1]), x]),
		parallel_iterations=1,
		swap_memory=True,
		name="unroll")
	
	update = x.assign(x_final)
	optimizer = tf.train.AdamOptimizer(.01)
	step_optim = optimizer.minimize(loss_final)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(100):
			print sess.run([update, func(x), get_gradients(func, x), loss_final, step_optim])		