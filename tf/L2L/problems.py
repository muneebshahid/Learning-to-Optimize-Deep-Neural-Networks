import tensorflow as tf

def var_square():
    var = tf.get_variable('x', shape=[2 ,1], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-5, maxval=20), trainable=False)
    def loss(var):
        return tf.square(var, name='var_squared')
    return  var, loss
