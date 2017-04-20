import tensorflow as tf

def var_square():
    var = tf.get_variable('x', shape=[1 ,1], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)
    def loss(var):
        return tf.square(var, name='var_squared')
    return  var, loss
