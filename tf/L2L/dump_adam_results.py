import problems
import tensorflow as tf
import numpy as np

learning_rate_range = [.1, .05, .01, .005, .001, .0005, .0001, .00005, .00001]
adam_params = [[0.99, 0.9999],
               [0.96, 0.9996],
               [0.93, 0.9993],
               [0.9, 0.999],
               [0.86, 0.8886],
               [0.83, 0.8883],
               [0.8, 0.888],
               [0.75, 0.7775],
               [0.7, 0.777],
               [0.65, 0.6665],
               [0.6, 0.666]]

def write_to_file(f_name, list_var):
    with open(f_name, 'a') as log_file:
        for variable in list_var:
            log_file.write(str(variable) + ' ')
        log_file.write('\n')

problem_path = '/mhome/shahidm/thesis/thesis_code/tf/L2L/mnist_save_vars_mlp/mnist_variables'


for adam_param in adam_params:
    for learning_rate in learning_rate_range:
        id = str(adam_param) + ' ' + str(learning_rate)
        print('================================')
        tf.reset_default_graph()
        iis = tf.InteractiveSession()
        mnist = problems.Mnist({'minval': -100.0, 'maxval': 100.0})
        loss = tf.squeeze(mnist.loss(mnist.variables))
        mnist.restore(iis, problem_path)
        beta1 = adam_param[0]
        beta2 = adam_param[1]
        adam = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2)
        minimize = adam.minimize(loss, var_list=mnist.variables)
        iis.run(tf.global_variables_initializer())
        total_loss = 0
        for i in range(20000):
            _, loss_run = iis.run([minimize, loss])
            total_loss += loss_run
            if (i + 1) % 400 == 0:
                print(id)
                print(str(i + 1) + '/' + str(20000))
                avg_loss = np.log10(total_loss / 400.0)
                write_to_file('tf_summary/adam_sig_' + str(learning_rate) + '_' + str(beta1) + '_' + str(beta2), [avg_loss])
                print(avg_loss)
                total_loss = 0
