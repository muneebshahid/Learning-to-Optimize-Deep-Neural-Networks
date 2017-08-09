import tensorflow as tf
args = {'meta_learning_rate'}

def common():
    args = {}
    args['meta_learning_rate'] = .0001
    args['layer_width'] = 50
    args['hidden_layers'] = 1
    args['network_activation'] = tf.nn.relu
    return args


def norm_history():
    args = common()
    args['limit'] = 20
    args['grad_only'] = True
    args['grad_sign_only'] = False
    args['moving_avg'] = False
    args['history_range'] = None
    args['min_step'] = 1e-5
    args['network_in_dims'] = args['limit'] if args['grad_only'] else args['limit'] * 2
    args['network_in_dims'] += (1 if args['moving_avg'] else 0)
    args['network_out_dims'] = 19 if args['min_step'] is None else 12
    return args


def gru_norm_history():
    args = norm_history()
    args['state_size'] = 5
    return args
