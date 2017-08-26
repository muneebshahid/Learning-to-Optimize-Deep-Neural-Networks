import tensorflow as tf
args = {'meta_learning_rate'}

def common():
    args = {}
    args['meta_learning_rate'] = .0001
    args['layer_width'] = 50
    args['hidden_layers'] = 1
    args['network_activation'] = tf.nn.relu
    args['unroll_len'] = 1
    return args


def mlp_norm_history():
    args = common()
    args['limit'] = 6
    args['grad_only'] = False
    args['grad_sign_only'] = False
    args['use_momentum'] = True
    args['momentum_limit'] = 5 if args['use_momentum'] else None
    args['momentum_base'] = 1.125
    args['history_range'] = None
    args['min_step'] = 1e-4
    args['min_step_max'] = False
    args['network_in_dims'] = args['limit'] if args['grad_only'] else args['limit'] * 2
    args['network_out_dims'] = 19 if args['min_step'] is None else 12
    args['normalize_with_sq_grad'] = False
    return args


def rnn_norm_history():
    args = mlp_norm_history()
    args['gru'] = False
    args['state_size'] = 5
    args['unroll_len'] = 20
    return args

def l2l2():
    args = {}
    args['meta_learning_rate'] = .00001
    args['state_size'] = 5
    args['unroll_len'] = 20
    args['num_time_scales'] = 5
    args['network_in_dims'] = args['num_time_scales'] * 2 + 1
    args['network_out_dims'] = 5
    return args
