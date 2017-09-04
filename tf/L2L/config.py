import tensorflow as tf
args = {'meta_learning_rate'}

def common():
    args = {}
    args['meta_learning_rate'] = .00005
    args['layer_width'] = 50
    args['hidden_layers'] = 1
    args['network_activation'] = tf.nn.relu
    args['unroll_len'] = 1
    args['use_guide_step'] = False
    return args


def mlp_norm_history():
    args = common()
    args['limit'] = 6
    args['step_dist_max_step'] = 1.0
    args['grad_only'] = False
    args['grad_sign_only'] = False
    args['use_momentum'] = True
    args['momentum_base'] = 1.1
    args['history_range'] = None
    args['min_lr'] = 1e-3
    args['decay_min_lr'] = True
    args['decay_min_lr_max'] = 1e-3
    args['decay_min_lr_min'] = 1e-4
    args['decay_min_lr_steps'] = 20000
    args['learn_lr'] = False
    args['learn_lr_delta'] = False
    args['min_step_max'] = False
    args['learn_momentum_base'] = False
    args['enable_noise_est'] = True
    args['use_log_noise'] = False
    args['use_delta_mv_avg'] = False
    args['normalize_with_sq_grad'] = True
    args['use_dist_mv_avg'] = False
    args['network_in_dims'] = args['limit'] if args['grad_only'] else args['limit'] * 2
    args['network_in_dims'] *= (2 if args['enable_noise_est'] else 1)
    args['network_out_dims'] = 19 if args['min_lr'] is None else 12
    args['network_out_dims'] += (1 if args['learn_momentum_base'] else 0)
    args['network_out_dims'] += (9 if args['learn_lr'] else 0)
    args['network_out_dims'] += (1 if args['learn_lr_delta'] else 0)
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
