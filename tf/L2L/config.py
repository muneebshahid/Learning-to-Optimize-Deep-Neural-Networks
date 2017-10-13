import tensorflow as tf
args = {'meta_learning_rate'}

def common():
    args = {}
    args['meta_learning_rate'] = .0001
    args['Adam'] = False
    args['decay_meta_learning_rate'] = False
    args['starter_learning_rate'] = 0.0001
    args['end_learning_rate'] = 0.000005
    args['decay_steps'] = 'NEED TO SET'
    args['power'] = 4
    args['layer_width'] = 50
    args['hidden_layers'] = 0
    args['network_activation'] = tf.nn.relu
    args['unroll_len'] = 1
    args['unroll_len_val'] = 1
    args['use_guide_step'] = False
    return args


def mlp_norm_history():
    args = common()
    args['limit'] = 6
    args['step_dist_max_step'] = 1.0
    args['grad_only'] = True
    args['grad_sign_only'] = False
    args['use_momentum'] = True
    args['momentum_base'] = 1.1
    args['history_range'] = None
    args['min_lr'] = 1e-3
    args['ref_point'] = 0 #0 = last update, 1 = (max + min) / 2, 2 = learn a weighted average
    args['use_diff'] = False
    args['decay_min_lr'] = False
    args['decay_min_lr_max'] = 1e-3
    args['decay_min_lr_min'] = 1e-4
    args['decay_min_lr_steps'] = 20000
    args['learn_lr'] = False
    args['use_lr_mv_avg'] = False
    args['learn_lr_delta'] = False
    args['min_step_max'] = False
    args['learn_momentum_base'] = False
    args['enable_noise_est'] = False
    args['normalize_with_sq_grad'] = False
    args['use_log_noise'] = False
    args['use_delta_mv_avg'] = False
    args['use_dist_mv_avg'] = False
    args['use_tanh_output'] = False
    args['network_in_dims'] = args['limit'] if args['grad_only'] else args['limit'] * 2
    args['network_in_dims'] *= (2 if args['enable_noise_est'] else 1)
    args['network_out_dims'] = 1 if args['use_tanh_output'] else 12
    args['network_out_dims'] = 19 if args['min_lr'] is None else args['network_out_dims']
    args['network_out_dims'] += (1 if args['learn_momentum_base'] else 0)
    args['network_out_dims'] += (7 if args['learn_lr'] else 0)
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

def adam():
    args = common()
    args['beta_1'] = 0.9
    args['beta_2'] = 0.999
    args['lr'] = 0.01
    args['eps'] = 1e-8
    return args

def aug_optim():
    args = common()
    args['lr'] = .0005
    args['lr_input_optims'] = 1.0
    args['num_input_optims'] = 6
    args['use_network'] = True
    args['use_positive_weights'] = True
    args['normalize_weights'] = True
    args['network_out_dims'] = args['num_input_optims'] if args['use_network'] else 1
    args['use_input_optim_loss'] = False
    args['use_input_optim_loss_rel'] = False
    args['use_adam_loss'] = False

    args['decay_learning_rate'] = False
    args['min_lr'] = 0.0
    args['max_lr'] = args['lr']
    args['t_max'] = 'NEED TO SET'

    args['lr_dist'] = [1e-1, 1e-2, 1e-3, 1e-4, 0.0]
    args['beta_max'] = 0.9999
    args['learn_betas'] = False
    args['learn_lr'] = False
    args['use_rel_loss'] = False
    return args

def aug_optim_rnn():
    args = aug_optim()
    args['unroll_len'] = 20
    args['network_out_dims'] = args['num_input_optims'] + (2 if args['learn_betas'] else 0)
    args['network_out_dims'] += (len(args['lr_dist']) if args['learn_lr'] else 0)

    return args

def aug_optim_gru():
    args = aug_optim_rnn()
    args['state_size'] = 6
    return args

