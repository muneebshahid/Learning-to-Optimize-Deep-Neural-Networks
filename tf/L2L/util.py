from __future__ import print_function
from timeit import default_timer as timer



def run_epoch(sess, loss, ops, reset, num_unrolls):
    """Runs one optimization epoch."""
    cost = 0
    final_ops = list()
    final_ops.append(loss)
    if ops is not None:
        final_ops = final_ops + ops

    start = timer()
    if reset is not None:
        sess.run(reset)
    for _ in xrange(num_unrolls):
        cost += sess.run(final_ops)[0]
    return timer() - start, cost / num_unrolls


def print_update(epoch, epochs, loss, epoch_interval, time):
    print('Epoch/Total Epocs: ', epoch + 1, '/', epochs)
    print('Mean Log Loss: ', loss)
    print('Mean Epoch Time: ', time / epoch_interval)
    print('--------------------------------------------------------------------\n')


def write_update(loss, time, mean_mats_values_list):
    with open('loss_file_upd', 'a') as log_file:
        log_file.write("{:.5f}".format(loss) + " " + "{:.2f}".format(time) + "\n")
    with open('mean_var_upd', 'a') as log_file:
        for mean_vars in mean_mats_values_list:
            for value in mean_vars:
                log_file.write(str(value) + ' ')
            log_file.write('\n')


def get_model_path(flag_optimizer, model_id, preprocess_args=None, learning_rate=None, num_layer=None, layer_width=None, momentum=None, second_derivative=None):
    path = 'results/' + flag_optimizer + '_model_' + model_id
    if preprocess_args is not None:
        path += ('_' + preprocess_args[0].func_name)
        for key in preprocess_args[1]:
            path += ('_' + str(key) + '_' + str(preprocess_args[1][key]))
    path += (('_lr_' + str(learning_rate)) if learning_rate is not None else '')
    path += (('_nl_' + str(num_layer)) if num_layer is not None else '')
    path += (('_lw_' + str(layer_width)) if layer_width is not None else '')
    path += (('_mom_' + str(momentum)) if momentum is not None else '')
    path += (('_secdev_' + str(second_derivative)) if second_derivative is not None else '')
    return path