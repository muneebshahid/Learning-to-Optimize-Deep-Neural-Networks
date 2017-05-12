from timeit import default_timer as timer

def run_epoch(sess, loss, ops, reset, num_unrolls):
      """Runs one optimization epoch."""
      start = timer()
      sess.run(reset)
      for _ in xrange(num_unrolls):
        cost = sess.run([loss] + ops)[0]
      return timer() - start, cost


def print_update(epoch, epochs, loss, epoch_interval, time):
    print 'Epoch/Total Epocs: ', epoch + 1, '/', epochs
    print 'Mean Log Loss: ', loss
    print 'Mean Epoch Time: ', time / epoch_interval
    print '-----\n'


def write_update(loss, time, mean_mats_values_list):
    with open('loss_file_upd', 'a') as log_file:
        log_file.write("{:.5f}".format(loss) + " " + "{:.2f}".format(time) + "\n")
    with open('mean_var_upd', 'a') as log_file:
        for mean_vars in mean_mats_values_list:
            for value in mean_vars:
                log_file.write(str(value) + ' ')
            log_file.write('\n')
