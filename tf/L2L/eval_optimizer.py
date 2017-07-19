from __future__ import print_function
import tensorflow as tf
import numpy as np
import problems
import meta_optimizers
import util
from preprocess import Preprocess

l2l = tf.Graph()
with l2l.as_default():
    preprocess = [Preprocess.log_sign, {'k': 10}]


    model_id = '1000_FINAL'

    #########################
    epochs = 100
    epoch_interval = 1
    eval_interval = 200
    validation_epochs = 50
    test_epochs = 500
    #########################
    learning_rate = 0.0001
    layer_width = 50
    momentum = False
    #########################

    num_unrolls_per_epoch = 1
    io_path = util.get_model_path(flag_optimizer='Mlp', model_id=model_id)
    problem_batches = problems.create_batches_all(train=False)
    optim = meta_optimizers.MlpHistoryGradNorm(problem_batches, path=None,
                                               args={'hidden_layers': 1, 'learning_rate': learning_rate,
                                                     'meta_learning_rate': 0.0001, 'layer_width': layer_width,
                                                     'preprocess': preprocess, 'limit': 5})
    optim.build()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess)
        optim.set_session(sess)
        optim.run_init()
        l = []
        for i, (problem, problem_variables_history) in enumerate(zip(optim.problems, optim.variable_history)):
            name_prefix = problem.__class__.__name__ + "_" + str(i)
            with tf.name_scope(name_prefix):
                loss = tf.squeeze(tf.log(problem.loss(problem.variables)))
                l.append(loss)
                tf.summary.scalar('loss', loss)
                for j, (variable, variable_history) in enumerate(zip(problem.variables, problem_variables_history)):
                    # tf.summary.histogram('variable_' + str(j), variable)
                    tf.summary.histogram('variable_history_scl_' + str(j), variable_history)
        all_summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter('tf_summary/')
        writer.add_graph(sess.graph)
        l2l.finalize()
        print('---- Starting Evaluation ----')
        optim.load(io_path)
        print('Optimizer loaded.')
        total_itr = 1000
        for i in range(total_itr):
            print(str(i) + '/' + str(total_itr))
            _, curr_loss, summaries = sess.run([optim.ops_updates, l, all_summ])
            print(curr_loss)
            writer.add_summary(summaries, i)


