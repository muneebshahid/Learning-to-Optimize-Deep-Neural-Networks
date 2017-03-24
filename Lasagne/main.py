#code adapted from minst.py in Lasagne
import numpy as np
import theano
from theano import tensor as T
import theano.printing as pr

import lasagne
import time
import pickle

from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_mnist():
    import gzip

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        # Read the labels in Yann LeCun's binary format
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    root_datatsets = '../../datasets/mnist/'

    X_train = load_mnist_images(root_datatsets + 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(root_datatsets + 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(root_datatsets + 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(root_datatsets + 't10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_cifar10():
    xs = []
    ys = []
    root_folder = '../../datasets/cifar10/'
    for j in range(5):
      d = unpickle(root_folder + 'cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle(root_folder + 'cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)

    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip), axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return lasagne.utils.floatX(X_train), Y_train.astype('int32'), \
           None, None, \
           lasagne.utils.floatX(X_test), Y_test.astype('int32')

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_cnn_mnist(input_var):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def build_cnn_deep_cifar(input_var, n=5):
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(
            l, num_units=10,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return network


def main(args):
    X_train, Y_train, X_val, y_val, X_test, Y_test = args['dataset']
    input_var = args['input_var']
    target_var = args['target_var']
    network = args['network']
    num_epochs = args['num_epochs']

    # training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)

    if args['weight_decay']:
        all_layers = lasagne.layers.get_all_layers(network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
        loss = loss + l2_penalty

    loss_prev = T.fscalar('loss_prev')
    if args['optim'] == 'eve_adam':
        print args['optim']
        updates, branch, d, f_hat, f_prev, div_res, lb, ub = lasagne.updates.eve_adam(loss, params, loss_prev)
    elif args['optim'] == 'eve_adamax':
        print args['optim']
        updates, branch, d, f_hat, f_prev, div_res, lb, ub = lasagne.updates.eve_adamax(loss, params, loss_prev)
    elif args['optim'] == 'adamax':
        print 'using adamax'
        updates = lasagne.updates. adamax(loss, params)
    else:
        print 'using adam'
        updates = lasagne.updates.adam(loss, params)

    train_fn = theano.function([input_var, target_var, loss_prev], loss, updates=updates, on_unused_input='ignore')


    # testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    train_log_file_name  = args['optim'] + '.train'
    test_log_file_name  = args['optim'] + '.test'

    print('Starting Training...')
    for epoch in range(num_epochs):

        if args['dataset'] == 'cifar10':
            # shuffle whole dataset
            train_indices = np.arange(100000)
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices,:,:,:]
            Y_train = Y_train[train_indices]
            print 'shufflled all data'

        train_err = 0
        train_batches = 0
        start_time = time.time()
        batch_loss = 0
        d_batch = 0.0
        for batch in iterate_minibatches(X_train, Y_train, args['batch_size'], shuffle=True):
            inputs, targets = batch
            # print 'branch: ', branch.get_value(), ' d: ', d.get_value(), ' f_hat:', f_hat.get_value(), ' f_prev: ', f_prev.get_value(), ' div_res: ', div_res.get_value(), ' lb: ', lb.get_value(), ' ub: ', ub.get_value()
            if 'eve' in args['optim']:
                d_batch += d.get_value()
            batch_loss = train_fn(inputs, targets, batch_loss)
            train_err += batch_loss
            train_batches += 1

        if X_val is not None and y_val is not None:
            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
                epoch_val_loss = val_err / val_batches
                epoch_val_acc = val_acc / val_batches * 100
        else:
            epoch_val_loss = 0
            epoch_val_acc = 0

        # Then we print the results for this epoch:
        epoch_time_taken = time.time() - start_time
        epoch_train_loss = train_err / train_batches
        d_batch = d_batch / train_batches
        
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, epoch_time_taken))
        print("  training loss:\t\t{:.6f}".format(epoch_train_loss))
        print("  d_t:\t\t{:.6f}".format(d_batch))
        print("  validation loss:\t\t{:.6f}".format(epoch_val_loss))
        print("  validation accuracy:\t\t{:.2f} %".format(epoch_val_acc))

        with open(train_log_file_name, 'a')  as train_file:
            train_file.write("{:.3f}".format(epoch_time_taken) + "\t{:.6f}".format(epoch_train_loss) + "\t{:.6f}".format(epoch_val_loss) +\
            "\t\t{:.6f}".format(d_batch) + "\t{:.2f}".format(epoch_val_acc) + "\n")

        # Print and write test values
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        final_test_loss = test_err / test_batches
        final_test_acc = test_acc / test_batches * 100
        print("Test Results:")
        print("  test loss:\t\t\t{:.6f}".format(final_test_loss))
        print("  test accuracy:\t\t{:.2f} %".format(final_test_acc))
        with open(test_log_file_name, 'a') as test_file:
            test_file.write("{:.6f}".format(final_test_loss) + "\t{:.2f}".format(final_test_acc) + "\n")
        print("--------\n")

if __name__ == '__main__':
    mnist = False
    cifar10 = True

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    if mnist:
        print 'loading config for mnist'
        network = build_cnn_mnist(input_var)
        dataset = load_mnist()
        num_epochs = 500
        batch_size = 500
        optim = 'adamax'
        weight_decay = False
    elif cifar10:
        print 'loading config for cifar10'
        network = build_cnn_deep_cifar(input_var)
        dataset = load_cifar10()
        num_epochs = 82
        batch_size = 128
        optim = 'adamax'
        weight_decay = True

    args = {
            'network': network, \
            'dataset': dataset, \
            'input_var': input_var, \
            'target_var': target_var, \
            'optim': optim, \
            'dataset': dataset, \
            'num_epochs': num_epochs, \
            'weight_decay': weight_decay, \
            'batch_size': batch_size
            }
    main(args)
