import theano
import theano.tensor as T
import numpy as np
from blocks.initialization import NdarrayInitialization
import logging
from blocks.extensions import SimpleExtension
import cPickle
import gzip
import os
logger = logging.getLogger('main.utils')


class SaveLog(SimpleExtension):
    def __init__(self, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        current_row = self.main_loop.log.current_row
        logger.info("\nIter:%d" % epoch)
        for element in current_row:
            logger.info(str(element) + ":%f" % current_row[element])


class SaveParams(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, early_stop_var, model, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        self.early_stop_var = early_stop_var
        self.save_path = save_path
        params_dicts = model.get_parameter_dict()
        self.params_names = params_dicts.keys()
        self.params_values = params_dicts.values()
        self.to_save = {}
        self.best_value = None
        self.add_condition(('after_training',), self.save)
        self.add_condition(('on_interrupt',), self.save)
        self.add_condition(('after_epoch',), self.do)

    def save(self, which_callback, *args):
        to_save = {}
        for p_name, p_value in zip(self.params_names, self.params_values):
            to_save[p_name] = p_value.get_value()
        path = self.save_path + '/trained_params'
        np.savez_compressed(path, **to_save)

    def do(self, which_callback, *args):
        val = self.main_loop.log.current_row[self.early_stop_var]
        if self.best_value is None or val < self.best_value:
            self.best_value = val
            to_save = {}
            for p_name, p_value in zip(self.params_names, self.params_values):
                to_save[p_name] = p_value.get_value()
            path = self.save_path + '/trained_params_best'
            np.savez_compressed(path, **to_save)


class Glorot_0(NdarrayInitialization):
    def generate(self, rng, shape):
        if len(shape) == 2:
            input_size, output_size = shape
            high = np.sqrt(6) / np.sqrt(input_size + output_size)
            m = rng.uniform(-high, high, size=shape)
            if shape == (100, 400):
                high = np.sqrt(6) / np.sqrt(200)
                mi = rng.uniform(-high, high, size=(100, 100))
                mf = rng.uniform(-high, high, size=(100, 100))
                mc = np.identity(100)
                mo = rng.uniform(-high, high, size=(100, 100))
                m = np.hstack([mi, mf, mc, mo])
            else:
                import ipdb
                ipdb.set_trace
        return m.astype(theano.config.floatX)


class Glorot(NdarrayInitialization):
    def generate(self, rng, shape):
        if len(shape) == 2:
            if shape[0] == shape[1]:
                M = rng.randn(*shape).astype(theano.config.floatX)
                Q, R = np.linalg.qr(M)
                m = Q * np.sign(np.diag(R))

            else:

                M1 = rng.randn(shape[0], shape[0]).astype(theano.config.floatX)
                M2 = rng.randn(shape[1], shape[1]).astype(theano.config.floatX)

                # QR decomposition of matrix with entries in N(0, 1) is random
                Q1, R1 = np.linalg.qr(M1)
                Q2, R2 = np.linalg.qr(M2)
                # Correct that np doesn't force diagonal of R to be non-negative
                Q1 = Q1 * np.sign(np.diag(R1))
                Q2 = Q2 * np.sign(np.diag(R2))

                n_min = min(shape[0], shape[1])
                m = np.dot(Q1[:, :n_min], Q2[:n_min, :])

                # if shape == (100, 400):
                if False:
                    M = rng.randn(100, 100).astype(theano.config.floatX)
                    Q, R = np.linalg.qr(M)
                    mi = Q * np.sign(np.diag(R))

                    M = rng.randn(100, 100).astype(theano.config.floatX)
                    Q, R = np.linalg.qr(M)
                    mf = Q * np.sign(np.diag(R))

                    M = rng.randn(100, 100).astype(theano.config.floatX)
                    Q, R = np.linalg.qr(M)
                    mo = Q * np.sign(np.diag(R))

                    mc = np.identity(100)
                    m = np.hstack([mi, mf, mc, mo])

        return m.astype(theano.config.floatX)


def print_num_params(params):
    shapes = []
    for param in params:
        # shapes.append((param.name, param.eval().shape))
        shapes.append(np.prod(list(param.eval().shape)))
    print "Total number of parameters: " + str(np.sum(shapes))


def load_data(dataset):
    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    rval = [train_set, valid_set, test_set]
    return rval
