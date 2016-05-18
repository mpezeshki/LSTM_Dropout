import sys
import theano
import os
import numpy as np
import theano.tensor as T
from blocks.model import Model
from blocks.algorithms import (GradientDescent, Adam, StepClipping,
                               RemoveNotFinite, CompositeRule)
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.bricks import Linear, Tanh
from blocks.initialization import Constant
from blocks.main_loop import MainLoop
from blocks.extensions import Printing
from blocks.graph import ComputationGraph
import logging
from blocks.monitoring import aggregation
from utils import SaveLog, SaveParams, print_num_params, Glorot
from utils import crossentropy_lastaxes, softmax_lastaxis
from datasets import get_ptb_stream
from models import DropLSTM
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
floatX = theano.config.floatX


if __name__ == "__main__":
    batch_size = 64
    x_dim = 50
    y_dim = 50
    h_dim = 1000
    update_prob = 0.9
    model_type = int(sys.argv[1])
    print model_type
    if model_type == 1:
        save_path = 'LSTM_PTB_LSTM'
    if model_type == 2:
        save_path = 'LSTM_PTB_Dropout'
    if model_type == 3:
        save_path = 'LSTM_PTB_Elephant'
    if model_type == 4:
        save_path = 'LSTM_PTB_ZoneOut'

    print 'Building model ...'
    # shape: T x B x F
    x = T.tensor3('x', dtype=floatX)
    # shape: T x B x F'
    drops = T.tensor3('drops')
    # shape: T-1 x B x F
    x, y = x[:-1], x[1:]

    x_to_h1 = Linear(name='x_to_h1',
                     input_dim=x_dim,
                     output_dim=4 * h_dim)
    pre_rnn = x_to_h1.apply(x)
    rnn = DropLSTM(activation=Tanh(), dim=h_dim,
                   model_type=model_type, name="rnn")
    h1, c1 = rnn.apply(pre_rnn, drops)
    h1_to_o = Linear(name='h1_to_o',
                     input_dim=h_dim,
                     output_dim=y_dim)
    pre_softmax = h1_to_o.apply(h1)
    yhat = softmax_lastaxis(pre_softmax)
    errors = T.neq(T.argmax(y, axis=y.ndim - 1),
                   T.argmax(yhat, axis=yhat.ndim - 1))
    cross_entropies = crossentropy_lastaxes(yhat, y)
    error_rate = errors.mean().copy(name="error_rate")
    cross_entropy = cross_entropies.mean().copy(name="cross_entropy")
    cost = cross_entropy.copy(name="cost")

    # Initialization
    for brick in (x_to_h1, h1_to_o, rnn):
        brick.weights_init = Glorot()
        brick.biases_init = Constant(0)
        brick.initialize()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = save_path + '/log.txt'
    fh = logging.FileHandler(filename=log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    print 'Bulding training process...'
    model = Model(cost)
    params = ComputationGraph(cost).parameters
    print_num_params(params)
    clipping = StepClipping(threshold=np.cast[floatX](1.0))
    # Momentum(learning_rate=args.learning_rate, momentum=0.9)
    rm_non_finite = RemoveNotFinite()
    rms_prop = Adam(learning_rate=2 * 1e-3)
    step_rule = CompositeRule([clipping, rms_prop, rm_non_finite])
    algorithm = GradientDescent(
        cost=cost,
        parameters=params,
        step_rule=step_rule)

    # train_stream, valid_stream = get_seq_mnist_streams(
    #    h_dim, batch_size, update_prob)
    train_stream = get_ptb_stream(
        'train', batch_size, 100, update_prob, h_dim, False)
    train_stream_evaluation = get_ptb_stream(
        'train', batch_size, 100, update_prob, h_dim, True)
    valid_stream = get_ptb_stream(
        'valid', batch_size, 100, update_prob, h_dim, True)

    if False:
        with open(save_path + '/trained_params_best.npz') as f:
            loaded = np.load(f)
            params_dicts = model.get_parameter_dict()
            params_names = params_dicts.keys()
            for param_name in params_names:
                param = params_dicts[param_name]
                # '/f_6_.W' --> 'f_6_.W'
                slash_index = param_name.find('/')
                param_name = param_name[slash_index + 1:]
                if param.get_value().shape == loaded[param_name].shape:
                    print param
                    param.set_value(loaded[param_name])
                else:
                    print param_name
        f = theano.function([x, drops, y], error_rate)
        data_train = train_stream.get_epoch_iterator(as_dict=True).next()
        data_valid = valid_stream.get_epoch_iterator(as_dict=True).next()
        print f(data_train['x'], data_train['drops'], data_train['y'])
        print f(data_valid['x'], data_valid['drops'], data_valid['y'])

    monitored_variables = [
        cost, error_rate,
        aggregation.mean(algorithm.total_gradient_norm)]

    monitor_train_cost = TrainingDataMonitoring(
        monitored_variables,
        prefix="train",
        after_epoch=True)

    monitor_train_cost_evaluation = DataStreamMonitoring(
        monitored_variables,
        data_stream=train_stream_evaluation,
        prefix="train_evaluation",
        after_epoch=True)

    monitor_valid_cost = DataStreamMonitoring(
        monitored_variables,
        data_stream=valid_stream,
        prefix="valid",
        after_epoch=True)

    main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                         extensions=[monitor_train_cost,
                                     monitor_train_cost_evaluation,
                                     monitor_valid_cost,
                                     Printing(),
                                     SaveLog(after_epoch=True),
                                     SaveParams('valid_error_rate',
                                                model, save_path,
                                                after_epoch=True)],
                         model=model)

    print 'Starting training ...'
    main_loop.run()
