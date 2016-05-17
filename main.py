import theano
import os
import numpy as np
import theano.tensor as T
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.model import Model
from blocks.algorithms import (GradientDescent, RMSProp, StepClipping,
                               RemoveNotFinite, CompositeRule)
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.bricks import Linear, Tanh, Softmax
from blocks.initialization import Constant
from blocks.main_loop import MainLoop
from blocks.extensions import Printing
from blocks.graph import ComputationGraph
import logging
from blocks.monitoring import aggregation
from utils import SaveLog, SaveParams, print_num_params, Glorot
from datasets import get_stream
from models import DropRecurrent, DropLSTM, DropGRU
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
floatX = theano.config.floatX


batch_size = 100
x_dim = 1
y_dim = 10
h_dim = 100
update_prob = 0.85
is_LSTM = False
model_type = 3
save_path = 'Test_type_' + str(model_type)
load_path = 'GRU_type_3'

print 'Building model ...'
# shape: T x B x F
x = T.tensor3('x', dtype=floatX)
# shape: T x B x F
drops = T.tensor3('drops')
# shape: B
y = T.lvector('y')
# shape: T
is_for_test = T.vector('is_for_test')

if is_LSTM:
    output_dim = 4 * h_dim
else:
    output_dim = 3 * h_dim
x_to_h1 = Linear(name='x_to_h1',
                 input_dim=x_dim,
                 output_dim=output_dim)
pre_rnn = x_to_h1.apply(x)
if is_LSTM:
    rnn = DropLSTM(dim=h_dim, model_type=model_type,
                   update_prob=update_prob, name="rnn")
    h1, c1 = rnn.apply(pre_rnn, drops, is_for_test)
else:
    rnn = DropGRU(dim=h_dim, model_type=model_type,
                  update_prob=update_prob, name="rnn")
    h1, sd = rnn.apply(pre_rnn[:, :, :h_dim],
                   pre_rnn[:, :, h_dim:],
                   drops, is_for_test)
h1_to_o = Linear(name='h1_to_o',
                 input_dim=h_dim,
                 output_dim=y_dim)
pre_softmax = h1_to_o.apply(h1)
softmax = Softmax()
shape = pre_softmax.shape
softmax_out = softmax.apply(pre_softmax.reshape((-1, y_dim)))
softmax_out = softmax_out.reshape(shape)
softmax_out.name = 'softmax_out'

# comparing only last time-step
cost = CategoricalCrossEntropy().apply(y, softmax_out[-1])
cost.name = 'CrossEntropy'
error_rate = MisclassificationRate().apply(y, softmax_out[-1])
error_rate.name = 'error_rate'

# Initialization
for brick in (x_to_h1, h1_to_o, rnn):
    brick.weights_init = Glorot()
    brick.biases_init = Constant(0)
    brick.initialize()

train_stream = get_stream('train', batch_size, h_dim, False)
data = train_stream.get_epoch_iterator(as_dict=True).next()
cg = ComputationGraph(cost)
f = theano.function(cg.inputs, cost)
print f(data['y'], data['x'], data['is_for_test'], data['drops'])

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
rms_prop = RMSProp(learning_rate=1e-3, decay_rate=0.5)
step_rule = CompositeRule([clipping, rms_prop, rm_non_finite])
algorithm = GradientDescent(
    cost=cost,
    parameters=params,
    step_rule=step_rule)

# train_stream, valid_stream = get_seq_mnist_streams(
#    h_dim, batch_size, update_prob)
train_stream = get_stream('train', batch_size, h_dim, False)
train_stream_evaluation = get_stream('train', batch_size, h_dim, True)
valid_stream = get_stream('valid', batch_size, h_dim, True)

if load_path:
    with open(load_path + '/trained_params_best.npz') as f:
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
    f = theano.function([x, drops, is_for_test, y], error_rate)
    data_train = train_stream.get_epoch_iterator(as_dict=True).next()
    data_train_eval = train_stream_evaluation.get_epoch_iterator(
        as_dict=True).next()
    data_valid = valid_stream.get_epoch_iterator(as_dict=True).next()
    print f(data_train['x'], data_train['drops'],
            data['is_for_test'], data_train['y'])
    print f(data_train_eval['x'], data_train_eval['drops'],
            data['is_for_test'], data_train_eval['y'])
    print f(data_valid['x'], data_valid['drops'], data['is_for_test'], data_valid['y'])
    import ipdb; ipdb.set_trace()

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
