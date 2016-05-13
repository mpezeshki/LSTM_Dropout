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
from blocks.bricks import Linear, Rectifier, Softmax
from blocks.initialization import Constant
from blocks.main_loop import MainLoop
from blocks.extensions import Printing
from blocks.graph import ComputationGraph
import logging
from blocks.monitoring import aggregation
from utils import SaveLog, SaveParams, print_num_params, Glorot
from datasets import get_seq_mnist_streams
from models import DropRecurrent, DropLSTM
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
floatX = theano.config.floatX


batch_size = 100
x_dim = 1
y_dim = 10
h_dim = 100
update_prob = 0.8
model_type = 1
save_path = 'path_model_type_' + str(model_type)

print 'Building model ...'
# shape: T x B x F
x = T.tensor3('x', dtype=floatX)
# shape: T x B x F
drops = T.tensor3('drops')
# shape: B
y = T.lvector('y')


x_to_h1 = Linear(name='x_to_h1',
                 input_dim=x_dim,
                 output_dim=4 * h_dim)
pre_rnn = x_to_h1.apply(x)
rnn = DropLSTM(activation=Rectifier(), dim=h_dim,
               model_type=model_type, name="rnn")
h1, c1 = rnn.apply(pre_rnn, drops)
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

if not os.path.exists(save_path):
    os.makedirs(save_path)
log_path = save_path + '/log.txt'
fh = logging.FileHandler(filename=log_path)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

print 'Bulding training process...'
params = ComputationGraph(cost).parameters
print_num_params(params)
clipping = StepClipping(threshold=np.cast[floatX](1.0))
# Momentum(learning_rate=args.learning_rate, momentum=0.9)
rms_prop = RMSProp(learning_rate=1e-3, decay_rate=0.5)
step_rule = CompositeRule([clipping, rms_prop])
algorithm = GradientDescent(
    cost=cost,
    parameters=params,
    step_rule=step_rule)

train_stream, valid_stream = get_seq_mnist_streams(
    h_dim, batch_size, update_prob)

monitored_variables = [
    cost, error_rate,
    aggregation.mean(algorithm.total_gradient_norm)]

monitor_train_cost = TrainingDataMonitoring(monitored_variables,
                                            prefix="train",
                                            after_epoch=True)

monitor_valid_cost = DataStreamMonitoring(monitored_variables,
                                          data_stream=valid_stream,
                                          prefix="valid",
                                          after_epoch=True)

model = Model(cost)
main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     extensions=[monitor_train_cost,
                                 monitor_valid_cost,
                                 Printing(),
                                 SaveLog(after_epoch=True),
                                 SaveParams('valid_error_rate',
                                            model, save_path,
                                            after_epoch=True)],
                     model=model)

print 'Starting training ...'
main_loop.run()
