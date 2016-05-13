from theano import tensor
from blocks.bricks import Initializable, Tanh, Logistic
from blocks.bricks.base import application, lazy
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.bricks.recurrent import BaseRecurrent, recurrent


# model_type:
# 1: Regular RNN
# 2: Regular RNN + Regular Dropout
# 3: Res RNN
# 4: Res RNN + Regular Dropout
# 5: Res RNN + New Dropout
# 6: Regular RNN + New Dropout
class DropRecurrent(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, model_type=1, **kwargs):
        self.dim = dim
        self.model_type = model_type
        children = [activation]
        kwargs.setdefault('children', []).extend(children)
        super(DropRecurrent, self).__init__(**kwargs)

    @property
    def W(self):
        return self.parameters[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (DropRecurrent.apply.sequences +
                    DropRecurrent.apply.states):
            return self.dim
        return super(DropRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name="W"))
        add_role(self.parameters[0], WEIGHT)
        self.parameters.append(shared_floatx_zeros((self.dim,),
                                                   name="initial_state"))
        add_role(self.parameters[1], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'drops', 'mask'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs, drops, states, mask=None):
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)

        # In training time drops is either 0 or 1
        # In test time drops is 0.5 (if drop_prob=0.5)
        if self.model_type == 2:
            next_states = next_states * drops
        elif self.model_type == 3:
            next_states = next_states + states
        elif self.model_type == 4:
            next_states = next_states + states
            next_states = next_states * drops
        elif self.model_type == 5:
            next_states = next_states * drops + states
        elif self.model_type == 6:
            next_states = next_states * drops + (1 - drops) * states

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.repeat(self.parameters[1][None, :], batch_size, 0)


class DropLSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 model_type=1, **kwargs):
        self.dim = dim
        self.model_type = model_type

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = [self.activation, self.gate_activation]
        kwargs.setdefault('children', []).extend(children)
        super(DropLSTM, self).__init__(**kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells', 'drops']:
            return self.dim
        if name == 'mask':
            return 0
        return super(DropLSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4 * self.dim),
                                          name='W_state')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.parameters = [
            self.W_state, self.initial_state_, self.initial_cells]

    def _initialize(self):
        for weights in self.parameters[:1]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'drops', 'mask'],
               states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, drops, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no * self.dim: (no + 1) * self.dim]

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = self.gate_activation.apply(slice_last(activation, 0))
        forget_gate = self.gate_activation.apply(slice_last(activation, 1))
        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(slice_last(activation, 3))
        next_states = out_gate * self.activation.apply(next_cells)

        # In training time drops is either 0 or 1
        # In test time drops is 0.5 (if drop_prob=0.5)
        if self.model_type == 2:
            next_states = next_states * drops
            next_cells = next_cells * drops
        elif self.model_type == 3:
            next_states = next_states + states
        elif self.model_type == 4:
            next_states = (next_states + states) * drops
        elif self.model_type == 5:
            next_states = next_states * drops + states
        elif self.model_type == 6:
            next_states = next_states * drops + (1 - drops) * states
            next_cells = next_cells * drops + (1 - drops) * cells

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]
