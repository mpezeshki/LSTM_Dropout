import numpy as np
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
# 7: Recurrent Dropout without Memory Loss
# 8: Moon et al: one cell only
class DropRecurrent(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, model_type, update_prob, **kwargs):
        self.dim = dim
        self.model_type = model_type
        self.update_prob = update_prob
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
    def __init__(self, dim, model_type, update_prob,
                 activation=None, gate_activation=None, **kwargs):
        self.dim = dim
        self.model_type = model_type
        self.update_prob = update_prob

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

    @recurrent(sequences=['inputs', 'drops', 'is_for_test', 'mask'],
               states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, drops, is_for_test, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no * self.dim: (no + 1) * self.dim]

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = self.gate_activation.apply(slice_last(activation, 0))
        forget_gate_input = slice_last(activation, 1)
        forget_gate = self.gate_activation.apply(
            forget_gate_input + tensor.ones_like(forget_gate_input))
        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(slice_last(activation, 3))
        next_states = out_gate * self.activation.apply(next_cells)

        sampled_drops = ((1.0 - is_for_test) * drops < self.update_prob +
                         is_for_test * tensor.ones_like(drops) * self.update_prob)

        # In training time sampled_drops is either 0 or 1
        # In test time sampled_drops is 0.5 (if drop_prob=0.5)
        if self.model_type == 2:
            next_states = next_states * sampled_drops
            next_cells = next_cells * sampled_drops
        elif self.model_type == 3:
            next_states = (next_states + states) / 2
        elif self.model_type == 4:
            next_states = (next_states + states) * sampled_drops
        elif self.model_type == 5:
            next_states = next_states * sampled_drops + states
        elif self.model_type == 6:
            next_states = next_states * sampled_drops + (1 - sampled_drops) * states
            next_cells = next_cells * sampled_drops + (1 - sampled_drops) * cells
        elif self.model_type == 7:
            next_cells = (
                forget_gate * cells +
                in_gate * sampled_drops * self.activation.apply(slice_last(activation, 2)))
        elif self.model_type == 8:
            next_cells = next_cells * sampled_drops

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


class DropGRU(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, model_type, update_prob,
                 activation=None, gate_activation=None, **kwargs):
        self.dim = dim
        self.update_prob = update_prob

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation
        self.model_type = model_type

        children = [activation, gate_activation]
        kwargs.setdefault('children', []).extend(children)
        super(DropGRU, self).__init__(**kwargs)

    @property
    def state_to_state(self):
        return self.parameters[0]

    @property
    def state_to_gates(self):
        return self.parameters[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        if name == 'gate_inputs':
            return 2 * self.dim
        return super(DropGRU, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        self.parameters.append(shared_floatx_zeros((self.dim,),
                               name="initial_state"))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)
        add_role(self.parameters[2], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        state_to_update = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        state_to_reset = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        self.state_to_gates.set_value(
            np.hstack([state_to_update, state_to_reset]))

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs', 'drops', 'is_for_test'],
               states=['states'], outputs=['states', 'sampled_drops'], contexts=[])
    def apply(self, inputs, gate_inputs, drops, is_for_test, states, mask=None):
        gate_values = self.gate_activation.apply(
            states.dot(self.state_to_gates) + gate_inputs)
        update_values = gate_values[:, :self.dim]
        reset_values = gate_values[:, self.dim:]
        states_reset = states * reset_values

        sampled_drops = ((1.0 - is_for_test) * drops < self.update_prob +
                         is_for_test * tensor.ones_like(drops) * self.update_prob)

        if self.model_type == 1:
            next_states = self.activation.apply(
                states_reset.dot(self.state_to_state) + inputs)
            next_states = (next_states * update_values +
                           states * (1 - update_values))
        elif self.model_type == 2:
            next_states = self.activation.apply(
                states_reset.dot(self.state_to_state) + inputs)
            next_states = (next_states * update_values +
                           states * (1 - update_values))
            next_states = next_states * sampled_drops
        elif self.model_type == 3:
            next_states = self.activation.apply(
                states_reset.dot(self.state_to_state) + inputs)
            next_states = (sampled_drops * next_states * update_values +
                           states * (1 - update_values))
        elif self.model_type == 4:
            next_states = self.activation.apply(
                states_reset.dot(self.state_to_state) + inputs)
            sampled_drops = ((1.0 - is_for_test) * drops < update_values +
                             is_for_test * tensor.ones_like(drops) * update_values)
            next_states = (sampled_drops * next_states +
                           states * (1 - update_values))

        elif self.model_type == 5:
            next_states = self.activation.apply(
                states_reset.dot(self.state_to_state) + inputs)
            sampled_drops = ((1.0 - is_for_test) * drops < update_values +
                             is_for_test * tensor.ones_like(drops) * update_values)
            next_states = (sampled_drops * next_states * update_values +
                           states * (1 - update_values))

        elif self.model_type == 6:
            next_states = self.activation.apply(
                states_reset.dot(self.state_to_state) + inputs)
            next_states = (next_states * update_values +
                           states * (1 - update_values))

            next_states = sampled_drops * next_states + (1 - sampled_drops) * states

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states, sampled_drops

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.parameters[2][None, :], batch_size, 0)]
