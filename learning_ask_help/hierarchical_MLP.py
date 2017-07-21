import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import numpy as np
import itertools
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.core.layers_powered import LayersPowered


"""
Need to make changes to the network directly here
"""
class HierarchicalMLP(LayersPowered, Serializable):
    def __init__(self, name, output_dim, output_dim_binary, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, output_nonlinearity_binary, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer(),
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer(),
                 input_var=None, input_layer=None, input_shape=None, batch_normalization=False, weight_normalization=False,
                 ):

        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                l_hid = L.batch_norm(l_hid)
            for idx, hidden_size in enumerate(hidden_sizes):

                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_normalization=weight_normalization
                )
                if batch_normalization:
                    l_hid = L.batch_norm(l_hid)
                self._layers.append(l_hid)


            l_hid_binary = L.DenseLayer(
                l_hid, 
                num_units = hidden_size,
                nonlinearity = hidden_nonlinearity,
                name="hidden_binary",
                W=hidden_W_init,
                b=hidden_b_init,
                weight_normalization=weight_normalization
            )

            l_out_binary = L.DenseLayer(
                l_hid_binary,
                num_units=output_dim_binary,
                nonlinearity=output_nonlinearity_binary,
                name="output_binary",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )

            self._layers.append(l_out_binary)

            l_hid_out = L.DenseLayer(
                l_hid,
                num_units = hidden_size,
                nonlinearity = hidden_nonlinearity,
                name="hidden_final",
                W=hidden_W_init,
                b=hidden_b_init,
                weight_normalization=weight_normalization
            )


            l_out = L.DenseLayer(
                l_hid_out,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )

            if batch_normalization:
                l_out = L.batch_norm(l_out)

            self._layers.append(l_out)
            self._l_in = l_in

            self._l_out = l_out
            
            self._l_out_binary = l_out_binary

            self._output_binary = L.get_output(l_out_binary)
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, [l_out, l_out_binary])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def output_layer_binary(self):
        return self._l_out_binary

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

    @property
    def output_binary(self):
        return self._output_binary

