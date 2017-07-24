# import lasagne
# import lasagne.layers as L
# import lasagne.nonlinearities as NL
# import lasagne.init as LI
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import MLP
from shared_network import HierarchicalMLP
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.misc import tensor_utils

import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.layers import batch_norm
from sandbox.rocky.tf.spaces.discrete import Discrete
import tensorflow as tf


class LayeredDeterministicMLPPolicy(Policy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            oracle_policy,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            output_nonlinearity_binary=tf.nn.softmax,
            output_dim_binary=2,
            prob_network=None,

            bn=False):
        Serializable.quick_init(self, locals())


        with tf.variable_scope(name):
            if prob_network is None:

                prob_network = HierarchicalMLP(
                    input_shape=(env_spec.observation_space.flat_dim,),
                    output_dim=env_spec.action_space.flat_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    output_nonlinearity_binary=output_nonlinearity_binary,
                    output_dim_binary=output_dim_binary,
                    # batch_normalization=True,
                    name="prob_network",
                )

            self.oracle_policy = oracle_policy
            self._l_prob = prob_network.output_layer
            self._l_obs = prob_network.input_layer
            self._f_prob = tensor_utils.compile_function(
                [prob_network.input_layer.input_var],
                L.get_output(prob_network.output_layer, deterministic=True)
            )

            self._f_prob_binary = tensor_utils.compile_function(
                [prob_network.input_layer.input_var],
                L.get_output(prob_network.output_layer_binary, deterministic=True)
            )

            ## use tf.round here?


        self.output_layer_binary = prob_network.output_layer_binary
        #self.output_layer_binary = tf.round(prob_network.output_layer_binary)

        self.binary_output = L.get_output(prob_network.output_layer_binary, deterministic=True)
        self.prob_network = prob_network

        # Note the deterministic=True argument. It makes sure that when getting
        # actions from single observations, we do not update params in the
        # batch normalization layers.
        # TODO: this doesn't currently work properly in the tf version so we leave out batch_norm
        super(LayeredDeterministicMLPPolicy, self).__init__(env_spec)
        LayersPowered.__init__(self, [prob_network.output_layer, prob_network.output_layer_binary])
        # LayersPowered.__init__(self, [prob_network.output_layer_binary])



    @property
    def vectorized(self):
        return True

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        action = self._f_prob([flat_obs])[0]
        binary_action = self._f_prob_binary([flat_obs])[0]

        return action, dict()

    @overrides
    def get_action_with_binary(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        action = self._f_prob([flat_obs])[0]
        binary_action = self._f_prob_binary([flat_obs])[0]

        return action, binary_action, dict()


    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        actions = self._f_prob(flat_obs)
        return actions, dict()


    @overrides
    def get_binary_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        binary_actions = self._f_prob_binary(flat_obs)
        return binary_actions, dict()   


    def get_action_oracle_sym(self, obs_var):
        oracle_policy_sym = L.get_output(self.oracle_policy.prob_network.output_layer, obs_var)
        return tf.stop_gradient(oracle_policy_sym)

    def get_action_binary_gate_sym(self, obs_var):
        return L.get_output(self.prob_network.output_layer_binary, obs_var)


    def get_novice_policy_sym(self, obs_var):
        return L.get_output(self.prob_network.output_layer, obs_var)
        #
        # ### TO DO : out_bin right now is a soft gate and NOT a hard gate
        # out_bin = L.get_output(self.prob_network.output_layer_binary, obs_var)
        # oracle_policy_sym = L.get_output(self.oracle_policy.prob_network.output_layer, obs_var)
        #
        # ### here, when out_bin is 1, stop gradients with agent samples
        # ### and allow oracle samples to flow for training beta(s)
        #
        # ### have stop gradient for oracle samples on the agent policy
        # ### we dont want to train pi(s) with the oracle samples
        # return tf.stop_gradient() * (out_bin) + (1.0-out_bin) * tf.stop_gradient(oracle_policy_sym)


    # ## obs_var here is agent_samples only
    # def get_action_sym(self, obs_var):
    #
    #     ### TO DO : out_bin right now is a soft gate and NOT a hard gate
    #     out_bin = L.get_output(self.prob_network.output_layer_binary, obs_var)
    #
    #     oracle_policy_sym = L.get_output(self.oracle_policy.prob_network.output_layer, obs_var)
    #
    #     ### Why hold out the oracle samples when training pi(s) anyway???
    #
    #     ### explanation for line below:
    #     ### we do not have stop gradients for agent policy here, becase here we have agent samples
    #     ### and we train the agent policy with the agent samples only
    #     ### here beta(s) policy is also updated with the agent samples
    #     ### and above, beta(s) is updated with the oracle samples
    #     return (out_bin) * L.get_output(self.prob_network.output_layer, obs_var)  + (1.0-out_bin) * tf.stop_gradient(oracle_policy_sym)
    #     #return L.get_output(self.prob_network.output_layer, obs_var)
