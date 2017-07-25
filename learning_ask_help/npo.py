from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from batch_polopt import BatchPolopt

from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            gate_optimizer=None, 
            optimizer_args=None,
            step_size=0.01,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer

        if gate_optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            gate_optimizer = PenaltyLbfgsOptimizer(**optimizer_args)

        self.gate_optimizer = gate_optimizer

        self.step_size = step_size
        super(NPO, self).__init__(**kwargs)


    ### discrete initialisations - for the gate beta(s)
    @overrides
    def init_opt(self):

        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )

        ### action_var for the continuous pi(s)
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )

        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )

        discrete_action_var = tensor_utils.new_tensor(
            'discrete_action',
            ndim=2,
            dtype=tf.float32,
        )

        ### distribution for pi(s)
        dist = self.policy.distribution

        ### distribution for beta(s)
        ### TO DO HERE - it should NOT be a Gaussian Distribution, 
        ### but instead for discrete, it shoud be a Binomial or Categorical Distribution?
        discrete_dist = self.policy.discrete_distribution


        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]


        discrete_old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in discrete_dist.dist_info_specs
            }
        discrete_old_dist_info_vars_list = [discrete_old_dist_info_vars[k] for k in discrete_dist.dist_info_keys]



        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]


        discrete_state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        discrete_state_info_vars_list = [discrete_state_info_vars[k] for k in self.policy.state_info_keys]



        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None


        if is_recurrent:
            discrete_valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            discrete_valid_var = None


        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        # Likelihood ratio between policy distribution
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)


        ### TO DO HERE : policy.dist_info_sym - NEED TO MAKE FOR DISCRETE IN hierarchical_gaussian_mlp_policy
        discrete_dist_info_vars = self.policy.dist_info_sym(obs_var, discrete_state_info_vars)
        discrete_kl = discrete_dist.kl_sym(discrete_old_dist_info_vars, discrete_dist_info_vars)
        # Likelihood ratio between policy distribution
        discrete_lr = discrete_dist.likelihood_ratio_sym(discrete_action_var, discrete_old_dist_info_vars, discrete_dist_info_vars)




        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = - tf.reduce_mean(lr * advantage_var)


        if is_recurrent:
            discrete_mean_kl = tf.reduce_sum(discrete_kl * discrete_valid_var) / tf.reduce_sum(discrete_valid_var)
            discrete_surr_loss = - tf.reduce_sum(discrete_lr * advantage_var * discrete_valid_var) / tf.reduce_sum(discrete_valid_var)
        else:
            discrete_mean_kl = tf.reduce_mean(discrete_kl)
            discrete_surr_loss = - tf.reduce_mean(discrete_lr * advantage_var)




        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list


        discrete_input_list = [
                         obs_var,
                         discrete_action_var,
                         advantage_var,
                     ] + discrete_state_info_vars_list + discrete_old_dist_info_vars_list


        if is_recurrent:
            input_list.append(valid_var)

        if is_recurrent:
            discrete_input_list.append(discrete_valid_var)


        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )


        self.gate_optimizer.update_opt(
            loss=discrete_surr_loss,
            target=self.policy,
            leq_constraint=(discrete_mean_kl, self.step_size),
            inputs=discrete_input_list,
            constraint_name="discrete_mean_kl"
        )

        return dict()




    ### for optimizing gate beta(s) 
    @overrides
    def optimize_policy(self, itr, samples_data):

        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))


        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)


        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)


        logger.log("Computing loss before")
        loss_before = self.gate_optimizer.loss(all_input_values)


        logger.log("Computing KL before")
        mean_kl_before = self.gate_optimizer.constraint_val(all_input_values)


        logger.log("Optimizing")
        self.gate_optimizer.optimize(all_input_values)

        logger.log("Computing KL after")
        mean_kl = self.gate_optimizer.constraint_val(all_input_values)


        logger.log("Computing loss after")
        loss_after = self.gate_optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()



    #### optimize pi(s) as usual
    @overrides
    def optimize_agent_policy(self, itr, agent_samples_data):

        agent_only_all_input_values = tuple(ext.extract(
            agent_samples_data,
            "observations", "actions", "advantages"
        ))

        agent_only_infos = agent_samples_data["agent_infos"]
        agent_only_state_info_list = [agent_only_infos[k] for k in self.policy.state_info_keys]
        agent_only_dist_info_list = [agent_only_infos[k] for k in self.policy.distribution.dist_info_keys]
        agent_only_all_input_values += tuple(agent_only_state_info_list) + tuple(agent_only_dist_info_list)

        if self.policy.recurrent:
            agent_only_all_input_values += (samples_data["valids"],)


        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(agent_only_all_input_values)

        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(agent_only_all_input_values)

        logger.log("Optimizing")
        self.optimizer.optimize(agent_only_all_input_values)

        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(agent_only_all_input_values)

        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(agent_only_all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()





    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
