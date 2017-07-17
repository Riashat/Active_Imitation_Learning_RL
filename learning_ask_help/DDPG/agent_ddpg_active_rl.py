# FROM: https://raw.githubusercontent.com/shaneshixiang/rllabplusplus/master/sandbox/rocky/tf/algos/ddpg.py
from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import special
from sandbox.rocky.tf.misc import tensor_utils
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
from rllab.misc import ext
import rllab.misc.logger as logger
#import pickle as pickle
import numpy as np
import pyprind
import tensorflow as tf
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
#from sandbox.rocky.tf.core.parameterized import suppress_params_loading
from rllab.core.serializable import Serializable
from sampling_utils import SimpleReplayPool
from sandbox.rocky.tf.core.network import MLP
import sandbox.rocky.tf.core.layers as L


class DDPG(RLAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    ##originally - min pool size = 10000
    def __init__(
            self,
            env,
            policy,
            oracle_policy,
            qf,
            agent_strategy,
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size = 500,
            replay_pool_size=1000000,
            replacement_prob=1.0,
            discount=0.99,
            max_path_length=250,
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            policy_weight_decay=0,
            policy_update_method='adam',
            policy_learning_rate=1e-3,
            policy_updates_ratio=1.0,
            eval_samples=10000,
            soft_target=True,
            soft_target_tau=0.001,
            n_updates_per_sample=1,
            scale_reward=1.0,
            include_horizon_terminal_transitions=False,
            plot=False,
            pause_for_plot=False):
        """
        :param env: Environment
        :param policy: Policy
        :param qf: Q function
        :param es: Exploration strategy
        :param batch_size: Number of samples for each minibatch.
        :param n_epochs: Number of epochs. Policy will be evaluated after each epoch.
        :param epoch_length: How many timesteps for each epoch.
        :param min_pool_size: Minimum size of the pool to start training.
        :param replay_pool_size: Size of the experience replay pool.
        :param discount: Discount factor for the cumulative return.
        :param max_path_length: Discount factor for the cumulative return.
        :param qf_weight_decay: Weight decay factor for parameters of the Q function.
        :param qf_update_method: Online optimization method for training Q function.
        :param qf_learning_rate: Learning rate for training Q function.
        :param policy_weight_decay: Weight decay factor for parameters of the policy.
        :param policy_update_method: Online optimization method for training the policy.
        :param policy_learning_rate: Learning rate for training the policy.
        :param eval_samples: Number of samples (timesteps) for evaluating the policy.
        :param soft_target_tau: Interpolation parameter for doing the soft target update.
        :param n_updates_per_sample: Number of Q function and policy updates per new sample obtained
        :param scale_reward: The scaling factor applied to the rewards when training
        :param include_horizon_terminal_transitions: whether to include transitions with terminal=True because the
        horizon was reached. This might make the Q value back up less stable for certain tasks.
        :param plot: Whether to visualize the policy performance after each eval_interval.
        :param pause_for_plot: Whether to pause before continuing when plotting.
        :return:
        """
        self.env = env
        self.policy = policy
        self.oracle_policy = oracle_policy
        self.qf = qf
        self.agent_strategy = agent_strategy
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.replacement_prob = replacement_prob
        self.discount = discount
        self.max_path_length = max_path_length
        self.qf_weight_decay = qf_weight_decay

        self.qf_update_method = \
            FirstOrderOptimizer(
                update_method=qf_update_method,
                learning_rate=qf_learning_rate,
            )
        self.qf_learning_rate = qf_learning_rate
        self.policy_weight_decay = policy_weight_decay


        self.policy_update_method = \
            FirstOrderOptimizer(
                update_method=policy_update_method,
                learning_rate=policy_learning_rate,
            )

        self.gating_func_update_method = \
            FirstOrderOptimizer(
                update_method='adam',
                learning_rate=policy_learning_rate,
            )

        # self.policy_func_update_method = \
        #     FirstOrderOptimizer(
        #         update_method='adam',
        #         learning_rate=policy_learning_rate,
        #     )

        self.policy_learning_rate = policy_learning_rate
        self.policy_updates_ratio = policy_updates_ratio
        self.eval_samples = eval_samples
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.plot = plot
        self.pause_for_plot = pause_for_plot

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self.scale_reward = scale_reward

        self.train_policy_itr = 0

        self.opt_info = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    @overrides
    def train(self):
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            ###only initialise the uninitialised ones
            # import pdb; pdb.set_trace()
            # sess.run(self.initialize_uninitialized(sess))

            self.initialize_uninitialized(sess)

            # This seems like a rather sequential method
            pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_dim=self.env.observation_space.flat_dim,
                action_dim=self.env.action_space.flat_dim,
                replacement_prob=self.replacement_prob,
            )

            agent_only_pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_dim=self.env.observation_space.flat_dim,
                action_dim=self.env.action_space.flat_dim,
                replacement_prob=self.replacement_prob,
            )

            oracle_only_pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_dim=self.env.observation_space.flat_dim,
                action_dim=self.env.action_space.flat_dim,
                replacement_prob=self.replacement_prob,
            )

            self.start_worker()
            self.init_opt()

            # This initializes the optimizer parameters
            self.initialize_uninitialized(sess)
            itr = 0
            path_length = 0
            path_return = 0
            terminal = False
            initial = False
            observation = self.env.reset()

            with tf.variable_scope("sample_policy"):
                sample_policy = Serializable.clone(self.policy)

            # with tf.variable_scope("oracle_policy"):
            #     oracle_policy = Serializable.clone(self.oracle_policy)

            oracle_policy = self.oracle_policy


            for epoch in range(self.n_epochs):
                logger.push_prefix('epoch #%d | ' % epoch)
                logger.log("Training started")
                train_qf_itr, train_policy_itr = 0, 0


                for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):
                    # Execute policy
                    if terminal:  # or path_length > self.max_path_length:
                        # Note that if the last time step ends an episode, the very
                        # last state and observation will be ignored and not added
                        # to the replay pool
                        observation = self.env.reset()
                        self.agent_strategy.reset()
                        sample_policy.reset()
                        self.es_path_returns.append(path_return)
                        path_length = 0
                        path_return = 0
                        initial = True
                    else:
                        initial = False

                    ### both continuous
                    agent_action, binary_action = self.agent_strategy.get_action_with_binary(itr, observation, policy=sample_policy)  # qf=qf)
                    sigma = np.round(binary_action)
                    # print(sigma)

                    ### getting actons from the oracle policy

                    oracle_action = self.get_oracle_action(itr, observation, policy=oracle_policy)


                    #take action based on either oracle action or agent action
                    action = sigma[0] * agent_action + sigma[1] * oracle_action

                    next_observation, reward, terminal, _ = self.env.step(action)
                    path_length += 1
                    path_return += reward


                    ##### adding the samples in Agent Buffer, Oracle Buffer, Buffer with both samples
                    if not terminal and path_length >= self.max_path_length:
                        terminal = True
                        # only include the terminal transition in this case if the flag was set
                        if self.include_horizon_terminal_transitions:
                            pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)

                    #### pool here - filled with both agent and oracle tuples - should be used for training the gating function
                    else:
                        pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)

                    observation = next_observation

                    if pool.size >= self.min_pool_size:
                    # if pool.size >= self.min_pool_size:
                        for update_itr in range(self.n_updates_per_sample):
                            # Train policy
                            batch = pool.random_batch(self.batch_size)
                            # oracle_batch = oracle_only_pool.random_batch(self.batch_size)
                            # agent_batch = agent_only_pool.random_batch(self.batch_size)
                            itrs = self.do_training(itr, batch)
                            train_qf_itr += itrs[0]
                            train_policy_itr += itrs[1]
                        sample_policy.set_param_values(self.policy.get_param_values())

                    itr += 1


                logger.log("Training finished")
                logger.log("Trained qf %d steps, policy %d steps"%(train_qf_itr, train_policy_itr))
                # logger.log("Pool sizes agent (%d) oracle (%d)" %(agent_only_pool.size, oracle_only_pool.size))


                if pool.size >= self.min_pool_size:
                    self.evaluate(epoch, pool)
                    params = self.get_epoch_snapshot(epoch)
                    logger.save_itr_params(epoch, params)
                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")
            self.env.terminate()
            self.policy.terminate()




    def init_opt(self, lambda_s = 100,
                 lambda_v = 10,
                 tau = .5):

        with tf.variable_scope("target_policy"):
            target_policy = Serializable.clone(self.policy)

        # with tf.variable_scope("oracle_policy"):
        #     oracle_policy = Serializable.clone(self.oracle_policy)

        oracle_policy = self.oracle_policy

        with tf.variable_scope("target_qf"):
            target_qf = Serializable.clone(self.qf)


        # y need to be computed first
        obs = self.obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )

        yvar = tensor_utils.new_tensor(
            'ys',
            ndim=1,
            dtype=tf.float32,
        )

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([tf.reduce_sum(tf.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = tf.reduce_mean(tf.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([tf.reduce_sum(tf.square(param))
                                        for param in self.policy.get_params(regularizable=True)])


        #
        # policy_qval = self.qf.get_qval_sym(
        #     obs, self.policy.get_action_sym(obs),
        #     deterministic=True
        # )
        #
        # policy_surr = -tf.reduce_mean(policy_qval)
        #
        # policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_input_list = [yvar, obs, action]
        policy_input_list = [obs]


        # y need to be computed first
        obs_oracle = self.env.observation_space.new_tensor_variable(
            'obs_oracle',
            extra_dims=1,
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action_oracle = self.env.action_space.new_tensor_variable(
            'action_oracle',
            extra_dims=1,
        )

        yvar_oracle = tensor_utils.new_tensor(
            'ys_oracle',
            ndim=1,
            dtype=tf.float32,
        )

        qval_oracle = self.qf.get_qval_sym(obs_oracle, action_oracle)
        qf_loss_oracle = tf.reduce_mean(tf.square(yvar_oracle - qval_oracle))
        qf_reg_loss_oracle = qf_loss_oracle + qf_weight_decay_term



        policy_qval_novice = self.qf.get_qval_sym(
            obs, self.policy.get_novice_policy_sym(obs),
            deterministic=True
        )

        gating_network = self.policy.get_action_binary_gate_sym(obs)

        policy_qval_oracle = self.qf.get_qval_sym(
            obs, self.policy.get_action_oracle_sym(obs),
            deterministic=True
        )


        # policy_surr_gate = -tf.reduce_mean(policy_qval_gate)

        combined_losses = tf.concat([tf.reshape(policy_qval_novice, [-1, 1]), tf.reshape(policy_qval_oracle, [-1, 1])], axis=1)

        combined_loss = -tf.reduce_mean(tf.reshape(tf.reduce_mean(combined_losses * gating_network, axis=1), [-1, 1]), axis=0)

        lambda_s_loss = tf.constant(0.0)

        if lambda_s > 0.0:
            lambda_s_loss = lambda_s * (tf.reduce_mean((tf.reduce_mean(gating_network, axis=0) - tau)**2) +
                                    tf.reduce_mean((tf.reduce_mean(gating_network, axis=1) - tau)**2))

        lambda_v_loss = tf.constant(0.0)

        if lambda_v > 0.0:
            mean0, var0 = tf.nn.moments(gating_network, axes=[0])
            mean, var1 = tf.nn.moments(gating_network, axes=[1])
            lambda_v_loss = - lambda_v * (tf.reduce_mean(var0) + tf.reduce_mean(var1))
        policy_surr = combined_loss

        policy_reg_surr = combined_loss + policy_weight_decay_term + lambda_s_loss + lambda_v_loss

        gf_input_list = [obs_oracle, action_oracle, yvar_oracle] + qf_input_list


        # self.gating_func_update_method.update_opt(gf_loss, target=target_policy.output_layer_binary, inputs=gf_input_list)

        # self.f_train_gf = tensor_utils.compile_function(
        #     inputs=gf_input_list,
        #     outputs=[gf_loss, self.gating_func_update_method._train_op],
        # )

        self.qf_update_method.update_opt(
            loss=qf_reg_loss, target=self.qf, inputs=qf_input_list)


        self.policy_update_method.update_opt(
            loss=policy_reg_surr, target=self.policy, inputs=policy_input_list)

        f_train_qf = tensor_utils.compile_function(
            inputs=qf_input_list,
            outputs=[qf_loss, qval, self.qf_update_method._train_op],
        )


        """
        For Training only the target policy component pi(s)
        """
        f_train_policy = tensor_utils.compile_function(
            inputs=policy_input_list,
            outputs=[policy_surr, self.policy_update_method._train_op, gating_network],
        )



        # f_train_policy_gate = tensor_utils.compile_function(
        #     inputs=policy_input_list,
        #     outputs=[policy_surr_gate, self.policy_update_method._train_op],
        # )

        self.opt_info = dict(
            f_train_qf=f_train_qf,
            f_train_policy=f_train_policy,
            target_qf=target_qf,
            target_policy=target_policy,
            oracle_policy=oracle_policy,
            # f_train_policy_gate=f_train_policy_gate
        )




    def do_training(self, itr, batch):

        ###extracting the whole batch here
        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        target_qf = self.opt_info["target_qf"]
        #target Policy
        target_policy = self.opt_info["target_policy"]
        #oracle Policy
        oracle_policy = self.opt_info["oracle_policy"]


        """
        TO DO : Should we also use the
        binary actions from beta(s) here?
        """

        """
        Using samples from both oracle and target policy here
        """
        next_actions, _ = target_policy.get_actions(next_obs)
        next_qvals = target_qf.get_qval(next_obs, next_actions)

        ys = rewards + (1. - terminals) * self.discount * next_qvals.reshape(-1)

        f_train_qf = self.opt_info["f_train_qf"]

        """
        Training the Q function with both the oracle and agent samples here
        """
        qf_loss, qval, _ = f_train_qf(ys, obs, actions)


        target_qf.set_param_values(
            target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
            self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

        self.train_policy_itr += self.policy_updates_ratio
        train_policy_itr = 0


        # ### extracting agent only batch here
        # obs_agent_only, actions_agent_only, rewards_agent_only, next_obs_agent_only, terminals_agent_only = ext.extract(
        #     agent_batch,
        #     "observations", "actions", "rewards", "next_observations",
        #     "terminals"
        # )
        #
        # ### extracting oracle only batch here
        # obs_oracle_only, actions_oracle_only, rewards_oracle_only, next_obs_oracle_only, terminals_oracle_only = ext.extract(
        #     oracle_batch,
        #     "observations", "actions", "rewards", "next_observations",
        #     "terminals"
        # )


        #### do we need these here?
        # next_actions_agent_only, _ = target_policy.get_actions(next_obs_agent_only)
        # next_qvals_agent_only = target_qf.get_qval(next_obs_agent_only, next_actions_agent_only)
        # ys_agent_only = rewards_agent_only + (1. - terminals_agent_only) * self.discount * next_qvals_agent_only.reshape(-1)


        ### computing targets based on oracle samples
        # next_actions_oracle_only, _ = oracle_policy.get_actions(next_obs_oracle_only)
        # next_qvals_oracle_only = target_qf.get_qval(next_obs_oracle_only, next_actions_oracle_only)
        # ys_oracle_only = rewards_oracle_only + (1. - terminals_oracle_only) * self.discount * next_qvals_oracle_only.reshape(-1)


        # gf_loss, gval, _ = self.f_train_gf(obs_oracle_only, actions_oracle_only, ys_oracle_only, ys_agent_only, obs_agent_only, actions_agent_only)



        """
        Training pi(s) using the agent samples only
        """
        while self.train_policy_itr > 0:
            f_train_policy = self.opt_info["f_train_policy"]
            ### agent samples only here
            policy_surr, _ , gating_outputs= f_train_policy(obs)

            target_policy.set_param_values(
                target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
                self.policy.get_param_values() * self.soft_target_tau)
            self.policy_surr_averages.append(policy_surr)
            # print("Gating outputs")
            # print(gating_outputs)
            self.train_policy_itr -= 1
            train_policy_itr += 1



        """
        Training beta(s) using both oracle and agent samples
        but we trained beta(s) with agent samples already
        so training beta(s) with the oracle samples only here
        (obs_oracle_only)
        """
        # while self.train_policy_itr > 0:
        #     f_train_policy_gate = self.opt_info["f_train_policy_gate"]
        #
        #     policy_surr, _ = f_train_policy_gate(obs_oracle_only)
        #
        #     target_policy.set_param_values(
        #         target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
        #         self.policy.get_param_values() * self.soft_target_tau)
        #     self.policy_surr_averages.append(policy_surr)
        #     self.train_policy_itr -= 1
        #     train_policy_itr += 1

        return 1, train_policy_itr # number of itrs qf, policy are trained



    #evaluation of the learnt policy
    def evaluate(self, epoch, pool):
        logger.log("Collecting samples for evaluation")
        paths = parallel_sampler.sample_paths(
            policy_params=self.policy.get_param_values(),
            max_samples=self.eval_samples,
            max_path_length=self.max_path_length,
        )

        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]
        # import pdb; pdb.set_trace()

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_q_loss = np.mean(self.qf_loss_averages)
        average_policy_surr = np.mean(self.policy_surr_averages)
        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

        policy_reg_param_norm = np.linalg.norm(
            self.policy.get_param_values(regularizable=True)
        )
        qfun_reg_param_norm = np.linalg.norm(
            self.qf.get_param_values(regularizable=True)
        )

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Iteration', epoch)
        logger.record_tabular('AverageReturn', np.mean(returns))
        logger.record_tabular('StdReturn',
                              np.std(returns))
        logger.record_tabular('MaxReturn',
                              np.max(returns))
        logger.record_tabular('MinReturn',
                              np.min(returns))
        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageQLoss', average_q_loss)
        logger.record_tabular('AveragePolicySurr', average_policy_surr)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))
        logger.record_tabular('AverageAction', average_action)

        logger.record_tabular('PolicyRegParamNorm',
                              policy_reg_param_norm)
        logger.record_tabular('QFunRegParamNorm',
                              qfun_reg_param_norm)

        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)

        self.qf_loss_averages = []
        self.policy_surr_averages = []

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            policy=self.policy,
            target_qf=self.opt_info["target_qf"],
            target_policy=self.opt_info["target_policy"],
            es=self.agent_strategy,
        )


    def get_oracle_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        # ou_state = self.evolve_state()
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


    ### to reinitialise variables in TF graph
    ### which has not been initailised so far
    def initialize_uninitialized(self, sess):
        global_vars          = tf.global_variables()
        is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        print([str(i.name) for i in not_initialized_vars]) # only for testing
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))
