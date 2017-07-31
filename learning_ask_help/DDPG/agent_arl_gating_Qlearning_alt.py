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
    def __init__(
            self,
            env,
            policy,
            oracle_policy,
            qf,
            gate_qf,
            agent_strategy,
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size = 10000,
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
        self.discrete_qf = gate_qf
        self.gate_qf = gate_qf
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

        self.gate_qf_update_method = \
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

        self.policy_gate_update_method = \
            FirstOrderOptimizer(
                update_method=policy_update_method,
                learning_rate=policy_learning_rate,
            )


        self.gating_func_update_method = \
            FirstOrderOptimizer(
                update_method='adam',
                learning_rate=policy_learning_rate,
            )

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
        self.train_gate_policy_itr = 0

        self.opt_info = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    @overrides
    def train(self):
        with tf.Session() as sess:

            self.initialize_uninitialized(sess)

            # This seems like a rather sequential method
            pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_dim=self.env.observation_space.flat_dim,
                action_dim=self.env.action_space.flat_dim,
                replacement_prob=self.replacement_prob,
            )

            binary_pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_dim=self.env.observation_space.flat_dim,
                action_dim=2,
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

            with tf.variable_scope("sample_target_gate_qf"):
                target_gate_qf = Serializable.clone(self.gate_qf)


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


                    agent_action, _ = self.agent_strategy.get_action_with_binary(itr, observation, policy=sample_policy)  # qf=qf)
                    binary_action, _ = self.discrete_qf.get_action(observation)
                    oracle_action = self.get_oracle_action(itr, observation, policy=oracle_policy)


                    action = binary_action * agent_action + (1.0 - binary_action) * oracle_action
                    
                    next_observation, reward, terminal, _ = self.env.step(action)
                    path_length += 1
                    path_return += reward


                    if not terminal and path_length >= self.max_path_length:
                        terminal = True
                        if self.include_horizon_terminal_transitions:
                            pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)
                            binary_pool.add_sample(observation, binary_action, reward * self.scale_reward, terminal, initial)

                    else:
                        pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)
                        binary_pool.add_sample(observation, binary_action, reward * self.scale_reward, terminal, initial)

                    observation = next_observation

                    if pool.size >= self.min_pool_size:
                        for update_itr in range(self.n_updates_per_sample):
                            # Train policy
                            # batches from pool containing continuous actions and discrete actions
                            batch = pool.random_batch(self.batch_size)
                            binary_batch = binary_pool.random_batch(self.batch_size)

                            itrs = self.do_training(itr, batch, binary_batch)
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





    def init_opt(self):

        with tf.variable_scope("target_policy"):
            target_policy = Serializable.clone(self.policy)

        oracle_policy = self.oracle_policy

        with tf.variable_scope("target_qf"):
            target_qf = Serializable.clone(self.qf)

        with tf.variable_scope("target_gate_qf"):
            target_gate_qf = Serializable.clone(self.gate_qf)


        obs = self.obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )

        discrete_action = tensor_utils.new_tensor(
            'discrete_action',
            ndim=2,
            dtype=tf.float32,
        )

        yvar = tensor_utils.new_tensor(
            'ys',
            ndim=1,
            dtype=tf.float32,
        )

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([tf.reduce_sum(tf.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([tf.reduce_sum(tf.square(param))
                                        for param in self.policy.get_params(regularizable=True)])


        policy_qval_novice = self.qf.get_qval_sym(
            obs, self.policy.get_novice_policy_sym(obs),
            deterministic=True
        )

        policy_qval_gate = self.discrete_qf.get_qval_sym(
           obs, self.policy.get_action_binary_gate_sym(obs),
           deterministic=True
        )


        qval = self.qf.get_qval_sym(obs, action)
        qf_loss = tf.reduce_mean(tf.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term


        discrete_qval = self.gate_qf.get_qval_sym(obs, discrete_action)
        discrete_qf_loss = tf.reduce_mean(tf.square(yvar - discrete_qval))
        discrete_qf_reg_loss = discrete_qf_loss + qf_weight_decay_term


        qf_input_list = [yvar, obs, action]
        discrete_qf_input_list = [yvar, obs, discrete_action]

        policy_input_list = [obs]
        policy_gate_input_list = [obs]



        gating_network = self.policy.get_action_binary_gate_sym(obs)


        policy_surr = -tf.reduce_mean(policy_qval_novice)
        policy_reg_surr = policy_surr + policy_weight_decay_term

        policy_gate_surr = - tf.reduce_mean(policy_qval_gate) + policy_weight_decay_term
        policy_reg_gate_surr = policy_gate_surr + policy_weight_decay_term



        self.qf_update_method.update_opt(
            loss=qf_reg_loss, target=self.qf, inputs=qf_input_list)

        self.gate_qf_update_method.update_opt(
            loss=discrete_qf_reg_loss, target=self.gate_qf, inputs=discrete_qf_input_list)

        self.policy_update_method.update_opt(
            loss=policy_reg_surr, target=self.policy, inputs=policy_input_list)

        self.policy_gate_update_method.update_opt(
            loss=policy_reg_gate_surr, target=self.policy, inputs=policy_gate_input_list)



        f_train_qf = tensor_utils.compile_function(
            inputs=qf_input_list,
            outputs=[qf_loss, qval, self.qf_update_method._train_op],
        )


        f_train_discrete_qf = tensor_utils.compile_function(
            inputs=discrete_qf_input_list,
            outputs=[discrete_qf_loss, discrete_qval, self.gate_qf_update_method._train_op],
        )

        f_train_policy = tensor_utils.compile_function(
           inputs=policy_input_list,
           outputs=[policy_surr, self.policy_update_method._train_op],
        )

        f_train_policy_gate = tensor_utils.compile_function(
           inputs=policy_gate_input_list,
           outputs=[policy_gate_surr, self.policy_gate_update_method._train_op, gating_network],
        )




        self.opt_info = dict(
            f_train_qf=f_train_qf,
            f_train_discrete_qf=f_train_discrete_qf,
            f_train_policy=f_train_policy,
            f_train_policy_gate=f_train_policy_gate,
            target_qf=target_qf,
            target_gate_qf=target_gate_qf,
            target_policy=target_policy,
            oracle_policy=oracle_policy,
        )




    def do_training(self, itr, batch, binary_batch):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        binary_obs, binary_actions, binary_rewards, binary_next_obs, binary_terminals = ext.extract(
            binary_batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )


        target_qf = self.opt_info["target_qf"]
        target_gate_qf = self.opt_info["target_gate_qf"]
        target_policy = self.opt_info["target_policy"]

        ## training critic for pi(s)
        next_actions, _ = target_policy.get_actions(next_obs)
        next_qvals = target_qf.get_qval(next_obs, next_actions)
        ys = rewards + (1. - terminals) * self.discount * next_qvals.reshape(-1)

        ## training the critic
        f_train_qf = self.opt_info["f_train_qf"]
        qf_loss, qval, _ = f_train_qf(ys, obs, actions)
        target_qf.set_param_values(
            target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
            self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys)


        ## for training the actor for pi(s)
        self.train_policy_itr += self.policy_updates_ratio
        train_policy_itr = 0

        while self.train_policy_itr > 0:
            f_train_policy = self.opt_info["f_train_policy"]
            policy_surr, _ = f_train_policy(obs)
            target_policy.set_param_values(
                target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
                self.policy.get_param_values() * self.soft_target_tau)
            self.policy_surr_averages.append(policy_surr)
            self.train_policy_itr -= 1
            train_policy_itr += 1



        """
        Training the gate function with Q-learning here
        """
        ## next_binary_actions - probabilities of the binary actions
        next_binary_actions, _ = target_policy.get_binary_actions(next_obs)

        next_max_qvals = target_gate_qf.get_max_qval(next_obs)
        ys_discrete_qf = binary_rewards + (1. - terminals) * self.discount * next_max_qvals.reshape(-1)

        f_train_discrete_qf = self.opt_info["f_train_discrete_qf"]
        qf_loss, qval, _ = f_train_discrete_qf(ys_discrete_qf, binary_obs, binary_actions)


        ## for training the actor with Q-learning critic
        self.train_gate_policy_itr += self.policy_updates_ratio
        train_gate_policy_itr = 0

        while self.train_gate_policy_itr > 0:
            f_train_policy_gate = self.opt_info["f_train_policy_gate"]
            policy_surr, _ , gating_outputs = f_train_policy_gate(obs)
            # target_policy.set_param_values(
            #     target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
            #     self.policy.get_param_values() * self.soft_target_tau)
            self.policy_surr_averages.append(policy_surr)
            self.train_gate_policy_itr -= 1
            train_gate_policy_itr += 1


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
