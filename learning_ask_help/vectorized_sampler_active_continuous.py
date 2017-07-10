import pickle

import tensorflow as tf
from rllab.sampler.base import BaseSampler
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools
from sandbox.rocky.tf.envs.vec_env_executor_active import VecEnvExecutor


class VectorizedSampler(BaseSampler):

    def __init__(self, algo, n_envs=None):
        super(VectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length
            )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()


    """
    this returns PATHS - same as Trajectories?
    These samples are used for estimating the gradient
    """
    # def obtain_samples(self, itr, oracle_policy, env_action_space):
    def obtain_samples(self, itr, oracle_policy):
        logger.log("Obtaining samples for iteration %d..." % itr)
        paths = []
        agent_only_paths = []
        oracle_only_paths = []
        n_samples = 0
        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs
        agent_only_running_paths = [None] * self.vec_env.num_envs
        oracle_only_running_paths = [None] * self.vec_env.num_envs


        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        import time

        #batch size parameter for TRPO - 
        #determines the number of samples to collect?
        while n_samples < self.algo.batch_size:

            t = time.time()
            policy.reset(dones)

            #modifly POLICY.GET_ACTIONS HERE
            agent_actions, binary_actions, agent_infos = policy.get_actions(obses)

            sigma = np.round(binary_actions)

            oracle_actions, oracle_agent_infos = oracle_policy.get_actions(obses)

            #take action based on either oracle action or agent action
            actions = sigma * agent_actions + (1 - sigma) * oracle_actions

            policy_time += time.time() - t
            t = time.time()

            # next_obses, rewards, dones, env_infos = self.vec_env.step(actions, itr, env_action_space)
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions, itr)


            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]


            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                if done:
                    paths.append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None


            if sigma[0] == 1. or sigma[1] == 1.:

                for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                        rewards, env_infos, agent_infos,
                                                                                        dones):
                    if agent_only_running_paths[idx] is None:
                        agent_only_running_paths[idx] = dict(
                            observations=[],
                            actions=[],
                            rewards=[],
                            env_infos=[],
                            agent_infos=[],
                        )
                    agent_only_running_paths[idx]["observations"].append(observation)
                    agent_only_running_paths[idx]["actions"].append(action)
                    agent_only_running_paths[idx]["rewards"].append(reward)
                    agent_only_running_paths[idx]["env_infos"].append(env_info)
                    agent_only_running_paths[idx]["agent_infos"].append(agent_info)

                    if done:
                        agent_only_paths.append(dict(
                            observations=self.env_spec.observation_space.flatten_n(agent_only_running_paths[idx]["observations"]),
                            actions=self.env_spec.action_space.flatten_n(agent_only_running_paths[idx]["actions"]),
                            rewards=tensor_utils.stack_tensor_list(agent_only_running_paths[idx]["rewards"]),
                            env_infos=tensor_utils.stack_tensor_dict_list(agent_only_running_paths[idx]["env_infos"]),
                            agent_infos=tensor_utils.stack_tensor_dict_list(agent_only_running_paths[idx]["agent_infos"]),
                        ))
                        n_samples += len(agent_only_running_paths[idx]["rewards"])
                        agent_only_running_paths[idx] = None


            elif sigma[0] == 0. or sigma[1] == 0.:

                for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                        rewards, env_infos, agent_infos,
                                                                                        dones):
                    if oracle_only_running_paths[idx] is None:
                        oracle_only_running_paths[idx] = dict(
                            observations=[],
                            actions=[],
                            rewards=[],
                            env_infos=[],
                            agent_infos=[],
                        )
                    oracle_only_running_paths[idx]["observations"].append(observation)
                    oracle_only_running_paths[idx]["actions"].append(action)
                    oracle_only_running_paths[idx]["rewards"].append(reward)
                    oracle_only_running_paths[idx]["env_infos"].append(env_info)
                    oracle_only_running_paths[idx]["agent_infos"].append(agent_info)

                    if done:
                        oracle_only_paths.append(dict(
                            observations=self.env_spec.observation_space.flatten_n(oracle_only_running_paths[idx]["observations"]),
                            actions=self.env_spec.action_space.flatten_n(oracle_only_running_paths[idx]["actions"]),
                            rewards=tensor_utils.stack_tensor_list(oracle_only_running_paths[idx]["rewards"]),
                            env_infos=tensor_utils.stack_tensor_dict_list(oracle_only_running_paths[idx]["env_infos"]),
                            agent_infos=tensor_utils.stack_tensor_dict_list(oracle_only_running_paths[idx]["agent_infos"]),
                        ))
                        n_samples += len(oracle_only_running_paths[idx]["rewards"])
                        oracle_only_running_paths[idx] = None


            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return paths, agent_only_paths, oracle_only_paths


    # def oracle_interaction(self, obses, actions, oracle_policy, env_action_space):

    #     np_actions = np.asarray(actions)

    #     if env_action_space == 3:

    #         if np_actions[0] == 2 & np_actions[1] == 2:
    #             queried_oracle = True
    #             oracle_actions, oracle_agent_infos = oracle_policy.get_actions(obses)
    #             # oracle_chosen_action_0 = oracle_action(obses, 0)
    #             # oracle_chosen_action_1 = oracle_action(obses, 1)
    #             # actions = [oracle_chosen_action_0, oracle_chosen_action_1]
    #             actions = oracle_actions
    #             cost = [0, 0]


    #         elif np_actions[0] == 2:
    #             queried_oracle = True
    #             oracle_chosen_action, oracle_agent_infos = oracle_policy.get_actions(obses)

    #             actions = [oracle_chosen_action[0], np_actions[1]]
    #             cost = [0, 0]


    #         elif np_actions[1] == 2:
    #             queried_oracle = True
    #             oracle_chosen_action, oracle_agent_infos = oracle_policy.get_actions(obses)
    #             actions = [np_actions[0], oracle_chosen_action[1]]
    #             cost = [0, 0]   
    #             # oracle_chosen_action = oracle_action(obses, 1)


    #         else:
    #             queried_oracle = False
    #             actions = [np_actions[0], np_actions[1]]

    #     elif env_action_space == 4:

    #         if np_actions[0] == 3 & np_actions[1] == 3:
    #             queried_oracle = True
    #             oracle_actions, oracle_agent_infos = oracle_policy.get_actions(obses)
    #             # oracle_chosen_action_0 = oracle_action(obses, 0)
    #             # oracle_chosen_action_1 = oracle_action(obses, 1)
    #             # actions = [oracle_chosen_action_0, oracle_chosen_action_1]
    #             actions = oracle_actions
    #             cost = [0, 0]


    #         elif np_actions[0] == 3:
    #             queried_oracle = True
    #             oracle_chosen_action, oracle_agent_infos = oracle_policy.get_actions(obses)

    #             actions = [oracle_chosen_action[0], np_actions[1]]
    #             cost = [0, 0]


    #         elif np_actions[1] == 3:
    #             queried_oracle = True
    #             oracle_chosen_action, oracle_agent_infos = oracle_policy.get_actions(obses)
    #             actions = [np_actions[0], oracle_chosen_action[1]]
    #             cost = [0, 0]   
    #             # oracle_chosen_action = oracle_action(obses, 1)

    #         else:
    #             queried_oracle = False
    #             actions = [np_actions[0], np_actions[1]]


    #     elif env_action_space == 5:

    #         if np_actions[0] == 4 & np_actions[1] == 4:
    #             queried_oracle = True
    #             oracle_actions, oracle_agent_infos = oracle_policy.get_actions(obses)
    #             actions = oracle_actions
    #             cost = [-10, -10]


    #         elif np_actions[0] == 4:
    #             queried_oracle = True
    #             oracle_chosen_action, oracle_agent_infos = oracle_policy.get_actions(obses)
    #             actions = [oracle_chosen_action[0], np_actions[1]]
    #             cost = [-10, -1]

    #         elif np_actions[1] == 4:
    #             queried_oracle = True
    #             oracle_chosen_action, oracle_agent_infos = oracle_policy.get_actions(obses)
    #             actions = [np_actions[0], oracle_chosen_action[1]]
    #             cost = [-1, -10]

    #         else:
    #             queried_oracle = False
    #             actions = [np_actions[0], np_actions[1]]
    #             cost = [-1, -1]


    #     return actions, cost
