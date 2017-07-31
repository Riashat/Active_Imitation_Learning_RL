import numpy as np
import pickle as pickle
from sandbox.rocky.tf.misc import tensor_utils


class VecEnvExecutor(object):
    def __init__(self, envs, max_path_length):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

    #def step(self, action_n, itr, env_action_space):
    def step(self, action_n, itr):

        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]

        obs, rewards, dones, env_infos = list(map(list, list(zip(*all_results))))


        # if env_action_space == 5:
        #     ###function to modify the goal position
        #     rewards, dones = self.change_goal_state(itr, obs, rewards, dones)

        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)


    def change_goal_state(self, itr, obs, rewards, dones):

        #if itr <= itr/2:
        if itr <= 50:
            rewards = rewards
            dones = dones

        #elif itr > itr/2:
        elif itr > 50 <= 150:  

            #adding new goal position - new goal position is state = 50
            if obs[0] == 50:
                rewards = [1, rewards[1]]
                dones = [True, dones[1]]
            elif obs[1] == 50:
                rewards = [rewards[0], 1]
                dones = [dones[0], True]

            elif obs[0] == 50 & obs[1] == 50:
                rewards = [1, 1]
                dones = [True, True]

            #remove previous goal position
            elif obs[0] == 63:
                rewards = [0, rewards[0]]
                dones = [False, False]

            elif obs[1] == 63:
                rewards = [rewards[0], 0]
                dones = [False, False]

            elif obs[0] == 63 & obs[1] == 63:
                rewards = [0, 0]
                dones = [False, False]

            else:
                obs = obs
                rewards = rewards
                dones = dones

        elif itr > 150:  

            #adding new goal position - new goal position is state = 50
            if obs[0] == 25:
                rewards = [1, rewards[1]]
                dones = [True, dones[1]]
            elif obs[1] == 50:
                rewards = [rewards[0], 1]
                dones = [dones[0], True]

            elif obs[0] == 25 & obs[1] == 25:
                rewards = [1, 1]
                dones = [True, True]

            #remove previous goal position
            elif obs[0] == 63:
                rewards = [0, rewards[0]]
                dones = [False, False]

            elif obs[1] == 63:
                rewards = [rewards[0], 0]
                dones = [False, False]

            elif obs[0] == 63 & obs[1] == 63:
                rewards = [0, 0]
                dones = [False, False]

            else:
                obs = obs
                rewards = rewards
                dones = dones

        return rewards, dones




    def reset(self):
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass
