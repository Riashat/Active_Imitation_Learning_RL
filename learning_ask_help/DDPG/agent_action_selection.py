from rllab.misc.overrides import overrides
from rllab.misc.ext import AttrDict
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np
import numpy.random as nr


class AgentStrategy(ExplorationStrategy, Serializable):
    """
    This strategy builds up from the OUStrategy class
    This class decides which action to take - continuous or discrete
    based on the policy pi(s)
    """

    def __init__(self, env_spec, mu=0, theta=0.15, sigma=0.3, **kwargs):
        assert isinstance(env_spec.action_space, Box)
        assert len(env_spec.action_space.shape) == 1
        Serializable.quick_init(self, locals())
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = env_spec.action_space
        self.state = np.ones(self.action_space.flat_dim) * self.mu
        self.reset()

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["state"] = self.state
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.state = d["state"]

    @overrides
    def reset(self):
        self.state = np.ones(self.action_space.flat_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    # @overrides
    # def get_action(self, t, observation, policy, **kwargs):
    #     #action here is from the policy MLP function
    #     action, _ = policy.get_action(observation)
    #     ou_state = self.evolve_state()
    #     return np.clip(action + ou_state, self.action_space.low, self.action_space.high)


    @overrides
    def get_action(self, t, observation, policy, **kwargs):
        #action here is from the policy MLP function
        action, binary_action, _ = policy.get_action(observation)
        #action, binary_action, _ = policy.get_actions(observation)

        ou_state = self.evolve_state()

        continuous_action = np.clip(action + ou_state, self.action_space.low, self.action_space.high)

        return continuous_action, binary_action




if __name__ == "__main__":
    ou = AgentStrategy(env_spec=AttrDict(action_space=Box(low=-1, high=1, shape=(1,))), mu=0, theta=0.15, sigma=0.3)
    states = []
    for i in range(1000):
        states.append(ou.evolve_state()[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
