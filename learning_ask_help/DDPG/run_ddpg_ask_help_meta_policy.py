# from ddpg_tensorflow.ddpg import DDPG
from learning_active_learning.learning_ask_help.DDPG.oracle_ddpg import DDPG as Oracle_DDPG
from learning_active_learning.learning_ask_help.DDPG.agent_ddpg_meta_policy import DDPG as Agent_DDPG

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy

#if using the categorial policy to get action probabilities


from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.misc import ext
import pickle
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("oracle", help="Type of DDPG to run: oracle or agent")
parser.add_argument("agent", help="Type of DDPG to run: oracle or agent")
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--plot", action="store_true")
# parser.add_argument("--data_dir", default="./data/")
args = parser.parse_args()

stub(globals())
ext.set_seed(1)

supported_gym_envs = ["MountainCarContinuous-v0", "Hopper-v1", "Walker2d-v1", "Humanoid-v1", "Reacher-v1", "HalfCheetah-v1", "Swimmer-v1", "HumanoidStandup-v1"]

other_env_class_map  = { "Cartpole" :  CartpoleEnv}

if args.env in supported_gym_envs:
    gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=False)
    # gymenv.env.seed(1)
else:
    gymenv = other_env_class_map[args.env]()

#TODO: assert continuous space


env = TfEnv(normalize(gymenv))


es = OUStrategy(env_spec=env.spec)


### agent policy
policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    name="policy",
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
)


### oracle policy
oracle_policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    name="oracle_policy",
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
)



### agent critic
qf = ContinuousMLPQFunction(env_spec=env.spec,
                            hidden_sizes=(100,100),
                            hidden_nonlinearity=tf.nn.relu,)


### oracle critic
oracle_qf = ContinuousMLPQFunction(env_spec=env.spec,
                            hidden_sizes=(100,100),
                            hidden_nonlinearity=tf.nn.relu,)




ddpg_type = {"oracle" : Oracle_DDPG, "agent" : Agent_DDPG }


oracle_ddpg_class = ddpg_type[args.oracle]
agent_ddpg_class = ddpg_type[args.agent]


## loops:
num_experiments = 1
batch_size_values = [64]



print ("Output from MLP", env.action_space.flat_dim)
print ("Input to MLP", env.observation_space.flat_dim)

"""
Debugging tool
"""
# import pdb; pdb.set_trace()
#to quit from it : press q





for b in range(len(batch_size_values)): 
    
    for e in range(num_experiments):


        """
        Training the oracle policy
        """

        oracle_algo = oracle_ddpg_class(
            env=env,
            policy=oracle_policy,
            es=es,
            qf=oracle_qf,
            batch_size=batch_size_values[b],
            max_path_length=env.horizon,
            epoch_length=1000,
            min_pool_size=10000,
            n_epochs=args.num_epochs,
            discount=0.99,
            scale_reward=1.0,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            plot=args.plot,
        )


        run_experiment_lite(
            oracle_algo.train(),
            # log_dir=args.data_dir,
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            exp_name="Meta-Controller/" + "Oracle_DDPG/",
            seed=1,
            plot=args.plot,
        )

        
        """
        Agent policy
        """

        algo = agent_ddpg_class(
            env=env,
            policy=policy,
            oracle_policy=oracle_policy, 
            es=es,
            qf=qf,
            batch_size=batch_size_values[b],
            max_path_length=env.horizon,
            epoch_length=1000,
            min_pool_size=10000,
            n_epochs=args.num_epochs,
            discount=0.99,
            scale_reward=1.0,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            plot=args.plot,
        )


        run_experiment_lite(
            algo.train(),
            # log_dir=args.data_dir,
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            exp_name="Meta-Controller/" + "DDPG/",
            seed=1,
            plot=args.plot,
        )




