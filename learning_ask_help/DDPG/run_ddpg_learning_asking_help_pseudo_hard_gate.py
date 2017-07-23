from oracle_ddpg import DDPG as Oracle_DDPG
from agent_ddpg_active_rl_gating_ddpg import DDPG as Agent_DDPG


from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from hierarchical_deterministic_mlp_policy import LayeredDeterministicMLPPolicy


from agent_action_selection import AgentStrategy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from continuous_mlp_q_function import ContinuousMLPQFunction

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.misc import ext
import pickle
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("oracle", help="Type of DDPG to run: oracle or agent")
# parser.add_argument("agent", help="Type of DDPG to run: oracle or agent")
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--plot", action="store_true")
# parser.add_argument("--data_dir", default="./data/")
args = parser.parse_args()

stub(globals())
ext.set_seed(1)


supported_gym_envs = ["MountainCarContinuous-v0", "Hopper-v1", "Walker2d-v1", "Humanoid-v1", "Reacher-v1", "HalfCheetah-v1", "Swimmer-v1", "HumanoidStandup-v1"]
gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=False)

env = TfEnv(normalize(gymenv))
es = OUStrategy(env_spec=env.spec)
agent_strategy = AgentStrategy(env_spec=env.spec)

### oracle policy
oracle_policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    name="oracle_policy",
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
)


### agent policy
policy = LayeredDeterministicMLPPolicy(
    env_spec=env.spec,
    name="policy",

    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
    oracle_policy=oracle_policy,
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

# oracle_ddpg_class = ddpg_type[args.oracle]
# agent_ddpg_class = ddpg_type[args.agent]


oracle_ddpg_class = ddpg_type["oracle"]
agent_ddpg_class = ddpg_type["agent"]



num_experiments = 1

for e in range(num_experiments):
    """
    Training the oracle policy
    """
    oracle_algo = oracle_ddpg_class(
        env=env,
        policy=oracle_policy,
        es=es,
        qf=oracle_qf,
        batch_size=64,
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
        exp_name="Active_RL/" + "Oracle_DDPG/",
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
        agent_strategy=agent_strategy,
        qf=qf,
        batch_size=64,
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
        exp_name="Active_RL/" + "Agent_DDPG/",
        seed=1,
        plot=args.plot,
    )
