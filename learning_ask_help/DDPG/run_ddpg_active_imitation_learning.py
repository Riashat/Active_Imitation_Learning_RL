from agent_ddpg_active_rl import DDPG as Agent_DDPG
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy

from hierarchical_deterministic_mlp_policy import LayeredDeterministicMLPPolicy
from agent_action_selection import AgentStrategy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.misc import ext
import pickle
import tensorflow as tf
import argparse
from expert_policies_ddpg.load_policy import load_policy
import expert_policies_ddpg.tf_util


parser = argparse.ArgumentParser()
parser.add_argument("agent", help="Type of DDPG to run: oracle or agent")
parser.add_argument('expert_policy_file', type=str)
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--data_dir", default="./data/")
args = parser.parse_args()

stub(globals())
ext.set_seed(1)


supported_gym_envs = ["MountainCarContinuous-v0", "Hopper-v1", "Walker2d-v1", "Humanoid-v1", "Reacher-v1", "HalfCheetah-v1", "Swimmer-v1", "HumanoidStandup-v1"]
# other_env_class_map  = { "Cartpole" :  CartpoleEnv}

# if args.env in supported_gym_envs:
#     gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=False)
#     # gymenv.env.seed(1)
# else:
#     gymenv = other_env_class_map[args.env]()



gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=False)

env = TfEnv(normalize(gymenv))
es = OUStrategy(env_spec=env.spec)
agent_strategy = AgentStrategy(env_spec=env.spec)


## agent policy
policy = LayeredDeterministicMLPPolicy(
    env_spec=env.spec,
    name="policy",
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
)


### agent critic
qf = ContinuousMLPQFunction(env_spec=env.spec,
                            hidden_sizes=(100,100),
                            hidden_nonlinearity=tf.nn.relu,)

ddpg_type = {"agent" : Agent_DDPG }


agent_ddpg_class = ddpg_type[args.agent]

## loops:
num_experiments = 1
batch_size_values = [64]


print('loading and building expert policy')
oracle_policy = load_policy(args.expert_policy_file)
print('loaded and built')


##use a trained oracle policy here


for b in range(len(batch_size_values)): 
    
    for e in range(num_experiments):
        
        """
        Agent policy
        """

        algo = agent_ddpg_class(
            env=env,
            policy=policy,
            oracle_policy=oracle_policy, 
            agent_strategy=agent_strategy,
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
            exp_name="Active_RL/" + "Agent_DDPG/",
            seed=1,
            plot=args.plot,
        )




