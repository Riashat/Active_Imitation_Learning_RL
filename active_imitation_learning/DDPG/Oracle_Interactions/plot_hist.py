import matplotlib.pyplot as plt
import numpy as np
import pdb

oracle = np.load('/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/DDPG/Oracle_Interactions/oracle_interactons.npy')
agent = np.load('/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/DDPG/Oracle_Interactions/agent_interactions.npy')

ind = np.arange(len(oracle))
width = 0.35

fig = plt.figure(figsize=(40, 5))
p1 = plt.bar(ind, oracle, width, color='#d62728')
p2 = plt.bar(ind, agent, width)
plt.xlabel('Number of Episodes')
plt.ylabel('Proportion of Interactions with Oracle')
plt.title('Total number of Oracle Action vs Agent Action per episode')
plt.legend((p1[0], p2[0]), ('Oracle', 'Agent'))
plt.show()


fig = plt.figure(figsize=(40, 5))
plt.bar(ind, oracle)
plt.xlabel('Number of Episodes')
plt.ylabel('Interactions with Oracle')
plt.title('Number of Interactions with Oracle per episode')
plt.show()


