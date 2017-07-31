import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb

eps = range(2991)
hopper_variance_divide = np.sqrt(5)



"""
Gating with Q-Learning
"""

oracle_data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/DDPG/Results/rllab_results/Active_RL/Hard_Oracle_DDPG/progress.csv")
oracle = np.array(oracle_data["AverageReturn"])

agent_data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/DDPG/Results/rllab_results/Active_RL/Hard_Agent_DDPG/progress.csv")
agent = np.array(agent_data["AverageReturn"])


"""
Pseudo-Gate
"""

pseudo_oracle_data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/DDPG/Results/rllab_results/Active_RL/Pseudo_Oracle_DDPG/progress.csv")
pseudo_oracle = np.array(pseudo_oracle_data["AverageReturn"])

pseudo_agent_data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/DDPG/Results/rllab_results/Active_RL/Pseudo_Agent_DDPG/progress.csv")
pseudo_agent = np.array(pseudo_agent_data["AverageReturn"])


"""
Gating with Q-Learning - version 2
"""

oracle_q_data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/DDPG/Results/rllab_results/Active_RL/Hard_Q_Oracle_DDPG/progress.csv")
oracle_q = np.array(oracle_q_data["AverageReturn"])

agent_q_data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/DDPG/Results/rllab_results/Active_RL/Hard_Q_Agent_DDPG/progress.csv")
agent_q = np.array(agent_q_data["AverageReturn"])




def double_plot(stats1, stats2, smoothing_window=100, noshow=False):
    ## Figure 1
    fig = plt.figure(figsize=(40, 5))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "red", linewidth=2.5, label="Oracle")    
    # plt.fill_between( eps, rewards_smoothed_1 + poly_walker_std_return,   rewards_smoothed_1 - poly_walker_std_return, alpha=0.2, edgecolor='red', facecolor='red')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "blue", linewidth=2.5, label="Agent" )  
    # plt.fill_between( eps, rewards_smoothed_2 + ddpg_walker_std_return,   rewards_smoothed_2 - ddpg_walker_std_return, alpha=0.2, edgecolor='blue', facecolor='blue')

    plt.legend(handles=[cum_rwd_1, cum_rwd_2])
    plt.xlabel("Epsiode")
    plt.ylabel("Average Return")
    plt.title("Hopper Environment - Hard Gate Q Learning (v2)")
  
    plt.show()
    
    return fig





# def comparison_plot_first_two(stats1, stats2, stats3, stats4, smoothing_window=20, noshow=False):

# 	## Figure 1
#     fig = plt.figure(figsize=(10,1))
#     plt.subplot(121)
#     rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()

#     cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "red", linewidth=2.5, label="PolyRL + DDPG")    
#     plt.fill_between( eps, rewards_smoothed_1 + poly_hopper_std_return,   rewards_smoothed_1 - poly_hopper_std_return, alpha=0.2, edgecolor='red', facecolor='red')

#     cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "blue", linewidth=2.5, label="DDPG")    
#     plt.fill_between( eps, rewards_smoothed_2 + ddpg_hopper_std_return,   rewards_smoothed_2 - ddpg_hopper_std_return, alpha=0.2, edgecolor='blue', facecolor='blue')

#     plt.legend(handles=[cum_rwd_1, cum_rwd_2])
#     plt.xlabel("Number of Episodes")
#     plt.ylabel("Average Return")
#     plt.title("Hopper Environment")
  

#     ## Figure 2
#     plt.subplot(122)
#     rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()

#     cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "red", linewidth=2.5,  label="PolyRL + DDPG")    
#     plt.fill_between( eps, rewards_smoothed_3 + poly_walker_std_return,   rewards_smoothed_3 - poly_walker_std_return, alpha=0.2, edgecolor='red', facecolor='red')

#     cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, color = "blue", linewidth=2.5,  label="DDPG")    
#     plt.fill_between( eps, rewards_smoothed_4 + ddpg_walker_std_return,   rewards_smoothed_4 - ddpg_walker_std_return, alpha=0.2, edgecolor='blue', facecolor='blue')

#     plt.legend(handles=[cum_rwd_3, cum_rwd_4])
#     plt.xlabel("Number of Episodes")
#     plt.ylabel("Average Return")
#     plt.title("Walker Environment")



#     plt.show()
    
#     return fig




def main():

   double_plot(oracle_q, agent_q)
   # comparison_plot_first_two(poly_hopper_avg_rwd, ddpg_hopper_avg_rwd, poly_walker_avg_rwd, ddpg_walker_avg_rwd)


if __name__ == '__main__':
    main()