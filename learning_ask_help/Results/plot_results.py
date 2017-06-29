import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt

eps = range(300)



# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_cart.csv")
# trpo_cart_avg_rwd = np.array(data["AverageReturn"])
# trpo_cart_avg_rwd = trpo_cart_avg_rwd[0:500]

# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_cart_active.csv")
# trpo_cart_active_avg_rwd = np.array(data["AverageReturn"])

# print ("trpo_cart_avg_rwd", trpo_cart_avg_rwd.shape)
# print ("trpo_cart_active_avg_rwd", trpo_cart_active_avg_rwd.shape)



# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_mountain.csv")
# trpo_mountain_avg_rwd = np.array(data["AverageReturn"])
# trpo_mountain_avg_rwd = trpo_mountain_avg_rwd[0:500]

# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_mountain_active.csv")
# trpo_mountain_active_avg_rwd = np.array(data["AverageReturn"])

# print ("trpo_mountain_avg_rwd", trpo_mountain_avg_rwd.shape)
# print ("trpo_mountain_active_avg_rwd", trpo_mountain_active_avg_rwd.shape)



# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_cost.csv")
# trpo_grid_oracle = np.array(data["AverageReturn"])
# trpo_grid_oracle = trpo_grid_oracle[0:300]

# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_cost_active.csv")
# trpo_grid_agent = np.array(data["AverageReturn"])


# print ("trpo_grid_oracle", trpo_grid_oracle.shape)
# print ("trpo_grid_agent", trpo_grid_agent.shape)


# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_goal.csv")
# trpo_oracle_goal = np.array(data["AverageReturn"])
# trpo_oracle_goal = trpo_oracle_goal[0:300]

# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_goal_change_active.csv")
# trpo_agent_goal = np.array(data["AverageReturn"])


# print ("trpo_oracle_goal", trpo_oracle_goal.shape)
# print ("trpo_agent_goal", trpo_agent_goal.shape)


# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_goal_scene2.csv")
# trpo_oracle = np.array(data["AverageReturn"])
# trpo_oracle = trpo_oracle[0:300]

# data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/progress_goal_change_active.csv")
# trpo_agent = np.array(data["AverageReturn"])



data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/grid_dynamic_goal.csv")
trpo_oracle = np.array(data["AverageReturn"])
trpo_oracle = trpo_oracle[0:300]

data = pd.read_csv("/Users/Riashat/Documents/PhD_Research/RLLAB/rllab/learning_active_learning/learning_ask_help/data/grid_dynamic_goal_active.csv")
trpo_agent = np.array(data["AverageReturn"])





def double_plot(stats1, stats2,  smoothing_window=1, noshow=False):
    ## Figure 1
    fig = plt.figure(figsize=(40, 5))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "red", linewidth=2.5, label="Oracle (TRPO)")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "blue", linewidth=2.5, label="TRPO + Asking Oracle" )  

    plt.legend(handles=[cum_rwd_1, cum_rwd_2])
    plt.xlabel("Epsiode")
    plt.ylabel("Average Return")
    plt.title("Grid World Environment - Penalizing Oracle Query - Dynamically Changing Goal")
  
    plt.show()
    
    return fig



def main():
    double_plot(trpo_oracle, trpo_agent)

if __name__ == '__main__':
    main()