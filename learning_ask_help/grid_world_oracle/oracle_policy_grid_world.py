"""
Step taken by the oracle
"""

def oracle_action(observation, i_a):
    #left=0, down=1,right=2,up=3,

    if i_a ==0:
        observation = observation[0]
    elif i_a == 1:
        observation = observation[1]

    if observation == 0:
        action = 3


    if observation == 1:
        action = 3

    if observation == 2:
        action = 2

    if observation == 3:
        action = 1


    if observation == 4:
        action = 2


    if observation == 5:
        action = 3


    if observation ==6:
        action = 2


    if observation == 7:
        action = 1


    if observation == 8:
        action = 3


    if observation == 9:
        action = 3


    if observation == 10:
        action = 2


    if observation == 11:
        action = 1


    if observation == 12:
        action = 3
   

    if observation == 13:
        action = 3


    if observation == 14:
        action = 3


    if observation == 15:
        action = 1


    return action
