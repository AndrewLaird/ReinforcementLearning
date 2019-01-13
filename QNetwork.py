import gym
import numpy as np
import random
import time

Q_table = [] # should be a 2d array with one axis being states, 
# and the other being the number of possible moves



if __name__ == "__main__":
        environment_name = "FrozenLake-v0"
        env =  gym.make(environment_name)

