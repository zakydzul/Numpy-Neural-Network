import tensorflow as tf
import  numpy as np
from tensorflow import keras

class ActorCriticModel(keras.Model):
    def __init__(self, name='actor_critic', fc1=1024, fc2=512, n_actions=2):
        super(ActorCriticModel, self).__init__()
        self.model_name = name
        self.fc1 = fc1
        self.fc2 = fc2
        self.n_actions = n_actions

        self.model_name = name

        self.fc1_layer = keras.layers.Dense(self.fc1, activation='relu')
        self.fc2_layer = keras.layers.Dense(self.fc2, activation='relu')

        self.val = keras.layers.Dense(1, activation=None) #output for crtic network --> preditcting value given a state
        self.pi = keras.layers.Dense(self.n_actions, activation='softmax') #output for action network 

    def call(self, state):
        state_val = self.fc1_layer(state)
        state_val = self.fc2_layer(state_val)

        val = self.val(state_val)
        pi = self.pi(state_val)

        return val, pi

class Agent:
    def __init__(self, alpha=3e-1, gamma=0.99, n_actions=2):
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.n_actions_space = [i for i in range(self.n_actions)]

        self.ActorCriticNet = ActorCriticModel(n_actions=n_actions)





        