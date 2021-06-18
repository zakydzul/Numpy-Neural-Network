import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

class ReplayBuffer():
    def __init__(self, max_memory_size, input_dims) :
        self.max_memory_size = max_memory_size
        self.memory_counter = 0
        #---create buffer memory----#
        self.state_memory = np.zeros((self.max_memory_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.max_memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_memory_size, dtype=np.float32)
        self.future_state_memory = np.zeros((self.max_memory_size, *input_dims),
                                dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_memory_size, dtype=np.int32) #binary

    def StoreTransition(self, state, action, reward, future_state, terminal):
        index = self.memory_counter % self.max_memory_size #iterate through unocuppied memory
        self.state_memory[index] = state
        self.future_state_memory[index] = future_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1-int(terminal)
        self.memory_counter +=1

    def returunMemoryCounter(self):
        return self.memory_counter

    def SampleBuffer(self, batch_size):
        max_index = min(self.memory_counter, self.max_memory_size)
        batch = np.random.choice(max_index, batch_size, replace=False)

        states = self.state_memory[batch]
        future_states = self.future_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, future_states, terminal

def BuildDQN(lr, n_acions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu',),
        keras.layers.Dense(fc2_dims, activation='relu',),
        keras.layers.Dense(n_acions, activation=None)
    ])

    model.compile(optimizer =Adam(learning_rate=lr), loss= 'mean_squared_error')
    return model

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                    input_dims, epsilon_decrement = 1e-3, epsilon_min =0.01, 
                    max_mem_size=1000000, fname='dqn_model'):
        self.action = [i for i  in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decrement = epsilon_decrement
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = fname
        #create_memory_buffer
        self.memory = ReplayBuffer(max_mem_size, input_dims)
        #create_dqn
        self.q_eval = BuildDQN(lr, n_actions, input_dims, 256, 256)

    def StoreTransition(self, state, action, reward, future_state, terminal):
        self.memory.StoreTransition(state, action, reward, future_state, terminal)

    def ChooseAction(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)
        return action

    def Learning(self):
        if self.memory.memory_counter < self.batch_size:
            return

        states, actions, rewards, future_state, terminal =\
            self.memory.SampleBuffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_future = self.q_eval.predict(future_state)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + \
            self.gamma * np.max(q_future, axis=1) * terminal

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.epsilon_decrement if self.epsilon > \
            self.epsilon_min else self.epsilon_min     

    def SaveModel(self):
        self.q_eval.save(self.model_file)

            
    




    