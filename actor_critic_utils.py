import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

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
        self.ActorCriticNet.compile(optimizer=Adam(learning_rate=alpha))

    def ChooseAction(self, observation):
        _, action_probs = self.ActorCriticNet(tf.convert_to_tensor([observation]))
        action_probs_distrib = tfp.distributions.Categorical(action_probs)
        action = action_probs_distrib.sample()
        log_prob = action_probs_distrib.log_prob(action)
        self.action = action

        return action.numpy()[0]

    def Learning(self, state, action,  reward, future_state, terminal):

        with tf.GradientTape(persistent=True) as tape:
            state_value, action_probs = self.ActorCriticNet(tf.convert_to_tensor([state]))
            future_state_value,_ = self.ActorCriticNet(tf.convert_to_tensor([future_state]))

            state_value = tf.squeeze(state_value)
            future_state_value = tf.squeeze(future_state_value)

            action_probs_distrib = tfp.distributions.Categorical(probs=action_probs)
            log_prob = action_probs_distrib.log_prob(self.action)

            delta = reward + self.gamma*future_state_value*(1-int(terminal))-state_value
            actor_loss = -log_prob*delta #gradient ascent to achieve maximum reward
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss
        
        gradient = tape.gradient(total_loss, self.ActorCriticNet.trainable_variables)
        self.ActorCriticNet.optimizer.apply_gradients(zip(
                    gradient, self.ActorCriticNet.trainable_variables
                    ))








        