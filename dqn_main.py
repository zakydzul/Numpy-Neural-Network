import numpy as np
import tensorflow as tf
from dqn_utils import Agent
import gym

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    lr = 0.001
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr,
                input_dims=env.observation_space.shape,
                n_actions=env.action_space.n, max_mem_size=1000000, batch_size=64,
                epsilon_min=0.01)

    scores =[]
    epsilon_history =[]

    for i in range(n_games):
        terminal = False
        score = 0
        observation = env.reset()
        while not terminal:
            action = agent.ChooseAction(observation)
            observation_, reward, terminal, info = env.step(action)
            score =+ reward
            agent.StoreTransition(observation, action, reward, observation_, terminal)
            observation = observation_
            agent.Learning()
        epsilon_history.append(agent.epsilon)
        scores.append(score)

        avg_score=np.mean(scores[-100:])
        print('episode:',i,'score %.2f' % score,
                'average_score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
