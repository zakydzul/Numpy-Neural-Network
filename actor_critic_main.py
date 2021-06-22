from actor_critic_utils import Agent
import gym
import numpy as np
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 2000

    best_score = env.reward_range[0]
    score_history =[]

    for i in range(n_games):
        state = env.reset()
        terminal = False
        score = 0
        while not terminal:
            action = agent.ChooseAction(state)
            future_state, reward, terminal, info = env.step(action)
            score += reward
            agent.Learning(state, action, reward, future_state, terminal)
            state = future_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print('episode',i,'score %.1f' % score, 'avg_score %.1f' % avg_score)
    
    print('Best Score : ', best_score)



