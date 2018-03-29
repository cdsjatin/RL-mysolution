import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt
from lake_envs import *

def learn_Q_QLearning(env, num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  
  ############################
  # YOUR IMPLEMENTATION HERE #
  
  
  #Q = np.abs(np.random.randn(env.nS, env.nA))
  Q = np.zeros(shape=(env.nS, env.nA), dtype='float')
  #Q[-1,:] = np.zeros(shape=(env.nA), dtype='float')
  
  
  
  for i in range(num_episodes):
      
      s = np.random.randint(0,env.nS,dtype='int')
      
      while 1:
          
          if np.random.rand() > e:
              a = np.argmax(Q[s])
          else:
              a = np.random.randint(0, env.nA, dtype='int')
          
          _, next_s, reward, terminal = random.choice(env.P[s][a][:])
          
          Q[s][a] += lr * (reward + gamma* np.max( Q[next_s] ) - Q[s][a])
            
          s = next_s
          
          if terminal :
              break
    
      e *= decay_rate
      #print(i)
    
  ############################
  
  #print(Q)

  return Q

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  
  while not done:
    #env.render()
    #time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  #print "Episode reward: %f" % episode_reward
  return episode_reward

# Feel free to run your own debug code in main!
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  Q = learn_Q_QLearning(env)
  render_single_Q(env, Q)
  
  reward = 0
  avg_reward = []
  
  trials = 100
  
  for i in range(1, trials):
    Q = learn_Q_QLearning(env, lr=0.7)
    reward += render_single_Q(env, Q)
    
    avg_reward.append(reward/trials)
    
    plt.plot(avg_reward)
    plt.show()
  #print('average rewards: ',reward/100)
    
  

if __name__ == '__main__':
    
    main()
    
    
