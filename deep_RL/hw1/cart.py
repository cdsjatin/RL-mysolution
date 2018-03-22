import gym

env = gym.make('Hopper-v1')
env.reset()

for _ in range(1000):
    env.render(mode='rgb_array')
    env.step(env.action_space.sample())
