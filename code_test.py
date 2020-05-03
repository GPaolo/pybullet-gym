# Created by Giuseppe Paolo 
# Date: 20/04/2020

import gym
import pybullet
import pybulletgym
import matplotlib.pyplot as plt

env = gym.make('AntMuJoCoMazeEnv-v0')

env.render()
env.reset()

for i in range(2000):
  a = env.step(env.action_space.sample())

image = env.render(mode='rgb_array')
plt.figure()
plt.imshow(image)
plt.show()

