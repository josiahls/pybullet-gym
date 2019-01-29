from time import sleep

import gym
import pybulletgym.envs

env = gym.make('FetchPickAndPlace-v0')
env.reset()
env.render(mode="rgb_array")
for _ in range(1000):
    print(env.render(mode="rgb_array"))
    env.step(env.action_space.sample())
    sleep(1)
