from time import sleep

import gym
import pybulletgym.envs
import numpy as np
import pybullet as p

env = gym.make('FetchPickAndPlace-v0')  # type: gym.Env

env.render(mode="human")
env.reset()

baseId = -1
for i in range(p.getNumBodies()):
    print(p.getBodyInfo(i))
    if p.getBodyInfo(i)[0].decode() == "base_link":
        baseId = i
        print("found base")

for i in range(5):
    for _ in range(100):
        # print(env.render(mode="human"))
        # env.step(env.action_space.sample())

        fetchPos, fetchOrn = p.getBasePositionAndOrientation(baseId)
        distance = 2
        yaw = 90
        p.resetDebugVisualizerCamera(distance, yaw, -20, fetchPos)

        results = env.step(np.zeros(env.action_space.high.shape))
        # sleep(.5)
        sleep(0.02)
    print('Resetting')
    env.reset()
