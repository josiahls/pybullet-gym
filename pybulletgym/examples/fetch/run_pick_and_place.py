from time import sleep

import gym
import pybulletgym.envs
import numpy as np
import pybullet as p

import utils


# Load the OpenAI gym env
env = gym.make('FetchPickKnifeAndCutEnv-v0')  # type: gym.Env

# Render the display and perform reset operations that set up the state
env.render(mode="human")
env.reset()

# Find the robot's base
baseId = -1
for i in range(p.getNumBodies()):
    print(p.getBodyInfo(i))
    if p.getBodyInfo(i)[0].decode() == "base_link":
        baseId = i
        print("found base")

# Start matplotlib to show the reward progression
# plotter = utils.Plotter()

for i in range(50):
    for _ in range(50):
        # print(env.render(mode="human"))
        # results = env.step(env.action_space.sample())
        # #
        # fetchPos, fetchOrn = p.getBasePositionAndOrientation(baseId)
        # distance = 1.5
        # yaw = 90
        # p.resetDebugVisualizerCamera(distance, yaw, -45, fetchPos)

        results = env.step(np.zeros(env.action_space.high.shape))

        # plotter.live_plotter(results[1])
        sleep(0.02)

    print('Resetting')
    env.reset()
    sleep(1)
