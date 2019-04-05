from time import sleep

import gym
import pybulletgym.envs
import numpy as np
import pybullet as p

env_list = ['FetchPickKnifeAndCutTestEnv-v0']

for env_name in env_list:
    print(f'Testing {env_name}')
    # Load the OpenAI gym env
    env = gym.make(env_name)  # type: gym.Env

    # Render the display and perform reset operations that set up the state
    env.reset()
    env.render(mode="rgb_array")
    env.reset()

    # Find the robot's base
    baseId = -1
    for i in range(p.getNumBodies()):
        print(p.getBodyInfo(i))
        if p.getBodyInfo(i)[0].decode() == "base_link":
            baseId = i
            print("found base")

    for i in range(50):
        for _ in range(50):
            # print(env.render(mode="human"))
            # results = env.step(env.action_space.sample())
            fetchPos, fetchOrn = p.getBasePositionAndOrientation(baseId)
            distance = 1.5
            yaw = 90
            p.resetDebugVisualizerCamera(distance, yaw, -45, fetchPos)

            results = env.step(np.zeros(env.action_space.high.shape))
        print('Resetting')
        env.reset()
