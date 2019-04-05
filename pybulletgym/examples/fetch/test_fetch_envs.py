from time import sleep

import gym
import pybulletgym.envs
import numpy as np
import pybullet as p

env_list = [
    'FetchPickKnifeAndCutTestEnv-v0',
    'FetchMoveBlockEnv-v0',
    'FetchCutBlockEnv-v1'
]

for env_name in env_list:
    sleep(1)
    print(f'Testing {env_name}')
    # Load the OpenAI gym env
    env = gym.make(env_name)  # type: gym.Env

    # Render the display and perform reset operations that set up the state
    env.reset()
    # In order to loop this, we need it to run rgb_array not human. This is due to the global env variables
    # being initialized and then being confused on re-init
    env.render(mode="rgb_array")
    # env.render(mode="human")
    # env.reset()

    # Find the robot's base
    baseId = -1
    for i in range(env.env._p.getNumBodies()):
        print(env.env._p.getBodyInfo(i))
        if env.env._p.getBodyInfo(i)[0].decode() == "base_link":
            baseId = i
            print("found base")

    for i in range(2):
        for _ in range(50):
            # print(env.render(mode="human"))
            # results = env.step(env.action_space.sample())
            fetchPos, fetchOrn = env.env._p.getBasePositionAndOrientation(baseId)
            distance = 1.5
            yaw = 90
            env.env._p.resetDebugVisualizerCamera(distance, yaw, -45, fetchPos)

            results = env.step(np.zeros(env.action_space.high.shape))
        print('Resetting')
        print(env.env.get_full_state())
        env.reset()
