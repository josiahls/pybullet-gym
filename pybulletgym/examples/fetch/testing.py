from time import sleep

import gym
import pybulletgym.envs
import numpy as np
import pybullet as p
import matplotlib
import PyQt5
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt


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
rewards = []
plt.style.use('ggplot')


def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Reward: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1

size = 100
x_vec = np.linspace(0,1,size+1)[0:-1]
y_vec = np.random.randn(len(x_vec))
line1 = []

for i in range(50):
    for _ in range(100):
        # print(env.render(mode="human"))
        # env.step(env.action_space.sample())

        fetchPos, fetchOrn = p.getBasePositionAndOrientation(baseId)
        distance = 2
        yaw = 90
        p.resetDebugVisualizerCamera(distance, yaw, -40, fetchPos)

        results = env.step(np.zeros(env.action_space.high.shape))
        rewards.append(results[1])

        y_vec[-1] = results[1]
        line1 = live_plotter(x_vec, y_vec, line1)
        y_vec = np.append(y_vec[1:], 0.0)

    print('Resetting')
    env.reset()
    sleep(1)
