import matplotlib
import PyQt5

matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np


class Plotter(object):
    """
    This class is extended from https://engineersportal.com/blog/2018/8/14/real-time-graphing-in-python
    for being pretty and very useful.


    """

    def __init__(self, max_size=100):
        plt.style.use('ggplot')
        self.max_size = max_size
        self.x_vec = np.linspace(0, 1, max_size + 1)[0:-1]
        self.y_vec = np.random.randn(len(self.x_vec))
        self.line1 = []

    def live_plotter(self, value, identifier='', pause_time=0.1):
        self.y_vec[-1] = value
        if not self.line1:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            self.line1, = ax.plot(self.x_vec, self.y_vec, '-o', alpha=0.8)
            # update plot label/title
            plt.ylabel('Y Label')
            plt.title('Reward: {}'.format(identifier))
            plt.show()

        # after the figure, axis, and line are created, we only need to update the y-data
        self.line1.set_ydata(self.y_vec)
        # adjust limits if new data goes beyond bounds
        if np.min(self.y_vec) <= self.line1.axes.get_ylim()[0] or np.max(self.y_vec) >= self.line1.axes.get_ylim()[1]:
            plt.ylim([np.min(self.y_vec) - np.std(self.y_vec), np.max(self.y_vec) + np.std(self.y_vec)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)

        # Remove the oldest value
        self.y_vec = np.append(self.y_vec[1:], 0.0)
