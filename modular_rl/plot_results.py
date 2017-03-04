#!/usr/bin/env python

# Importing math and visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy import io as sio


class plot_graphs():
    def __init__(self, graph_names, out_dir=None, x_name='samples', fig_size=(16,8)):
        # Turn on interactive plotting
        plt.ion()

        # Create the main, super plot
        figures = {}
        axis = {}
        fig_free = 0
        graph_names = []
        for name in graph_names:
            graph_names.append(name)
            fig_free += 1
            figures[name] = plt.figure(fig_free, figsize=fig_size)
            axis[name] = figures[name].add_subplot(111)
            axis[name].set_title(name, fontsize=18)
        plt.show()
        plt.draw()
        if out_dir[-1] != '/':
            out_dir += '/'
        self.out_dir = out_dir

    def plot(self, name, data, indx, title=None):
        figures[name].clear()
        axis[name].plot(data)
        if title is not None:
            axis[name].set_title(title)

    def save(self):
        if self.out_dir is not None:
            for name in self.graph_name:
                filename = self.out_dir + 'fig_' + name + '.png'
                print 'Saving figure:', filename
                axis[name].savefig(filename)



