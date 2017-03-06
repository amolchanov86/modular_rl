#!/usr/bin/env python

# Importing math and visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import io as sio
import h5py


class plot_graphs():
    def __init__(self, graph_names, out_dir=None, x_name='samples', fig_start = 1, fig_size=(6,5)):
        # Turn on interactive plotting
        plt.ion()

        # Create the main, super plot
        self.colors_all = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.figures = {}
        self.axis = {}
        self.colors = {}
        fig_free = fig_start - 1
        self.graph_names = []
        gr_indx = -1;
        for name in graph_names:
            gr_indx += 1;
            self.graph_names.append(name)
            fig_free += 1
            self.figures[name] = plt.figure(fig_free, figsize=fig_size)
            self.axis[name] = self.figures[name].add_subplot(111)
            self.axis[name].set_title(name, fontsize=18)
            self.colors[name] = self.colors_all[gr_indx % len(self.colors_all)]
        # plt.draw()
        plt.show()
        if out_dir[-1] != '/':
            out_dir += '/'
        self.out_dir = out_dir

    def plot(self, name, data, indx, title=None):
        self.axis[name].plot(indx, data, self.colors[name])
        if title is not None:
            self.axis[name].set_title(title)
        plt.pause(.01)
        # plt.draw()
        # plt.show()

    def save(self):
        if self.out_dir is not None:
            for name in self.graph_names:
                filename = self.out_dir + 'fig_' + name + '.png'
                print 'Saving figure:', filename
                self.figures[name].savefig(filename)

## Tests functionality
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        help="Either list of h5 files or a yaml file containing the list of files"
    )
    parser.add_argument(
        "var",
        default="/diagnostics/EpRewMean",
        help="Variables to plot in h5 files (full path inside h5)"
    )

    args = parser.parse_args()
    files = args.files.split(',')
    vars = args.var.split(',')
    data = {}
    h5files = {}
    import h5py
    # for file in files:
    #     for var in vars:
    #         h5files[] = h5py.File(fname,"w")




if __name__ == '__main__':
    main(sys.argv)