#!/usr/bin/env python

# Importing math and visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import io as sio
import h5py
import yaml


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
        gr_indx = -1
        for name in graph_names:
            self.graph_names.append(name)
            if isinstance(name, str):
                gr_indx += 1
                fig_free += 1
                self.figures[name] = plt.figure(fig_free, figsize=fig_size)
                self.axis[name] = self.figures[name].add_subplot(111)
                self.axis[name].set_title(name, fontsize=18)
                self.colors[name] = self.colors_all[gr_indx % len(self.colors_all)]
            elif isinstance(name, list):
                fig_free += 1
                fig_cur = plt.figure(fig_free, figsize=fig_size)
                axis_cur = fig_cur.add_subplot(111)
                # axis_cur.set_title(name, fontsize=18)
                name_count = 0
                for name_i in name:
                    name_count += 1
                    gr_indx += 1
                    self.figures[name_i] = fig_cur
                    self.axis[name_i] = axis_cur
                    self.colors[name_i] = self.colors_all[name_count % len(self.colors_all)]
            else:
                raise ValueError("PLOT GRAPHS: string or list of strings are expected as graph name")

        # plt.draw()
        plt.show()
        if out_dir[-1] != '/':
            out_dir += '/'
        self.out_dir = out_dir

    # def adj_axis(self, fig, ax):
    #     plt.draw()  # to know size of legend
    #
    #     padLeft = ax.get_position().x0 * fig.get_size_inches()[0]
    #     padBottom = ax.get_position().y0 * fig.get_size_inches()[1]
    #     padTop = (1 - ax.get_position().y0 - ax.get_position().height) * fig.get_size_inches()[1]
    #     padRight = (1 - ax.get_position().x0 - ax.get_position().width) * fig.get_size_inches()[0]
    #     dpi = fig.get_dpi()
    #     padLegend = ax.get_legend().get_frame().get_width() / dpi
    #
    #     widthAx = 3  # inches
    #     heightAx = 3  # inches
    #     widthTot = widthAx + padLeft + padRight + padLegend
    #     heightTot = heightAx + padTop + padBottom
    #
    #     # resize ipython window (optional)
    #     posScreenX = 1366 / 2 - 10  # pixel
    #     posScreenY = 0  # pixel
    #     canvasPadding = 6  # pixel
    #     canvasBottom = 40  # pixel
    #     ipythonWindowSize = '{0}x{1}+{2}+{3}'.format(int(round(widthTot * dpi)) + 2 * canvasPadding
    #                                                  , int(round(heightTot * dpi)) + 2 * canvasPadding + canvasBottom
    #                                                  , posScreenX, posScreenY)
    #     fig.canvas._tkcanvas.master.geometry(ipythonWindowSize)
    #     plt.draw()  # to resize ipython window. Has to be done BEFORE figure resizing!
    #
    #     # set figure size and ax position
    #     fig.set_size_inches(widthTot, heightTot)
    #     ax.set_position([padLeft / widthTot, padBottom / heightTot, widthAx / widthTot, heightAx / heightTot])
    #     plt.draw()
    #     plt.show()

    def plot(self, name, data, indx, title=None):
        """
        Plot data.
        :param name: (string or list of strings)
        :param data: (array or list of arrays or dictionary)
        :param indx: (array)
        :param title: (string)
        :return:
        """
        if isinstance(name, str):
            name = [name]

        count = 0
        self.axis[name[0]].clear()

        for name_i in name:
            if isinstance(data, dict):
                data_plot = data[name_i]
            else:
                data_plot = data[count]

            if isinstance(data_plot[0], np.ndarray):
                data_plot = np.concatenate(data_plot, axis=0)
                print 'Plot dimenstions data_plot', data_plot.shape
                if data_plot.ndim >= 2:
                    if data_plot.ndim != 2:
                        data_plot = np.reshape(data_plot, (data_plot.shape[0], -1))
                    for i in range(0, data_plot.shape[1]):
                        print 'Plot dimenstions data_plot', data_plot.shape, ' data_plot[:,i] = ', data_plot[:,i].shape
                        self.axis[name_i].plot(indx, data_plot[:, i], self.colors_all[i % len(self.colors_all)], label=name_i + ('_%d' % i))
                else:
                    self.axis[name_i].plot(indx, data_plot, self.colors[name_i], label=name_i)
            else:
                self.axis[name_i].plot(indx, data_plot, self.colors[name_i], label=name_i)
            count += 1

        name = name_i

        if title is not None:
            self.axis[name].set_title(title)

        # plt.tight_layout()
        self.axis[name_i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0.)
        # self.axis[name_i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.draw()
        plt.show()

        print 'axis = ', self.axis[name], ' name = ', name
        # self.adj_axis(self.figures[name], self.axis[name])
        plt.pause(.01)
        # plt.show()
        # plt.draw()

    def save(self):
        if self.out_dir is not None:
            for name in self.graph_names:
                if isinstance(name, str):
                    filename = self.out_dir + 'fig__' + name + '.png'
                else:
                    filename = self.out_dir + 'fig__' + "_".join(name) + '.png'
                    name = name[0]
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