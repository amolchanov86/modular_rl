#!/usr/bin/env python
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from modular_rl import plot_results as pltres
import yaml

outdir = '/tmp/gym_basic_test'

yaml_stream = file("config/train_params.yaml", 'r')
params = yaml.load(yaml_stream)
print 'Params = ', params

plotter = pltres.plot_graphs(graph_names=params['plot_names'],
                             out_dir=outdir,
                             fig_start=2)


stat = {}
indx = []
episode_count = 1000

magnitudes = {}
bias = 1.0
freq = 1/100.
for name in params['plot_names']:
    if isinstance(name, list):
        for name_i in name:
            magnitudes[name_i] = np.random.random() + bias
            stat[name_i] = []
    else:
        magnitudes[name] = np.random.random() + bias
        stat[name] = []

print 'magnitudes = ', magnitudes

def gen_data(i, names, stat, manitudes):
    data =  np.sin(2*np.pi*freq*i) + bias
    for name in names:
        if isinstance(name, list):
            for name_i in name:
                stat[name_i].append(manitudes[name_i] * data)
        else:
            stat[name].append(manitudes[name] * data)


## RUNNING
for i in range(episode_count):
    indx.append(i)
    print 'episode = ', i
    gen_data(i, params['plot_names'], stat, magnitudes)
    # axis.plot(stat['EpRewMean'], 'r')
    for name in plotter.graph_names:
        plotter.plot(name, data=stat, indx=indx)
    plt.pause(0.01)

plt.draw()
plt.pause(10.)

