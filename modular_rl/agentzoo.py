"""
In this codebase, the "Agent" is a container with the policy, value function, etc.
This file contains a bunch of agents
"""


from modular_rl import *
from gym.spaces import Box, Discrete
from collections import OrderedDict
from keras.models import Sequential
# from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import  BatchNormalization
from keras.utils import np_utils

from modular_rl.trpo import TrpoUpdater
from modular_rl.ppo import PpoLbfgsUpdater, PpoSgdUpdater

MLP_OPTIONS = [
    ("hid_sizes", comma_sep_ints, [64,64], "Sizes of hidden layers of MLP"),
    ("activation", str, "relu", "nonlinearity")
]

def make_mlps(ob_space, ac_space, cfg):
    assert isinstance(ob_space, Box)
    hid_sizes = cfg["hid_sizes"]
    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)
    net = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=ob_space.shape) if i==0 else {}
        net.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    if isinstance(ac_space, Box):
        net.add(Dense(outdim))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
        net.add(ConcatFixedStd())
    else:
        net.add(Dense(outdim, activation="softmax"))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    vfnet = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i==0 else {} # add one extra feature for timestep
        vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    vfnet.add(Dense(1))
    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline


def make_cnns(ob_space, ac_space, cfg):
    assert isinstance(ob_space, Box)
    hid_sizes = cfg["hid_sizes"]

    #TEMP: Hardcoding sizes here temporary


    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)
    net = Sequential()

    # Neural network
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AgentZoo: CNN agent is created"
    net.add(Convolution2D(32, 4, 4, border_mode='same',
                            input_shape=ob_space.shape,
                            activation=cfg["activation"],
                            subsample=(2, 2)))
    net.add(Convolution2D(32, 4, 4, activation=cfg["activation"],
                            subsample=(2, 2)))
    net.add(Convolution2D(32, 4, 4, border_mode='same', activation=cfg["activation"]))
    net.add(Flatten())
    net.add(Dense(256, activation=cfg["activation"])) #200 in DDPG paper
    net.add(Dense(128, activation=cfg["activation"]))

    # Creating the output layer
    if isinstance(ac_space, Box):
        net.add(Dense(outdim))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
        net.add(ConcatFixedStd())
    else:
        net.add(Dense(outdim, activation="softmax"))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)


    vfnet = Sequential()
    # for (i, layeroutsize) in enumerate(hid_sizes):
    #     inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i==0 else {} # add one extra feature for timestep
    #     vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AgentZoo: CNN Vf is created"
    vfnet.add(Convolution2D(32, 4, 4, border_mode='same',
                            input_shape=ob_space.shape,
                            activation=cfg["activation"],
                            subsample=(2, 2)))
    vfnet.add(Convolution2D(32, 4, 4, activation=cfg["activation"],
                            subsample=(2, 2)))
    vfnet.add(Convolution2D(32, 4, 4, border_mode='same', activation=cfg["activation"]))
    vfnet.add(Flatten())
    vfnet.add(Dense(256, activation=cfg["activation"])) #200 in DDPG paper
    vfnet.add(Dense(128, activation=cfg["activation"]))
    vfnet.add(Dense(1))
    
    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline



def make_cnns_oclmnist(ob_space, ac_space, cfg):
    print 'AGENTZOO: Creating OCLMNIST CNNs ...'
    assert isinstance(ob_space, Box)
    # hid_sizes = cfg["hid_sizes"]

    ####################################################################################
    ## SHARED
    # Creating model for shared visual features
    vis_feat_model = keras_tools.oclmnist_vis_feat(input_shape=ob_space.shape,
                                                   out_num=128)
    if isinstance(ac_space, Box):
        print '!!!!!!!!!! Continuous control initialized'
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)

    ####################################################################################
    ## AGENT NET
    print "AGENTZOO: Building actor network ..."
    input_img = Input(shape=ob_space.shape)
    x_vis = vis_feat_model(input_img, activation=cfg["activation"])

    x_act = Dense(128, activation=cfg["activation"])(x_vis)
    x_act = BatchNormalization(mode=1)(x_act)
    x_act = Activation(cfg["activation"])(x_act)

    x_act = Dense(128, activation=cfg["activation"])(x_act)
    x_act = BatchNormalization(mode=1)(x_act)
    x_act = Activation(cfg["activation"])(x_act)

    x_act = Dense(outdim, activation=cfg["activation"])(x_act)
    act_out = Activation("tanh")(x_act)
    net = Model(input=input_img, output=[act_out])

    # Creating the output layer
    if isinstance(ac_space, Box):
        net.add(Dense(outdim))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True) * 0.1)
        net.add(ConcatFixedStd())
    else:
        net.add(Dense(outdim, activation="softmax"))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True) * 0.1)
    policy = StochPolicyKeras(net, probtype)


    ####################################################################################
    ## VF NET
    print "AGENTZOO: Building value networks ..."
    vfnet_input_img = Input(shape=ob_space.shape)
    x_vf_vis = vis_feat_model(vfnet_input_img, activation=cfg["activation"])

    x_vf = Dense(128, activation=cfg["activation"])(x_vf_vis)
    x_vf = BatchNormalization(mode=1)(x_vf)
    x_vf = Activation(cfg["activation"])(x_vf)

    vf_out = Dense(1, activation=cfg["activation"])(x_vf)
    vfnet = Model(input=vfnet_input_img, output=[vf_out])

    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline


def make_deterministic_mlp(ob_space, ac_space, cfg):
    assert isinstance(ob_space, Box)
    hid_sizes = cfg["hid_sizes"]
    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)
    net = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=ob_space.shape) if i==0 else {}
        net.add(Dense(layeroutsize, activation="tanh", **inshp))
    inshp = dict(input_shape=ob_space.shape) if len(hid_sizes) == 0 else {}
    net.add(Dense(outdim, **inshp))
    Wlast = net.layers[-1].W
    Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    return policy

FILTER_OPTIONS = [
    ("filter", int, 1, "Whether to do a running average filter of the incoming observations and rewards")
]

def make_filters(cfg, ob_space):
    if cfg["filter"]:
        obfilter = ZFilter(ob_space.shape, clip=5)
        rewfilter = ZFilter((), demean=False, clip=10)
    else:
        obfilter = IDENTITY
        rewfilter = IDENTITY
    return obfilter, rewfilter


class AgentWithPolicy(object):
    def __init__(self, policy, obfilter, rewfilter):
        self.policy = policy
        self.obfilter = obfilter
        self.rewfilter = rewfilter
        self.stochastic = True
    def set_stochastic(self, stochastic):
        self.stochastic = stochastic
    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic = self.stochastic)
    def get_flat(self):
        return self.policy.get_flat()
    def set_from_flat(self, th):
        return self.policy.set_from_flat(th)
    def obfilt(self, ob):
        return self.obfilter(ob)
    def rewfilt(self, rew):
        return self.rewfilter(rew)

class DeterministicAgent(AgentWithPolicy):
    options = MLP_OPTIONS + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy = make_deterministic_mlp(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)
        self.set_stochastic(False)

class TrpoAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        # policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        policy, self.baseline = make_cnns_oclmnist(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = TrpoUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)

class PpoLbfgsAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + PpoLbfgsUpdater.options + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = PpoLbfgsUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)

class PpoSgdAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + PpoSgdUpdater.options + FILTER_OPTIONS
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = PpoSgdUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)