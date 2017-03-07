#!/usr/bin/env python
"""
Load a snapshotted agent from an hdf5 file and animate it's behavior
"""
import os
import argparse
import cPickle, h5py, numpy as np, time
from collections import defaultdict
import gym
from run_pg import wrap_env

def printDictTypes(dict_in, indent='  '):
    for key in dict_in.keys():
        print indent, key, ' : ', type(dict_in[key])
        if isinstance(dict_in[key], dict):
            printDictTypes(dict_in[key], indent=indent + '  ')

def h5params2dict(dict_in, indent='  '):
    params = {}
    for key in dict_in.keys():
        params[key] = dict_in[key].value
        print indent, key, ' : ', type(params[key])
    return params

def animate_rollout(env, agent, n_timesteps,delay=.01):
    infos = defaultdict(list)
    ob = env.reset()
    if hasattr(agent,"reset"): agent.reset()
    env.render()
    for i in xrange(n_timesteps):
        ob = agent.obfilt(ob)
        a, _info = agent.act(ob)
        (ob, rew, done, info) = env.step(a)
        env.render()
        if done:
            print("terminated after %s timesteps"%i)
            break
        for (k,v) in info.items():
            infos[k].append(v)
        infos['ob'].append(ob)
        infos['reward'].append(rew)
        infos['action'].append(a)
        time.sleep(delay)
    return infos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--timestep_limit",type=int)
    parser.add_argument("--snapname")
    parser.add_argument("--outdir", default='results_temp/test_agent/')
    args = parser.parse_args()

    hdf = h5py.File(args.hdf,'r')

    snapnames = hdf['agent_snapshots'].keys()
    print "snapshots:\n",snapnames
    if args.snapname is None: 
        snapname = snapnames[-1]
    elif args.snapname not in snapnames:
        raise ValueError("Invalid snapshot name %s"%args.snapname)
    else: 
        snapname = args.snapname

    # params = hdf["params"]
    params = h5params2dict(hdf["params"])
    out_dir = args.outdir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    env = gym.make(hdf["env_id"].value)
    env = wrap_env(env, cfg=params, logdir_root=out_dir)

    agent = cPickle.loads(hdf['agent_snapshots'][snapname].value)
    agent.stochastic=False

    timestep_limit = args.timestep_limit or env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    while True:
        infos = animate_rollout(env,agent,n_timesteps=timestep_limit, 
            delay=1.0/env.metadata.get('video.frames_per_second', 30))
        for (k,v) in infos.items():
            if k.startswith("reward"):
                print "%s: %f"%(k, np.sum(v))
        raw_input("press enter to continue")

if __name__ == "__main__":
    main()