#!/usr/bin/env python
"""
Load a snapshotted agent from an hdf5 file and animate it's behavior
"""

import argparse
import cPickle, h5py, numpy as np, time
from collections import defaultdict
import gym
import gym_blocks
from gym import wrappers

import e2eap_training.env_blocks.blocks_action_wrap as baw
# import e2eap_training.env_blocks.blocks_reward_wrap as brw
import e2eap_training.core.env_postproc_wrapper as normwrap

def wrap_env(env, logdir_root, cfg):
    """
    Set of wrappers for env normalization and added functionality
    :param env:
    :param logdir_root:
    :param cfg:
    :return:
    """

    if logdir_root[-1] != '/':
        logdir_root += '/'

    if env.spec.id[:6] == 'Blocks':
        if cfg['vis_force']:
            # print_warn('Force visualization wrapper turned on')
            env = gym_blocks.wrappers.Visualization(env)
        # This wrapper should come before normalizer
        env = baw.action2dWrap(env, fz=0)

    env = normwrap.make_norm_env(env=env,
                                 normalize=cfg['env_norm'])

    if env.spec.id[:6] == 'Blocks':
        env.unwrapped.reload_model(yaml_path='config/blocks_config.yaml')
        # All additional parameters should be specified AFTER reloading (everytime you reload re-spicify them)
        env.unwrapped.step_limit = cfg['timestep_limit']
    return env


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

    cfg ={}
    cfg['env_norm'] = True
    cfg['vis_force'] = True
    cfg['timestep_limit'] = args.timestep_limit

    logdir = 'results_temp/sim_results/'
    mondir = logdir + 'gym_log/'
    env = gym.make(hdf["env_id"].value)
    VIDEO_NEVER = False
    def video_schedule(episode_id):
        return True

    env = wrap_env(env, cfg=cfg, logdir_root='results_temp/sim_results')
    # env = wrappers.Monitor(env, mondir, video_callable=video_schedule if args.video_record_every else VIDEO_NEVER)
    env = wrappers.Monitor(env, mondir, video_callable=video_schedule)

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