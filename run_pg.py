#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""
from gym.envs import make
from modular_rl import *
from modular_rl import plot_results as pltres
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym
from gym import wrappers
import env_postproc_wrapper as env_proc
import gym_blocks

import e2eap_training.env_blocks.blocks_action_wrap as baw
import e2eap_training.env_blocks.blocks_reward_wrap as brw
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
        env = baw.action2dWrap(env)

    env = normwrap.make_norm_env(env=env,
                                 normalize=cfg['env_norm'])

    if env.spec.id[:6] == 'Blocks':
        env = brw.nnetReward(env, nnet_params=cfg,
                             log_dir=logdir_root + 'classif_wrong_pred', framework='keras')

        env.unwrapped.step_limit = cfg['timestep_limit']
        env.unwrapped.frame_skip = 10
        env.unwrapped.reload_model(yaml_path='config/blocks_config.yaml')
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])

    # Making name and folder structure for the output file
    out_dir = args.outdir
    if os.path.exists(out_dir):
        if osp.exists(out_dir):
            raw_input("WARNING: %s already exists. Press ENTER to DELETE existing and continue. (exit with Ctrl+C)" % out_dir)
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    if out_dir[-1] != '/':
        out_dir += '/'
    mondir = out_dir + 'gym_log/'
    if os.path.exists(mondir): shutil.rmtree(mondir, out_dir=out_dir)
    os.mkdir(mondir)


    ###############################################################################
    # MAKING ENVIRONMENT
    env = make(args.env)

    update_argument_parser(parser, core.ENV_OPTIONS)
    args, __ = parser.parse_known_args()
    cfg = args.__dict__
    print 'Env updated Config = ', cfg
    env = wrap_env(env, cfg=cfg, logdir_root=out_dir)

    # Bugfix: render should be called before agents
    env.reset()
    if args.plot:
        env.render()

    # --- MONITORING
    # Video scheduler (function that states when we should record a video)
    video_step_episodes = args.video_record_every
    def video_schedule(episode_id):
        global video_step_episodes
        result = (episode_id % video_step_episodes == 0)
        if result:
            print "!!!!!!!!!!!!!!!!!!!! Recording a video !!!!!!!!!!!!!!!!!"
        return result


    env = wrappers.Monitor(env, mondir, video_callable=video_schedule if args.video_record_every else VIDEO_NEVER)

    ###############################################################################
    # INITIALIZATION OF THE AGENT
    # The function gets agents name from args and converts this name into function (interesting and a smart move)
    # Ex: modular_rl.agentzoo.TrpoAgent  will get TrpoAgent constructor from modular_rl.agentzoo module
    # PS: previously he used ctor = constructor
    agent_constructor = get_agent_cls(args.agent)

    # Previously we parsed only known arguments. Now, it parses the rest according to agents options
    # PS: I usually create a single params dictionary and read it from some yaml file or fill it in a script
    update_argument_parser(parser, agent_constructor.options)
    args = parser.parse_args()

    # Updating time step limit
    if args.timestep_limit == 0 or args.video_record_every:
        if env.spec.id[:6] != 'Blocks': #For blocks we can actually control number of time steps
            args.timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    # Initializing an agent
    # Here he just gets params dictionary from command line. I think it is just less convenient. I prefer yaml files
    cfg = args.__dict__
    np.random.seed(args.seed)
    agent = agent_constructor(env.observation_space, env.action_space, cfg)

    ###############################################################################
    # DIAGNOSTICS AND SNAPSHOTS
    # Opening hdf5 file for diagnostics
    # Preparing dictionary for diagnostics
    # Preparing hdf5 handler for saving diagnostics upon exit
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)

    ###############################################################################
    # Plotting handling
    fig_handler = pltres.plot_graphs(graph_names=['EpRewMean'], out_dir=out_dir)

    ###############################################################################
    # CREATING ITERATION END HANDLER:
    # - saves diagnostics
    # - saves snapshots
    COUNTER = 0 # Iteration counter
    def callback(stats):
        global COUNTER
        COUNTER += 1
        # Print stats
        print "*********** Iteration %i ****************" % COUNTER
        print tabulate(filter(lambda (k,v) : np.asarray(v).size==1, stats.items())) #pylint: disable=W0110
        # Store to hdf5
        if args.use_hdf:
            for (stat,val) in stats.items():
                if np.asarray(val).ndim==0:
                    diagnostics[stat].append(val)
                else:
                    assert val.ndim == 1
                    diagnostics[stat].extend(val)
            if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)):
                hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))

        ######################################
        ## Animating rollouts
        if args.plot:
            print "Animating rollout ..."
            animate_rollout(env, agent, min(500, args.timestep_limit))

        ######################################
        ## Plotting progress and saving figures
        samples_num = 0
        samp_i = []
        for val in diagnostics['EpSampNum']:
            samples_num += val
            samp_i.append(samples_num)

        for name in fig_handler.graph_names:
            fig_handler.plot(name, indx=samp_i, data=diagnostics[name])

        fig_handler.save()

    ###############################################################################
    # RUNNING TRAINING
    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg)

    ###############################################################################
    # CLEANING UP
    if args.use_hdf:
        hdf['env_id'] = env.spec.id
        try: hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception: print "failed to pickle env" #pylint: disable=W0703

    env.monitor.close()
