#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""


from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym
from gym import wrappers
import env_postproc_wrapper as env_proc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])

    # Making name and folder structure for the output file
    fname = args.outfile[:-3] if args.outfile[-3:] == '.h5' else args.outfile
    mondir = fname + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)

    ###############################################################################
    # MAKING ENVIRONMENT
    env_src = make(args.env)
    env = env_proc.make_norm_env(env_src, normalize=False)
    env_spec = env.spec
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
        # Plot
        if args.plot:
            print "Animating rollout ..."
            animate_rollout(env, agent, min(500, args.timestep_limit))

    ###############################################################################
    # RUNNING TRAINING
    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg)

    ###############################################################################
    # CLEANING UP
    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try: hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception: print "failed to pickle env" #pylint: disable=W0703

    env.monitor.close()
