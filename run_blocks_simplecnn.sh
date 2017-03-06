#!/bin/bash
out_dir=results_temp/blocks_simplecnn

python ./run_pg.py --env Blocks-v0\
 --agent modular_rl.agentzoo.TrpoAgent\
 --outdir ${out_dir}\
 --n_iter 10000\
 --snapshot_every 10\
 --video_record_every 10\
 --timesteps_per_batch 300\
 --timestep_limit 10\
 --vis_force True\
 --use_hdf 1 "$@"
