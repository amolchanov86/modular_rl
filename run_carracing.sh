#!/bin/bash
export outdir='results_temp/car_racing'

python ./run_pg.py --env CarRacing-v0\
 --agent modular_rl.agentzoo.TrpoAgent\
 --outdir ${outdir}\
 --n_iter 250\
 --snapshot_every 10\
 --video_record_every 10\
 --timesteps_per_batch 3200\
 --use_hdf 1 "$@"
