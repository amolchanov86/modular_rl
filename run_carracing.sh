#!/bin/bash
out_file=results_temp/exp_current
if [ -d "${out_file}.dir" ]; then
  rm -rf ${out_file}.h5 
  rm -rf ${out_file}.dir 
fi

python ./run_pg.py --env CarRacing-v0\
 --agent modular_rl.agentzoo.TrpoAgent\
 --outfile ${out_file}.h5\
 --n_iter 250\
 --snapshot_every 10\
 --video_record_every 10\
 --timesteps_per_batch 3200\
 --use_hdf 1 "$@"
