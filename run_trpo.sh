#!/bin/bash
python ./run_pg.py --env CarRacing-v0 --agent modular_rl.agentzoo.TrpoAgent --outfile results_temp ${1}
