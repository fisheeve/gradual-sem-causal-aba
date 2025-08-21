#!/bin/bash
#PBS -l walltime=13:00:00
#PBS -l select=1:ncpus=8:mem=128gb:cpu_type=rome
#PBS -N v2_bnlearn_sachs_batch10

# Setup env variables and the python venv
cd $PBS_O_WORKDIR
. scripts/shell_scripts/setup.sh

module load R
export CUDA_VISIBLE_DEVICES="[]"


python scripts/python_scripts/gradual/v2_run_for_bnlearn.py \
    --path "./results/gradual/v2_run_for_bnlearn/sachs/batch10" \
    --search-depth 12 \
    --name "sachs" \
    --neighbourhood-n-nodes 7 \
    --c-set-size 0 \
    --n-runs 50 \
    --batch-number 10 \
    --total-batches 10

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the find_optimal_params experiment script."
    exit 1
fi
