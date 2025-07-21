#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=128gb:cpu_type=rome
#PBS -N experiment_v2_minimal_scale_no_collider

# Setup env variables and the python venv
cd $PBS_O_WORKDIR
. scripts/shell_scripts/setup.sh

module load R
export CUDA_VISIBLE_DEVICES="[]"

python scripts/python_scripts/gradual/experiment_v2_minimal_scale_no_collider.py \
    --n-runs 50 \
    --sample-size 5000 \
    --device 0

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the bnlearn experiment script."
    exit 1
fi
