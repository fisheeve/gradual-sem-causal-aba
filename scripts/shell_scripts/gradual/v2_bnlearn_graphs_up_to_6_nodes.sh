#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8:mem=128gb:cpu_type=rome
#PBS -N v2_bnlearn_graphs_up_to_6_nodes

# Setup env variables and the python venv
cd $PBS_O_WORKDIR
. scripts/shell_scripts/setup.sh

module load R
export CUDA_VISIBLE_DEVICES="[]"

python scripts/python_scripts/gradual/v2_bnlearn_graphs_up_to_6_nodes.py \
    --n-runs 50 \
    --search-depth 12 \
    --use-every-step-search "false"

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the bnlearn experiment script."
    exit 1
fi
