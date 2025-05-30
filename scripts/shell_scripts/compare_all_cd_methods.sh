#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=16:mem=128gb:ngpus=1
#PBS -N bnlearn_aba_experiment

cd $PBS_O_WORKDIR

# Setup env variables and the python venv
. scripts/shell_scripts/setup.sh

python scripts/python_scripts/compare_all_cd_methods.py \
    --n-runs 50 \
    --sample-size 5000 \
    --device 0

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the bnlearn experiment script."
    exit 1
fi
