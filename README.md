# gradual-sem-causal-aba

## Prerequisites

Python 3.12  
R https://cran.rstudio.com/
[OPTIONAL] conda 25.1 (Miniconda installation would suffice)  


## Installation

Clone the repo from GitHub.
```bash
git clone git@github.com:fisheeve/gradual-sem-causal-aba.git
cd gradual-sem-causal-aba
```

Run the following to create and setup a python virtual environment, clone sub-repos, and install R packages:
```bash
make
```
Running `make` will prompt inputting the location of Rscript and R Library of your setup. The value of those parameters will be stored in environment variables `RPATH` and `R_LIB_DIR` respectively. The values of these variables are then stored in a `.env` file. The resulting `.env` should look similar to the following:
```
RPATH=/your/path/to/R/4.4.2-gfbf-2024a/bin/Rscript

R_LIB_DIR=/your/path/to/miniforge3/envs/aba-env/lib/R/library
```

Finally, activate the virtual environment before running experiments or tests:
```bash
conda activate aba-env
```
or if you don't have conda installed:
```bash
source .venv/bin/activate
```

## Unit-tests
Run the following in the terminal
```bash
export PYTHONPATH=. && pytest tests/unit
```

## Experiments
All of the experiments relevant to this project are in folders `scripts/` and `notebooks/`. Experiment results are in the `results/` folder.
The `notebooks/` folder contains mostly demonstration notebooks that plot results yielded from python scripts in `scripts/`.
The `scripts/` contains python scripts to run experiments. It also contains shell scripts used for submitting python scripts as jobs on [Imperial HPC](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/getting-started/).
