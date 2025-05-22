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

Create and setup a python virtual environment, clone sub-repos, and install R packages required:
```bash
make
```

Activate the virtual environment before running experiments or tests:
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
