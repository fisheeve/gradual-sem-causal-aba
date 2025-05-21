# gradual-sem-causal-aba

## Prerequisites

Python 3.12  
R https://cran.rstudio.com/

## Installation

Clone the repo from GitHub.
```bash
git clone git@github.com:fisheeve/gradual-sem-causal-aba.git
cd gradual-sem-causal-aba
```
Configure environment variables to make it work for your system by editing the .env file.

Create and setup a python virtual environment, clone sub-repos, and install R packages required:
```bash
make
```

Activate the virtual environment before running experiments or tests:
```bash
source .venv/bin/activate
```

## Unit-tests
Run the following in the terminal
```bash
export PYTHOPATH=. && pytest tests/unit
```
