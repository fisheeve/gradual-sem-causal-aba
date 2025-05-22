PYTHON_VERSION=3.12
ENV_NAME=aba-env
VENV_DIR=.venv

ifneq (,$(wildcard .env))
    include .env
    export
endif

R_LIB_DIR?=$(HOME)/R/library

.PHONY: install clean

install: clean
	@echo "Setting up Python environment..."
	@if command -v conda > /dev/null 2>&1; then \
		echo "Conda found. Creating and activating conda environment..."; \
		conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION); \
		bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && pip install --upgrade pip && pip install -r requirements.txt"; \
	else \
		echo "Conda not found. Falling back to python venv..."; \
		python$(PYTHON_VERSION) -m venv $(VENV_DIR); \
		. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt; \
	fi

	@echo "Cloning required repositories..."
	git clone https://bitbucket.org/coreo-group/aspforaba.git
	git clone https://github.com/briziorusso/ArgCausalDisco.git
	git clone https://github.com/xunzheng/notears.git

	@echo "[TODO: fix this somehow] Applying custom changes to aspforaba sub-repo..."
	cd aspforaba && git apply ../aspforaba.diff

	@echo "Preparing R library directory at $(R_LIB_DIR)..."
	mkdir -p $(R_LIB_DIR)

	@echo "Installing additional dependencies..."
	R_INTERACTIVE=FALSE R_PAPERSIZE=letter PAGER=cat R_OPTS="--no-save --no-restore --quiet" \
	Rscript -e 'install.packages("BiocManager", repos="https://cloud.r-project.org", lib="$(R_LIB_DIR)")'

	R_INTERACTIVE=FALSE R_PAPERSIZE=letter PAGER=cat R_OPTS="--no-save --no-restore --quiet" \
	Rscript -e '.libPaths(c("$(R_LIB_DIR)", .libPaths())); BiocManager::install(c("SID", "bnlearn", "pcalg", "kpcalg", "glmnet", "mboost"))'

clean:
	@echo "Removing virtual environment '$(VENV_DIR)' or conda env '$(ENV_NAME)' (if exists)..."
	@if command -v conda > /dev/null 2>&1; then \
		conda remove -y --name $(ENV_NAME) --all || true; \
	fi
	-rm -rf $(VENV_DIR)

	@echo "Removing cloned repositories..."
	-rm -rf aspforaba
	-rm -rf ArgCausalDisco
	-rm -rf notears
