ENV_NAME=aba-env
PYTHON_VERSION=3.12
VENV_DIR=.venv
R_LIB_DIR=~/R/library

.PHONY: install clean

install:
	@echo "Checking if Conda is installed..."
	@if command -v conda >/dev/null 2>&1; then \
		echo "Conda found. Creating Conda environment '$(ENV_NAME)' with Python $(PYTHON_VERSION)..."; \
		conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION); \
		echo "Activating Conda environment and installing requirements..."; \
		/bin/bash -c "source activate $(ENV_NAME) && pip install -r requirements.txt"; \
	else \
		echo "Conda not found. Creating Python venv in '$(VENV_DIR)'..."; \
		python$(PYTHON_VERSION) -m venv $(VENV_DIR); \
		echo "Activating venv and installing requirements..."; \
		. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt; \
	fi

	@echo "Cloning required repositories..."
	git clone git@bitbucket.org:lehtonen/ababaf.git
	git clone git@bitbucket.org:coreo-group/aspforaba.git
	git clone git@github.com:briziorusso/ArgCausalDisco.git
	git clone https://github.com/xunzheng/notears.git

	@echo "Installing additional dependencies..."
	Rscript -e 'install.packages("BiocManager", repos="https://cloud.r-project.org", lib="$(R_LIB_DIR)")'
	Rscript -e '.libPaths(c("$(R_LIB_DIR)", .libPaths())); BiocManager::install(c("SID", "bnlearn", "pcalg", "kpcalg", "glmnet", "mboost"))'

	@echo "[TODO: fix this somehow] Applying custom changes to ArgCausalDisco sub-repo..."
	cd ArgCausalDisco && git apply ../causalaba.diff

	@echo "[TODO: fix this somehow] Applying custom changes to aspforaba sub-repo..."
	cd aspforaba && git apply ../aspforaba.diff

clean:
	@echo "Removing Conda environment '$(ENV_NAME)' (if it exists)..."
	-conda remove -y --name $(ENV_NAME) --all

	@echo "Removing virtual environment directory '$(VENV_DIR)' (if it exists)..."
	-rm -rf $(VENV_DIR)
