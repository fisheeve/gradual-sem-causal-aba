PYTHON_VERSION=3.12
ENV_NAME=aba-env
VENV_DIR=.venv

# Always remove and recreate .env
$(shell rm -f .env; \
	read -p "RPATH is not set. Enter Rscript path (e.g. /usr/bin/Rscript): " rpath; \
	echo "RPATH=$$rpath" >> .env)

include .env
export

.PHONY: install clean

install: clean
	@echo "Setting up Python environment..."
	@if command -v conda > /dev/null 2>&1; then \
		echo "Conda found. Creating and activating conda environment..."; \
		conda create -y -n $(ENV_NAME) -c conda-forge python=$(PYTHON_VERSION) r-base r-biocmanager r-sid r-bnlearn r-pcalg r-kpcalg r-glmnet r-mboost; \
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

	@echo "Installing additional dependencies..."
	Rscript -e 'BiocManager::install("SID")'

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
