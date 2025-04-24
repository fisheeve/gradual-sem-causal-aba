# Makefile

ENV_NAME=aba-env
PYTHON_VERSION=3.12

.PHONY: install clean wow

install:
	@echo "Creating Conda environment '$(ENV_NAME)' with Python $(PYTHON_VERSION)..."
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

	@echo "Activating environment and installing requirements..."
	/bin/bash -c "source activate $(ENV_NAME) && pip install -r requirements.txt"

	@echo "Cloning required repositories..."
	git clone git@bitbucket.org:lehtonen/ababaf.git
	git clone git@bitbucket.org:coreo-group/aspforaba.git
	git clone git@github.com:briziorusso/ArgCausalDisco.git
	git clone https://github.com/xunzheng/notears.git

	@echo "Installing additional dependencies..."
	Rscript -e 'install.packages("BiocManager", repos="https://cloud.r-project.org")'
	Rscript -e 'BiocManager::install(c("SID", "bnlearn", "pcalg", "kpcalg", "glmnet", "mboost"))'

	@echo "[TODO: fix this somehow] Applying custom changes to cloned repos..."
	cd ArgCausalDisco && git apply ../causalaba.diff

clean:
	@echo "Removing Conda environment '$(ENV_NAME)'..."
	conda remove -y --name $(ENV_NAME) --all
