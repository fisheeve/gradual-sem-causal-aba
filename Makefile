PYTHON_VERSION=3.12
VENV_DIR=.venv

ifneq (,$(wildcard .env))
    include .env
    export
endif

R_LIB_DIR?=$(HOME)/R/library

.PHONY: install clean

install: clean
	python$(PYTHON_VERSION) -m venv $(VENV_DIR); \
	echo "Activating venv and installing requirements..."; \
	. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt; \

	@echo "Cloning required repositories..."
	git clone git@bitbucket.org:coreo-group/aspforaba.git
	git clone git@github.com:briziorusso/ArgCausalDisco.git
	git clone https://github.com/xunzheng/notears.git

	@echo "[TODO: fix this somehow] Applying custom changes to ArgCausalDisco sub-repo..."
	cd ArgCausalDisco && git apply ../causalaba.diff

	@echo "[TODO: fix this somehow] Applying custom changes to aspforaba sub-repo..."
	cd aspforaba && git apply ../aspforaba.diff

	@echo "Installing additional dependencies..."
	R_INTERACTIVE=FALSE R_PAPERSIZE=letter PAGER=cat R_OPTS="--no-save --no-restore --quiet" \
	Rscript -e 'install.packages("BiocManager", repos="https://cloud.r-project.org", lib="$(R_LIB_DIR)")'

	R_INTERACTIVE=FALSE R_PAPERSIZE=letter PAGER=cat R_OPTS="--no-save --no-restore --quiet" \
	Rscript -e '.libPaths(c("$(R_LIB_DIR)", .libPaths())); BiocManager::install(c("SID", "bnlearn", "pcalg", "kpcalg", "glmnet", "mboost"))'

clean:
	@echo "Removing virtual environment directory '$(VENV_DIR)' (if it exists)..."
	-rm -rf $(VENV_DIR)

	@echo "Removing cloned repositories..."
	-rm -rf aspforaba
	-rm -rf ArgCausalDisco
	-rm -rf notears
