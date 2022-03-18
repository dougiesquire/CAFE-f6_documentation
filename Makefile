.PHONY: environment data docs clean lint

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = Squire_2022_CAFE-f6
ENV_NAME = forecast_analysis
NCI_PROJECT = xv83
RAW_DATA_DIR = ./data/raw/
TEST_DATA_DIR = ./data/testing/

data_config = CAFE60v1.yml CAFEf5.yml CAFEf6.yml CAFE_hist.yml CanESM5_hist.yml CanESM5.yml EC_Earth3.yml EC_Earth3_hist.yml EN422.yml GPCP.yml HadISST.yml JRA55.yml
skill_config = CAFEf6_sst.yml CanESM5_sst.yml EC_Earth3_sst.yml CAFEf6_t_ref.yml CanESM5_t_ref.yml EC_Earth3_t_ref.yml CAFEf6_precip.yml CanESM5_precip.yml EC_Earth3_precip.yml CAFEf6_amv.yml CanESM5_amv.yml EC_Earth3_amv.yml CAFEf6_ipo.yml CanESM5_ipo.yml EC_Earth3_ipo.yml CAFEf6_ohc300.yml

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
CONDA_DIR=$(shell conda info --base)
ENV_DIR=$(CONDA_DIR)/envs/$(ENV_NAME)
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
endif

ifeq ($(MAKECMDGOALS),data)
config=$(data_config)
else ifeq ($(MAKECMDGOALS),skill)
config=$(skill_config)
endif

define HEADER
#!/bin/bash -l
#PBS -P $(NCI_PROJECT)
#PBS -q express
#PBS -l walltime=04:00:00
#PBS -l mem=192gb
#PBS -l ncpus=48
#PBS -l jobfs=100GB
#PBS -l wd
#PBS -l storage=gdata/xv83+gdata/oi10+gdata/ua8
#PBS -j oe

conda activate $(ENV_NAME)
endef

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create the python environment or update it if it exists
environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifneq ("$(wildcard $(ENV_DIR))","") # check if the directory is there
	@echo ">>> Project environment already exists, updating'"
	conda env update --name $(ENV_NAME) --file environment.yml --prune
else
	conda env create -f environment.yml
endif
else
	@echo ">>> This project uses conda to install the environment. Please install conda"
endif

## Prepare datasets for analysis
data:
	mkdir -p $(TEST_DATA_DIR)/reference_exectuable
	mkdir -p $(RAW_DATA_DIR)/gridinfo
	ln -sfn /home/599/ds0092/src/mom_cafe/exec/gadi.nci.org.au/CM2M/fms_CM2M.x $(TEST_DATA_DIR)/reference_exectuable/fms_CM2M.x
	ln -sfn /g/data/xv83/dcfp/CAFE60v1/ $(RAW_DATA_DIR)/CAFE60v1
	ln -sfn /g/data/xv83/users/ds0092/data/CAFE/historical/WIP/c5-d60-pX-ctrl-19601101/ZARR/ $(RAW_DATA_DIR)/CAFE_ctrl
	ln -sfn /g/data/xv83/dcfp/CAFE-f5/ $(RAW_DATA_DIR)/CAFEf5
	ln -sfn /g/data/xv83/dcfp/CAFE-f6/ $(RAW_DATA_DIR)/CAFEf6
	ln -sfn /g/data/xv83/users/ds0092/data/CAFE/historical/WIP/c5-d60-pX-hist-19601101/ZARR/ $(RAW_DATA_DIR)/CAFE_hist
	ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/CCCma/CanESM5/dcppA-hindcast/ $(RAW_DATA_DIR)/CanESM5
	ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/CCCma/CanESM5/historical/ $(RAW_DATA_DIR)/CanESM5_hist
	ln -sfn /g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppA-hindcast $(RAW_DATA_DIR)/EC-Earth3
	ln -sfn /g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical $(RAW_DATA_DIR)/EC-Earth3_hist
	ln -sfn /g/data/xv83/reanalyses/EN.4.2.2/ $(RAW_DATA_DIR)/EN.4.2.2
	ln -sfn /g/data/xv83/reanalyses/HadISST/ $(RAW_DATA_DIR)/HadISST
	ln -sfn /g/data/xv83/reanalyses/JRA55/ $(RAW_DATA_DIR)/JRA55	
	ln -sfn /g/data/ua8/Precipitation/GPCP/mon/v2-3/ $(RAW_DATA_DIR)/GPCP
	ln -sfn /g/data/xv83/users/ds0092/active_projects/Squire_2022_CAFE-f6/data/raw/EC-Earth3_hist/r1i1p1f1/Ofx/areacello/gn/v20200918/areacello_Ofx_EC-Earth3_historical_r1i1p1f1_gn.nc $(RAW_DATA_DIR)/gridinfo/EC-Earth3_ocean_are.nc
	ln -sfn /g/data/xv83/users/ds0092/active_projects/Squire_2022_CAFE-f6/data/raw/EC-Earth3_hist/r1i1p1f1/fx/areacella/gr/v20210324/areacella_fx_EC-Earth3_historical_r1i1p1f1_gr.nc $(RAW_DATA_DIR)/gridinfo/EC-Earth3_atmos_area.nc
	ln -sfn /g/data/xv83/users/ds0092/active_projects/Squire_2022_CAFE-f6/data/raw/CanESM5_hist/r1i1p2f1/Ofx/areacello/gn/v20190429/areacello_Ofx_CanESM5_historical_r1i1p2f1_gn.nc $(RAW_DATA_DIR)/gridinfo/CanESM5_ocean_area.nc
	ln -sfn /g/data/xv83/users/ds0092/active_projects/Squire_2022_CAFE-f6/data/raw/CanESM5_hist/r1i1p2f1/fx/areacella/gn/v20190429/areacella_fx_CanESM5_historical_r1i1p2f1_gn.nc $(RAW_DATA_DIR)/gridinfo/CanESM5_atmos_area.nc
	$(foreach c,$(config),$(file >data_$(c),$(HEADER)) $(file >>data_$(c),python src/prepare_data.py $(c)))
	for c in $(config); do qsub data_$${c}; rm data_$${c}; done

## Prepare datasets for analysis
skill:
	$(foreach c,$(config),$(file >skill_$(c),$(HEADER)) $(file >>skill_$(c),python src/verify.py $(c)))
	for c in $(config); do qsub skill_$${c}; rm skill_$${c}; done

## Build the documentation
docs:
	cd docs && $(MAKE) clean && $(MAKE) html

## Delete unneeded Python files, dask-worker files and PBS output files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.ipynb_checkpoints" -exec rm -rf {} +
	find . -type d -name "dask-worker-space" -exec rm -rf {} +
	find . -type f -name "*.o????????" -delete

## Lint using black and flake8
lint:
	($(CONDA_ACTIVATE) $(ENV_NAME) ; black src ; flake8 src)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
