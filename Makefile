.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

###########################
# VARIABLES
###########################
PROJECTNAME := core
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)
# docker
DOCKER_CMD := docker run -v $$PWD:/workspace/ --gpus='all' --shm-size 8G -it $(PROJECTNAME):$(GIT_BRANCH)

###########################
# COMMANDS
###########################
# Thanks to: https://stackoverflow.com/a/10858332
# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

###########################
# PROJECT UTILS
###########################
.PHONY: clean
clean:  ##@Utils clean the project
	@black .
	@isort .
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store
	@rm -f .coverage coverage.xml report.xml
	@rm -f -R .pytest_cache
	@rm -f -R .idea
	@rm -f -R tmp/
	@rm -f -R cov_html/

.PHONY: install
install:  ##@Utils install the dependencies for the project
	python3 -m pip install -r requirements.txt

###########################
# DOCKER
###########################
_build:
	@echo "Build image $(GIT_BRANCH)..."
	@docker build -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker run an interactive bash inside the docker image
	@echo "Run inside docker image"
	$(DOCKER_CMD) /bin/bash

###########################
# TESTS
###########################
.PHONY: test
test: _build  ##@Test run all tests in the project
    # Ignore integration tests flag: --ignore=test/manual_integration_tests/
	$(DOCKER_CMD) /bin/bash -c "wandb offline && python -m pytest --cov-report html:cov_html --cov-report term --cov=src --cov-report xml --junitxml=report.xml tests && coverage xml"
