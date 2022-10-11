all: summary figures tables
.PHONY: venv
SHELL := /bin/bash
PROJBASE := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

##########################################################
####################      VENV     #######################
##########################################################

venv:
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip
	source venv/bin/activate && python -m pip install torch==1.8.1+cu111 \
		-f https://download.pytorch.org/whl/torch_stable.html
	source venv/bin/activate && python -m pip install cupy-cuda111==10.0.0
	source venv/bin/activate && python -m pip install -r requirements.txt
