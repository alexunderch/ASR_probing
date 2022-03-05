SHELL = /bin/bash

VENV_PATH = venv


help:
	@echo "make test: Run basic tests (not testing most integrations)"
	@echo "make build: Does full and slow testing procedure on latest python versions"

venv:
	virtualenv -p python3 $(VENV_PATH)
	$(VENV_PATH)/bin/pip install -r requirements.txt

build: venv test
`.PHONY: test

test: venv
	@$(VENV_PATH)/bin/python -m pytest --cov-report html --cov-report term --cov=df_engine tests/
.PHONY: test

build_docs:
