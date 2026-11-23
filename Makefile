PYTHON ?= .venv/bin/python
HOST_TESTS := knn gmm mfcc ns commands audio persistence

.PHONY: test test-host test-python evaluate notebooks collect

test: test-host test-python

test-host:
	@set -e; for suite in $(HOST_TESTS); do $(MAKE) -C tests/host/$$suite run; done

test-python:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests/python -v

evaluate:
	PYTHONPATH=src $(PYTHON) -m recvoz.evaluation --audio-dir audio_exemplo --output docs/results/session-holdout.json

notebooks:
	$(PYTHON) -m jupyterlab notebooks

collect:
	$(PYTHON) tools/coletor.py
