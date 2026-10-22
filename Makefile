# Makefile — Developer & Researcher Automation Tasks
.PHONY: help install install-dev test run docker-build docker-run k8s-apply clean

PYTHON ?= python3

help:
	@echo "======================================================================="
	@echo "Predictive Hardware-Aware Digital Twin Benchmark Automation"
	@echo "======================================================================="
	@echo "Available targets:"
	@echo "  make install       Install production dependencies from requirements.txt"
	@echo "  make install-dev   Install package in editable mode with development tools"
	@echo "  make test          Run full scientific verification unittest suite"
	@echo "  make run           Execute closed-loop digital twin benchmark simulation"
	@echo "  make docker-build  Build reproducible Docker container image"
	@echo "  make docker-run    Run benchmark inside isolated Docker container"
	@echo "  make k8s-apply     Deploy simulation job to local Kubernetes cluster"
	@echo "  make clean         Remove compiled Python artifacts and temporary caches"
	@echo "======================================================================="

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m unittest discover -s tests -v

run:
	$(PYTHON) main.py

docker-build:
	docker compose build

docker-run:
	docker compose up

k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/job-simulation.yaml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info .eggs
