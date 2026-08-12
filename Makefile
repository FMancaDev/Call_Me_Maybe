PYTHON = uv run python3
PYTEST = uv run pytest
FLAKE8 = uv run flake8

CACHE_DIR = $(PWD)/.uv-cache
HF_HOME = $(PWD)/.hf-cache
TRANSFORMERS_CACHE = $(PWD)/.hf-cache/transformers

ENV = UV_CACHE_DIR=$(CACHE_DIR) HF_HOME=$(HF_HOME) TRANSFORMERS_CACHE=$(TRANSFORMERS_CACHE)

.PHONY: help install test test-sdk test-token test-prompt test-selection lint run clean

help:
	@echo "Available commands:"
	@echo "  make install        Install project dependencies"
	@echo "  make test           Run all tests"
	@echo "  make test-sdk       Test the SDK/model"
	@echo "  make test-token     Test token generation"
	@echo "  make test-prompt    Test prompt generation"
	@echo "  make test-selection Test function selection"
	@echo "  make lint           Run flake8"
	@echo "  make run            Run the application"
	@echo "  make clean          Remove cache files"

install:
	uv sync
	uv add --dev pytest flake8

test:
	$(ENV) $(PYTEST) tests

test-sdk:
	$(ENV) $(PYTHON) -m tests.test_sdk

test-token:
	$(ENV) $(PYTHON) -m tests.test_token_generation

test-prompt:
	$(ENV) $(PYTHON) -m tests.test_prompt

test-selection:
	$(ENV) $(PYTHON) -m tests.test_function_selection

lint:
	$(ENV) $(FLAKE8) src tests --exclude=.uv-cache,.hf-cache,.venv

run:
	$(ENV) $(PYTHON) -m src

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
