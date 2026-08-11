PYTHON = python3
UV = uv

UV_CACHE_DIR = $(PWD)/.uv-cache
HF_HOME = $(PWD)/.hf-cache
TRANSFORMERS_CACHE = $(HF_HOME)/transformers

ENV = UV_CACHE_DIR=$(UV_CACHE_DIR) \
	HF_HOME=$(HF_HOME) \
	TRANSFORMERS_CACHE=$(TRANSFORMERS_CACHE)

.PHONY: install run debug \
	test test-sdk test-token test-prompt \
	test-selection test-vocabulary test-parameters \
	lint lint-strict clean fclean re

install:
	mkdir -p $(UV_CACHE_DIR)
	mkdir -p $(HF_HOME)
	$(ENV) $(UV) sync

run:
	$(ENV) $(UV) run $(PYTHON) -m src

debug:
	$(ENV) $(UV) run $(PYTHON) -m pdb -m src

test:
	$(ENV) $(UV) run $(PYTHON) -m pytest tests

test-sdk:
	$(ENV) $(UV) run $(PYTHON) -m tests.test_sdk

test-token:
	$(ENV) $(UV) run $(PYTHON) -m tests.test_token_generation

test-prompt:
	$(ENV) $(UV) run $(PYTHON) -m tests.test_prompt

test-selection:
	$(ENV) $(UV) run $(PYTHON) -m tests.test_function_selection

test-vocabulary:
	$(ENV) $(UV) run $(PYTHON) -m tests.test_vocabulary

test-parameters:
	$(ENV) $(UV) run $(PYTHON) -m tests.test_parameter_generation

lint:
	$(ENV) $(UV) run flake8 .
	$(ENV) $(UV) run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(ENV) $(UV) run flake8 .
	$(ENV) $(UV) run mypy . --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

fclean: clean
	rm -rf .venv
	rm -rf .uv-cache
	rm -rf .hf-cache

re: fclean install
