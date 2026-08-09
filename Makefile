PYTHON = python3
UV = uv

UV_CACHE_DIR = $(PWD)/.uv-cache
HF_HOME = $(PWD)/.hf-cache
TRANSFORMERS_CACHE = $(HF_HOME)/transformers

ENV = UV_CACHE_DIR=$(UV_CACHE_DIR) \
	HF_HOME=$(HF_HOME) \
	TRANSFORMERS_CACHE=$(TRANSFORMERS_CACHE)

.PHONY: install run test test-sdk test-token lint clean fclean re

install:
	mkdir -p $(UV_CACHE_DIR)
	mkdir -p $(HF_HOME)
	$(ENV) $(UV) sync

run:
	$(ENV) $(UV) run $(PYTHON) -m src

test:
	$(ENV) $(UV) run $(PYTHON) -m pytest tests

test-sdk:
	$(ENV) $(UV) run $(PYTHON) -m tests.test_sdk

test-token:
	$(ENV) $(UV) run $(PYTHON) -m tests.test_token_generation

lint:
	$(ENV) $(UV) run flake8 src tests
	$(ENV) $(UV) run mypy src tests \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache

fclean: clean
	rm -rf .venv
	rm -rf .uv-cache
	rm -rf .hf-cache

re: fclean install
