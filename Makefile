.PHONY: install test clean run help

VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
ACTIVATE = source $(VENV_DIR)/bin/activate

# Dependencias
REQUIREMENTS = torch transformers numpy

help:
	@echo "Comandos disponiveis:"
	@echo "  make install   - Cria virtual environment e instala dependencias"
	@echo "  make test      - Executa testes"
	@echo "  make run       - Executa o projeto principal"
	@echo "  make clean     - Remove virtual environment"
	@echo "  make help      - Mostra esta mensagem"

install:
	@echo "Criando virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Instalando dependencias..."
	$(PIP) install --upgrade pip
	$(PIP) install $(REQUIREMENTS)
	@echo "Instalacao completa!"
	@echo "Para ativar o ambiente: source $(VENV_DIR)/bin/activate"

test:
	@echo "Executando testes..."
	$(PYTHON) test_sdk.py
	$(PYTHON) investigate_vocab.py
	$(PYTHON) test_token_generation.py

run:
	@echo "Executando projeto principal..."
	$(PYTHON) src/main.py

clean:
	@echo "Removendo virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Limpeza completa!"

dev:
	@echo "Ativando ambiente de desenvolvimento..."
	$(ACTIVATE)
