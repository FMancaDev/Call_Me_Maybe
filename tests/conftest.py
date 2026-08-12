import pytest
from llm_sdk.llm_sdk import Small_LLM_Model


@pytest.fixture(scope="session")
def model() -> Small_LLM_Model:
    print("\nLoading model...")
    model = Small_LLM_Model()
    print("Model loaded")
    return model


@pytest.fixture
def user_prompt() -> str:
    return "What is the sum of 2 and 3?"
