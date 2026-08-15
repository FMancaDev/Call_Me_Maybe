from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model
from src.decoder import (
    find_function, generate_number_parameters,
    select_function_name,
)
from src.loader import load_function_definitions
from src.prompt import build_function_selection_prompt


def test_request(model: Small_LLM_Model, user_prompt: str) -> None:
    """Test one function-calling request"""

    functions = load_function_definitions(
        Path(
            "data/input/functions_definition.json"
        )
    )

    selection_prompt = (
        build_function_selection_prompt(
            user_prompt,
            functions,
        )
    )

    function_name = select_function_name(
        model,
        selection_prompt,
        functions,
    )

    function = find_function(
        functions,
        function_name,
    )

    parameters = generate_number_parameters(
        model,
        user_prompt,
        function,
    )

    print(f"User:       {user_prompt}")
    print(f"Function:   {function_name}")
    print(f"Parameters: {parameters}")
    print()


def main() -> None:
    """Run numeric parameter tests"""

    print("Loading model...")

    model = Small_LLM_Model(
        device="cpu"
    )

    print("Model loaded.")
    print()

    test_request(
        model,
        "What is the sum of 2 and 3?",
    )

    test_request(
        model,
        "What is the sum of 265 and 345?",
    )

    test_request(
        model,
        "What is the square root of 144?",
    )


if __name__ == "__main__":
    main()
