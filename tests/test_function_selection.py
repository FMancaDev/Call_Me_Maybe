from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model
from src.decoder import select_function_name
from src.loader import load_function_definitions
from src.prompt import build_function_selection_prompt


def test_prompt(model: Small_LLM_Model, user_prompt: str) -> None:
    """Test function selection for one user prompt"""

    functions = load_function_definitions(
        Path("data/input/functions_definition.json")
    )

    prompt = build_function_selection_prompt(
        user_prompt,
        functions,
    )

    selected = select_function_name(
        model,
        prompt,
        functions,
    )

    print(f"User:     {user_prompt}")
    print(f"Selected: {selected}")
    print()


def main() -> None:
    """Run function-selection tests"""

    print("Loading model...")

    model = Small_LLM_Model(device="cpu")

    print("Model loaded.")
    print()

    test_prompt(
        model,
        "What is the sum of 2 and 3?",
    )

    test_prompt(
        model,
        "Greet Shrek",
    )

    test_prompt(
        model,
        "Reverse the string hello",
    )

    test_prompt(
        model,
        "What is the square root of 25?",
    )


if __name__ == "__main__":
    main()
