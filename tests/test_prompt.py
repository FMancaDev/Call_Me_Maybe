from pathlib import Path
from src.loader import load_function_definitions
from src.prompt import build_function_calling_prompt, build_parameter_prompt


def main() -> None:
    """Test the prompt builders"""

    functions_path = Path("data/input/functions_definition.json")

    functions = load_function_definitions(functions_path)

    user_prompt = "What is the sum of 2 and 3?"

    print("=== FUNCTION CALLING PROMPT ===")
    print()

    calling_prompt = build_function_calling_prompt(
        user_prompt,
        functions,
    )

    print(calling_prompt)

    print()
    print("=" * 60)
    print()

    selected_function = None

    for function in functions:
        if function.name == "fn_add_numbers":
            selected_function = function
            break

    if selected_function is None:
        raise ValueError(
            "fn_add_numbers was not found"
        )

    print("=== PARAMETER EXTRACTION PROMPT ===")
    print()

    parameter_prompt = build_parameter_prompt(
        user_prompt,
        selected_function,
    )

    print(parameter_prompt)


if __name__ == "__main__":
    main()
