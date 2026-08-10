from pathlib import Path
from src.loader import load_function_definition
from src.prompt import build_function_calling_prompt


def main() -> None:
    """Print a function-calling prompt"""

    functions = load_function_definition(
        Path("data/input/functions_definition.json")
    )

    prompt = build_function_calling_prompt(
        "What is the sum of 2 and 3?",
        functions
    )

    print(prompt)


if __name__ == "__main__":
    main()
