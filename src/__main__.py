import argparse
import sys
from pathlib import Path
from src.loader import load_function_definition, load_prompts


DEFAULT_FUNCTION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Translate natural language into function calls"
    )

    parser.add_argument(
        "--function_definition",
        type=Path,
        default=DEFAULT_FUNCTION,
        help="Path to the function definitions JSON file",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the prompts JSON file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output JSON file"
    )

    return parser.parse_args()


def main() -> int:
    """Run the application"""

    args = parse_args()

    try:
        prompts = load_prompts(args.input)
        functions = load_function_definition(
            args.function_definition
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(prompts)} prompts")
    print(f"Loaded {len(functions)} functions")

    for function in functions:
        print(
            f"- {function.name}: "
            f"{function.description}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
