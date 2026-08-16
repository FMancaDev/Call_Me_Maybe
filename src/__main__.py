import argparse
import json
import sys
from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model
from src.decoder import (
    select_function_name,
    generate_parameters,
)
from src.loader import load_function_definitions, load_prompts
from src.prompt import build_function_selection_prompt


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
        help="Path to the output JSON file",
    )

    return parser.parse_args()


def main() -> int:
    """Run the application"""

    args = parse_args()

    try:
        prompts = load_prompts(args.input)
        functions = load_function_definitions(
            args.function_definition
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(prompts)} prompts")
    print(f"Loaded {len(functions)} functions")

    print("\nLoading model...")
    model = Small_LLM_Model()
    print("Model loaded")

    results = []

    for prompt in prompts:
        print()
        print(f"User: {prompt.prompt}")

        try:
            selection_prompt = build_function_selection_prompt(
                prompt.prompt,
                functions
            )

            function_name = select_function_name(
                model,
                selection_prompt,
                functions,
            )

            function = None

            for item in functions:
                if item.name == function_name:
                    function = item
                    break

            if function is None:
                raise ValueError(
                    f"Function not found: {function_name}"
                )

            parameters = generate_parameters(
                model,
                prompt.prompt,
                function,
            )

            result = {
                "prompt": prompt.prompt,
                "name": function.name,
                "parameters": parameters,
            }

            results.append(result)

            print(f"Function: {function.name}")
            print(
                "Parameters: "
                f"{json.dumps(parameters)}"
            )

        except (ValueError, TypeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    try:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=2,
                ensure_ascii=False,
            )

    except OSError as exc:
        print(
            f"Error writing output: {exc}",
            file=sys.stderr,
        )
        return 1

    print()
    print(f"Results written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
