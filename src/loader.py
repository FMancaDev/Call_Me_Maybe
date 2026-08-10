import json
from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from src.models import FunctionDefinition, PromptInput


T = TypeVar("T", bound=BaseModel)


def load_json_file(path: Path) -> object:
    """
       Load JSON data from a file
       path: Path to the JSON file
    """

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {path}") from exc

    except PermissionError as exc:
        raise ValueError(f"Permission denied: {path}") from exc

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}: "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc

    except OSError as exc:
        raise ValueError(
            f"Could not read file {path}: {exc}"
        ) from exc


def validate_list(data: object, model: type[T], source: Path) -> list[T]:
    """
       Validate every item of a JSON array
       data: Parsed JSON data
       model: Pydantic model used for validation
       source: Path of the source file
    """

    if not isinstance(data, list):
        raise ValueError(
            f"{source} must contain a JSON array"
        )

    validated_items: list[T] = []

    try:
        for item in data:
            validated = model.model_validate(item)
            validated_items.append(validated)

    except ValidationError as exc:
        raise ValueError(
            f"Invalid data in {source}: {exc}"
        ) from exc

    return validated_items


def load_prompts(path: Path) -> list[PromptInput]:
    """Load and validate prompt inputs"""

    data = load_json_file(path)

    return validate_list(
        data,
        PromptInput,
        path,
    )


def load_function_definitions(path: Path,) -> list[FunctionDefinition]:
    """Load and validate available function definitions"""

    data = load_json_file(path)

    return validate_list(
        data,
        FunctionDefinition,
        path,
    )
