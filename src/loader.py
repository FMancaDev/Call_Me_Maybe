import json
from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from src.models import FunctionDefinition, PromptInput


T = TypeVar("T", bound=BaseModel)


def load_json_file(path: Path) -> object:
    """Load JSON data from a file,
       Args: path: path to the json file,
       Return: Parsed Json data.
       Raises: valueerros: if the file cannot be read or contains invalid json
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
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"could not read file {path}: {exc}") from exc


def validate_list(data: object, model: type[T], source: Path) -> list[T]:
    """Validate a JSON array using Pydantic model,
       Args: data: raw json data,
             model: pydantic model array used for each array,
             source: file path used for error messages
       returns: list of validated pydantic object
    """

    if not isinstance(data, list):
        raise ValueError(f"{source} must contain a JSON array")

    try:
        return [model.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(
            f"Invalid data in {source}: {exc}"
        ) from exc


def load_prompts(path: Path) -> list[PromptInput]:
    """load and validate prompt inputs"""

    data = load_json_file(path)
    return validate_list(data, PromptInput, path)


def load_function_definition(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions"""

    data = load_json_file(path)
    return validate_list(data, FunctionDefinition, path)
