import json
from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model


NUMBER_CHARACTERS = set("0123456789-+.eE ")


def load_tokenizer_vocabulary(model: Small_LLM_Model) -> dict[str, int]:
    """Load the tokenizer vocabulary"""

    tokenizer_path = Path(model.get_path_to_tokenizer_file())

    try:
        with tokenizer_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not load tokenizer: {exc}"
        ) from exc

    model_data = data.get("model")

    if not isinstance(model_data, dict):
        raise ValueError(
            "Tokenizer does not contain model data"
        )

    vocabulary = model_data.get("vocab")

    if not isinstance(vocabulary, dict):
        raise ValueError(
            "Tokenizer does not contain a vocabulary"
        )

    result: dict[str, int] = {}

    for token, token_id in vocabulary.items():
        if not isinstance(token, str):
            continue

        if not isinstance(token_id, int):
            continue

        result[token] = token_id

    return result


def looks_like_number_token(raw_token: str) -> bool:
    """Check whether a raw tokenizer token may represent a number"""

    cleaned = raw_token

    cleaned = cleaned.replace("Ġ", "")
    cleaned = cleaned.replace("▁", "")

    if not cleaned:
        return False

    for character in cleaned:
        if character not in NUMBER_CHARACTERS:
            return False

    return True


def build_number_token_map(model: Small_LLM_Model) -> dict[int, str]:
    """Build a map of token IDs usable while generating numbers"""

    vocabulary = load_tokenizer_vocabulary(model)

    number_tokens: dict[int, str] = {}

    for raw_token, token_id in vocabulary.items():
        if not looks_like_number_token(raw_token):
            continue

        decoded = model.decode([token_id])

        if not decoded:
            continue

        stripped = decoded.strip()

        if not stripped:
            continue

        valid = True

        for character in stripped:
            if character not in "0123456789-+.eE":
                valid = False
                break

        if valid:
            number_tokens[token_id] = decoded

    return number_tokens
