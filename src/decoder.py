import re
from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from src.prompt import build_parameter_value_prompt
from src.vocabulary import build_number_token_map


NUMBER_COMPLETE = re.compile(
    r"^-?(?:0|[1-9][0-9]*)"
    r"(?:\.[0-9]+)?"
    r"(?:[eE][+-]?[0-9]+)?$"
)

NUMBER_PREFIX = re.compile(
    r"^-?[0-9]*"
    r"(?:\.[0-9]*)?"
    r"(?:[eE][+-]?[0-9]*)?$"
)


def choose_allowed_token(logits: list[float], allowed_token_ids: set[int]) -> int:
    """Choose the highest-logit allowed token"""

    if not allowed_token_ids:
        raise ValueError(
            "No allowed tokens available"
        )

    best_token_id = -1
    best_logit = float("-inf")

    for token_id in allowed_token_ids:
        token_logit = logits[token_id]

        if token_logit > best_logit:
            best_logit = token_logit
            best_token_id = token_id

    if best_token_id == -1:
        raise ValueError(
            "Could not select an allowed token"
        )

    return best_token_id


def greedy_generate(model: Small_LLM_Model, prompt: str, max_tokens: int = 32) -> str:
    """Generate tokens without constraints"""

    encoded = model.encode(prompt)
    input_ids = encoded[0].tolist()

    generated_ids: list[int] = []

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            input_ids
        )

        best_token_id = 0
        best_logit = logits[0]

        for token_id in range(1, len(logits)):
            if logits[token_id] > best_logit:
                best_logit = logits[token_id]
                best_token_id = token_id

        input_ids.append(best_token_id)
        generated_ids.append(best_token_id)

    return model.decode(generated_ids)


def encode_function_names(model: Small_LLM_Model, functions: list[FunctionDefinition]) -> dict[str, list[int]]:
    """Encode available function names"""

    candidates: dict[str, list[int]] = {}

    for function in functions:
        text = " " + function.name + "\n"

        encoded = model.encode(text)

        candidates[function.name] = (
            encoded[0].tolist()
        )

    return candidates


def get_allowed_tokens(candidates: dict[str, list[int]], position: int) -> set[int]:
    """Get allowed tokens for function selection"""

    allowed: set[int] = set()

    for token_ids in candidates.values():
        if position < len(token_ids):
            allowed.add(
                token_ids[position]
            )

    return allowed


def filter_candidates(candidates: dict[str, list[int]], position: int, selected_token_id: int) -> dict[str, list[int]]:
    """Remove incompatible function candidates"""

    remaining: dict[str, list[int]] = {}

    for name, token_ids in candidates.items():
        if position >= len(token_ids):
            continue

        if token_ids[position] == selected_token_id:
            remaining[name] = token_ids

    return remaining


def find_completed_candidate(candidates: dict[str, list[int]], generated_length: int) -> str | None:
    """Return a completed function name if one exists"""

    for name, token_ids in candidates.items():
        if len(token_ids) == generated_length:
            return name

    return None


def select_function_name(model: Small_LLM_Model, prompt: str, functions: list[FunctionDefinition]) -> str:
    """Select a function using constrained decoding"""

    if not functions:
        raise ValueError(
            "No function definitions available"
        )

    encoded_prompt = model.encode(prompt)
    input_ids = encoded_prompt[0].tolist()

    candidates = encode_function_names(
        model,
        functions,
    )

    generated_ids: list[int] = []

    while candidates:
        position = len(generated_ids)

        allowed_token_ids = get_allowed_tokens(
            candidates,
            position,
        )

        if not allowed_token_ids:
            break

        logits = model.get_logits_from_input_ids(
            input_ids
        )

        next_token_id = choose_allowed_token(
            logits,
            allowed_token_ids,
        )

        generated_ids.append(next_token_id)
        input_ids.append(next_token_id)

        candidates = filter_candidates(
            candidates,
            position,
            next_token_id,
        )

        completed = find_completed_candidate(
            candidates,
            len(generated_ids),
        )

        if completed is not None:
            return completed

    raise ValueError(
        "Could not select a valid function"
    )


def is_valid_number_prefix(value: str) -> bool:
    """Check whether text can still become a valid number"""

    value = value.strip()

    if value == "":
        return True

    return NUMBER_PREFIX.fullmatch(
        value
    ) is not None


def is_complete_number(value: str) -> bool:
    """Check whether text is a complete JSON-compatible number"""

    value = value.strip()

    return NUMBER_COMPLETE.fullmatch(
        value
    ) is not None


def get_number_allowed_tokens(generated_text: str, number_tokens: dict[int, str]) -> set[int]:
    """Find tokens that keep the number prefix valid"""

    allowed: set[int] = set()

    for token_id, token_text in number_tokens.items():
        candidate = generated_text + token_text

        if is_valid_number_prefix(candidate):
            allowed.add(token_id)

    return allowed


def generate_number(model: Small_LLM_Model, prompt: str, number_tokens: dict[int, str], max_tokens: int = 8) -> float:
    """Generate one numeric value with constrained decoding

        model: Loaded language model.
        prompt: Prompt requesting one number.
        number_tokens: Tokens that can participate in numbers.
        max_tokens: Maximum generated tokens.
    """

    encoded_prompt = model.encode(prompt)
    input_ids = encoded_prompt[0].tolist()

    generated_text = ""

    newline_encoded = model.encode("\n")
    newline_ids = newline_encoded[0].tolist()

    if len(newline_ids) != 1:
        raise ValueError(
            "Newline is not represented by one token"
        )

    end_token_id = newline_ids[0]

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            input_ids
        )

        allowed = get_number_allowed_tokens(
            generated_text,
            number_tokens,
        )

        if is_complete_number(generated_text):
            allowed.add(end_token_id)

        next_token_id = choose_allowed_token(
            logits,
            allowed,
        )

        if next_token_id == end_token_id:
            value = generated_text.strip()

            return float(value)

        token_text = number_tokens.get(
            next_token_id
        )

        if token_text is None:
            raise ValueError(
                "Selected token is not numeric"
            )

        generated_text += token_text
        input_ids.append(next_token_id)

    if is_complete_number(generated_text):
        return float(
            generated_text.strip()
        )

    raise ValueError(
        "Could not generate a valid number"
    )


def find_function(functions: list[FunctionDefinition], function_name: str) -> FunctionDefinition:
    """Find a function definition by name"""

    for function in functions:
        if function.name == function_name:
            return function

    raise ValueError(
        f"Function not found: {function_name}"
    )


def generate_number_parameters(model: Small_LLM_Model, user_prompt: str, function: FunctionDefinition) -> dict[str, float]:
    """Generate all numeric parameters of a function"""

    number_tokens = build_number_token_map(
        model
    )

    result: dict[str, float] = {}

    for name, parameter in function.parameters.items():
        if parameter.type != "number":
            raise ValueError(
                f"Unsupported parameter type: "
                f"{parameter.type}"
            )

        prompt = build_parameter_value_prompt(
            user_prompt,
            function,
            name,
        )

        value = generate_number(
            model,
            prompt,
            number_tokens,
        )

        result[name] = value

    return result
