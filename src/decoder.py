from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition


def choose_allowed_token(logits: list[float], allowed_token_ids: set[int]) -> int:
    """
       Choose the allowed token with the highest logit

       logits: Logits for all vocabulary tokens
       allowed_token_ids: Token IDs allowed at the current step
    """

    if not allowed_token_ids:
        raise ValueError("No allowed tokens available")

    best_token_id = -1
    best_logit = float("-inf")

    for token_id in allowed_token_ids:
        token_logit = logits[token_id]

        if token_logit > best_logit:
            best_logit = token_logit
            best_token_id = token_id

    if best_token_id == -1:
        raise ValueError("Could not select an allowed token")

    return best_token_id


def greedy_generate(model: Small_LLM_Model, prompt: str, max_tokens: int = 32) -> str:
    """
       Generate text token by token with greedy decoding

       model: Loaded language model
       prompt: Input prompt
       max_tokens: Maximum number of generated tokens
    """

    encoded = model.encode(prompt)
    input_ids = encoded[0].tolist()

    generated_ids: list[int] = []

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

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
    """
       Encode all available function names

       model: Loaded language model
       functions: Available function definitions
    """

    candidates: dict[str, list[int]] = {}

    for function in functions:
        text = " " + function.name + "\n"
        encoded = model.encode(text)

        candidates[function.name] = encoded[0].tolist()

    return candidates


def get_allowed_tokens(candidates: dict[str, list[int]], position: int) -> set[int]:
    """
       Get valid token IDs for the current decoding position

       candidates: Remaining function-name candidates
       position: Current token position
    """

    allowed: set[int] = set()

    for token_ids in candidates.values():
        if position < len(token_ids):
            allowed.add(token_ids[position])

    return allowed


def filter_candidates(candidates: dict[str, list[int]], position: int, selected_token_id: int) -> dict[str, list[int]]:
    """
       Remove candidates incompatible with the selected token

       candidates: Current function-name candidates
       position: Current token position
       selected_token_id: Token selected by the model
    """

    remaining: dict[str, list[int]] = {}

    for name, token_ids in candidates.items():
        if position >= len(token_ids):
            continue

        if token_ids[position] == selected_token_id:
            remaining[name] = token_ids

    return remaining


def find_completed_candidate(candidates: dict[str, list[int]], generated_length: int) -> str | None:
    """
       Check whether a function name has been completely generated

       candidates: Remaining candidates
       generated_length: Number of generated tokens
    """

    for name, token_ids in candidates.items():
        if len(token_ids) == generated_length:
            return name

    return None


def select_function_name(model: Small_LLM_Model, prompt: str, functions: list[FunctionDefinition]) -> str:
    """
       Select a valid function name using constrained decoding

       model: Loaded language model.
       prompt: Function-selection prompt.
       functions: Available function definitions.
    """

    if not functions:
        raise ValueError("No function definitions available")

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

        logits = model.get_logits_from_input_ids(input_ids)

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

    raise ValueError("Could not select a valid function")
