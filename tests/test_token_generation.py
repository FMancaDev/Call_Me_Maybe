import time
from llm_sdk.llm_sdk import Small_LLM_Model
from src.decoder import choose_allowed_token, greedy_generate


def test_choose_allowed_token() -> None:
    """Test selection of the best token among allowed tokens"""

    logits = [1.0, 9.0, 5.0, 7.0]
    allowed_token_ids = {0, 2, 3}

    result = choose_allowed_token(
        logits,
        allowed_token_ids,
    )

    assert result == 3


def main() -> None:
    """Test greedy token generation with the real LLM"""

    print("Testing constrained token selection...")

    test_choose_allowed_token()

    print("Constrained token selection OK")
    print("Loading model...")

    model = Small_LLM_Model(device="cpu")

    print("Model loaded.")
    prompt = "The capital of France is"

    start = time.perf_counter()

    result = greedy_generate(
        model=model,
        prompt=prompt,
        max_tokens=10,
    )

    elapsed = time.perf_counter() - start

    print(f"Prompt: {prompt}")
    print(f"Generated: {result!r}")
    print(f"Generation time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
