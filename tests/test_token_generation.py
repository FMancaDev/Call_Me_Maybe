import time
from llm_sdk.llm_sdk import Small_LLM_Model


def main() -> None:
    """Generate one token manually from the model logits"""

    print("Loading Model...")

    model = Small_LLM_Model(device="cpu")
    print("Model Loaded")

    prompt = "The capital of France is"
    encoded = model.encode(prompt)

    # encode returns a tensor with shape [1, sequence_lenght]
    inputs_id = encoded[0].tolist()

    print(f"Prompt: {prompt}")
    print(f"Input IDs: {inputs_id}")

    # get the logits for every possible next token
    logits = model.get_logits_from_input_ids(inputs_id)
    print(f"Vocabulary size: {len(logits)}")

    # greedy decoding: choode the toke with the largest logit
    next_token_id = max(
        range(len(logits)),
        key=logits.__getitem__,
    )

    next_token = model.decode([next_token_id])
    print(f"Next token ID: {next_token_id}")
    print(f"Next token: {next_token!r}")


if __name__ == "__main__":
    main()
