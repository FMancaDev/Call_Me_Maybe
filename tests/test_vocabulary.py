from llm_sdk.llm_sdk import Small_LLM_Model
from src.vocabulary import build_number_token_map, load_tokenizer_vocabulary


def main() -> None:
    """Inspect the tokenizer vocabulary"""

    print("Loading model...")

    model = Small_LLM_Model(device="cpu")

    print("Model loaded.")

    vocabulary = load_tokenizer_vocabulary(model)

    print(
        f"Vocabulary entries: {len(vocabulary)}"
    )

    number_tokens = build_number_token_map(model)

    print(
        f"Number candidate tokens: "
        f"{len(number_tokens)}"
    )

    print()
    print("Some number tokens:")

    shown = 0

    for token_id, text in number_tokens.items():
        print(
            f"{token_id:<8} -> {text!r}"
        )

        shown += 1

        if shown == 30:
            break


if __name__ == "__main__":
    main()
