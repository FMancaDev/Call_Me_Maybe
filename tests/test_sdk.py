from llm_sdk.llm_sdk import Small_LLM_Model


def main() -> None:
    """Test model loading, encoding, and decoding."""
    print("Loading model...")

    model = Small_LLM_Model(device="cpu")

    print("Model loaded.")

    text = "Hello world"

    input_ids = model.encode(text)

    print(f"Original text: {text}")
    print(f"Input IDs: {input_ids}")

    decoded = model.decode(input_ids[0])

    print(f"Decoded text: {decoded}")


if __name__ == "__main__":
    main()
