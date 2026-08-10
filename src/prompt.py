import json
from src.models import FunctionDefinition


def build_function_calling_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the prompt used for function-call generation

       user_promt: Natural-language request from the user
       function: Function available to the model
    """

    function_data = []

    for function in functions:
        function_data.append(function.model_dump())

    functions_json = json.dumps(
        function_data,
        ensure_ascii=False,
        indent=2,
    )

    prompt = (
        "You are a function-calling assistant\n"
        "Select the function that best matches the user request\n"
        "Extract the required parameters from the request\n"
        "Do not execute the function\n"
        "Return only a JSON object with this structure:\n"
        '{"name": "function_name", "parameters": {...}}\n\n'
        "Available functions:\n"
        f"{functions_json}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "JSON:\n"
    )

    return prompt


def build_function_selection_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build a prompt for function-name selection"""

    descriptions = ""

    for function in functions:
        descriptions += (
            f"- {function.name}: "
            f"{function.description}\n"
        )

    prompt = (
        "Choose the function that best matches the user request\n"
        "Do not execute the function\n"
        "Return only the function name\n\n"
        "Available functions:\n"
        f"{descriptions}\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "Function name:"
    )

    return prompt
