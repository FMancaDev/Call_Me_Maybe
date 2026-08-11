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


def build_parameter_prompt(user_prompt: str, function: FunctionDefinition) -> str:
    """Build a prompt for parameter extraction"""

    parameter_data = {}

    for name, parameter in function.parameters.items():
        parameter_data[name] = parameter.model_dump()

        parameters_json = json.dumps(
            parameter_data,
            ensure_ascii=False,
            indent=2
        )

    prompt = (
        "Extract the parameters required by the function.\n"
        "Use only information from the user request.\n"
        "Return only the parameter values as JSON.\n\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters:\n{parameters_json}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "Parameters JSON:"
    )
    return prompt


def build_parameter_value_prompt(user_prompt: str, function: FunctionDefinition, parameter_name: str) -> str:
    """Build a prompt for extracting one parameter value

        user_prompt: Original user request.
        function: Selected function.
        parameter_name: Parameter to extract.
    """

    parameter = function.parameters.get(parameter_name)

    if parameter is None:
        raise ValueError(
            f"Unknown parameter: {parameter_name}"
        )

    prompt = (
        "Extract exactly one value from the user request.\n"
        "Do not calculate or execute the function.\n"
        "Return only the value, with no explanation.\n\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameter: {parameter_name}\n"
        f"Parameter type: {parameter.type}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        f"Value for {parameter_name}:"
    )

    return prompt
