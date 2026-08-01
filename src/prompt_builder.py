from .models import FunctionDefinition


def build_prompt(
    functions: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    prompt = (
        "You are a function calling assistant.\n"
        "Your task is to map a user request to exactly one function.\n"
        "Choose the function whose PURPOSE best matches the request.\n"
        "After choosing the function, extract all required parameters.\n"
        "Return only a valid JSON object.\n\n"
    )

    prompt += "Available functions:\n\n"

    for function in functions:
        prompt += f"Function: {function.name}\n"
        prompt += f"Purpose: {function.description}\n"

        if function.parameters:
            prompt += "Parameter names: "
            prompt += ", ".join(function.parameters.keys())
            prompt += "\n"

            prompt += "Arguments:\n"
            for name, parameter in function.parameters.items():
                prompt += f"- {name}: {parameter.type}\n"
        else:
            prompt += "Arguments: none\n"

        prompt += f"Returns: {function.returns.type}\n"
        prompt += "\n"

    prompt += (
        "Examples:\n\n"

        "User: Reverse the string 'example'\n"
        "Output:\n"
        '{ "name": "fn_reverse_string", '
        '"parameters": { "s": "example" } }\n\n'

        "User: Greet Alice\n"
        "Output:\n"
        '{ "name": "fn_greet", '
        '"parameters": { "name": "Alice" } }\n\n'

        'User: Replace all numbers in \"Age: 18, Score: 42\" with NUMBERS\n'
        "Output:\n"
        "{\n"
        '  "name": "fn_substitute_string_with_regex",\n'
        '  "parameters": {\n'
        '    "source_string": "Age: 18, Score: 42",\n'
        '    "regex": "\\\\d",\n'
        '    "replacement": "NUMBERS"\n'
        "  }\n"
        "}\n\n"
    )
    prompt += "User request:\n"
    prompt += user_prompt

    return prompt
