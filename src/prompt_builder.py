from .models import FunctionDefinition


def build_prompt(functions: list[FunctionDefinition], user_prompt: str) -> str:
    prompt = ""
    prompt = "Available functions: \n\n"

    for function in functions:
        prompt += f"Function: {function.name}\n"
        prompt += f"Description: {function.description}\n"

        for parameter_name, parameter in function.parameters.items():
            prompt += (
                f"Parameter: {parameter_name} ({parameter.type})\n"
            )
        prompt += f"Returns: {function.returns.type}\n\n"
    return prompt
