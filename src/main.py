from llm_sdk.llm_sdk import Small_LLM_Model
from .prompt_builder import build_prompt
from .parser import (
    load_function_calling_tests,
    load_function_definitions
)


def main():
    functions = load_function_definitions(
        "data/input/functions_definition.json"
    )
    tests = load_function_calling_tests(
        "data/input/function_calling_tests.json"
    )
    model = Small_LLM_Model()

    for test in tests:
        prompt = build_prompt(functions, test.prompt)
        input_ids = model.encode(prompt)
        input_ids = input_ids.tolist()[0]

        logits = model.get_logits_from_input_ids(input_ids)

        print(type(logits))
        print(len(logits))
        print(logits[:10])
