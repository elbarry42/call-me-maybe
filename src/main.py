import json
from llm_sdk.llm_sdk import Small_LLM_Model
from .decoder import Decoder
from .cli import parse_arguments, write_output
from .prompt_builder import build_prompt
from .parser import (load_function_calling_tests, load_function_definitions)


def main() -> None:

    args = parse_arguments()

    functions = load_function_definitions(args.functions_definition)
    if not functions:
        return

    tests = load_function_calling_tests(args.input)
    if not tests:
        return

    model = Small_LLM_Model()
    decoder = Decoder(model)

    results = []

    for test in tests:
        try:
            print("=" * 50)
            print(test.prompt)

            prompt = build_prompt(
                functions,
                test.prompt,
            )

            input_ids = model.encode(prompt).tolist()[0]

            decoded = decoder.decode(
                input_ids,
                functions,
                test.prompt,
            )

            print(json.dumps(decoded, indent=4))

            results.append(
                {
                    "prompt": test.prompt,
                    "name": decoded["name"],
                    "parameters": decoded["parameters"],
                }
            )

        except Exception as error:
            print(f"Error while processing test: {error}")
            continue

        try:
            write_output(args.output, results)
        except (OSError, ValueError, TypeError) as error:
            print(f"Error while writing output file: {error}")
