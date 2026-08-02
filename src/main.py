import os
import json
from llm_sdk.llm_sdk import Small_LLM_Model
from .decoder import Decoder
from .prompt_builder import build_prompt
from .parser import (load_function_calling_tests, load_function_definitions)


def main() -> None:
    try:
        functions = load_function_definitions(
            "data/input/functions_definition.json"
        )

        if not functions:
            return

        tests = load_function_calling_tests(
            "data/input/function_calling_tests.json"
        )

        if not tests:
            return

        model = Small_LLM_Model()
        decoder = Decoder(model)

        results = []

        for test in tests:
            print("=" * 50)
            print(test.prompt)
            prompt = build_prompt(functions, test.prompt)

            input_ids = model.encode(prompt).tolist()[0]

            decoded = decoder.decode(input_ids, functions)
            print(json.dumps(decoded, indent=4))

            results.append(
                {
                    "prompt": test.prompt,
                    "name": decoded["name"],
                    "parameters": decoded["parameters"],
                }
            )

        os.makedirs("data/output", exist_ok=True)

        with open(
            "data/output/function_calling_results.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(results, file, indent=4)
    except KeyboardInterrupt:
            print("\nInterrupted by user.")

