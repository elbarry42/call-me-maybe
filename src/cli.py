import os
import json
import argparse
from typing import Any


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the function calling assistant."
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input tests JSON file.",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )

    return parser.parse_args()


def write_output(path: str, results: list[dict[str, Any]]) -> None:
    """Write function calling results to a JSON file."""
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, allow_nan=False)
