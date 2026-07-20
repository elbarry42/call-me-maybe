import json
from typing import Any
from .models import FunctionDefinition, FunctionCallTest

JSONDict = dict[str, Any]
JSONList = list[JSONDict]


def load_json_file(path: str) -> JSONList:
    """
    Load and return the content of a JSON file.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{path}' not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{path}' is not a valid JSON file.")


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    raw_data = load_json_file(path)

    functions = []

    for item in raw_data:
        function = FunctionDefinition(**item)
        functions.append(function)

    return functions


def load_function_calling_tests(path: str) -> list[FunctionCallTest]:
    raw_data = load_json_file(path)

    tests = []

    for item in raw_data:
        function = FunctionCallTest(**item)
        tests.append(function)

    return tests
