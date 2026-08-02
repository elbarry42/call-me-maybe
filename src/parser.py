import json
from typing import Any
from typing import cast
from pydantic import ValidationError
from .models import FunctionDefinition, FunctionCallTest

JSONDict = dict[str, Any]
JSONList = list[JSONDict]


def load_json_file(path: str) -> JSONList | None:
    """
    Load and return the content of a JSON file.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if not isinstance(data, list):
                print(f"Error: '{path}' must contain a JSON array.")
                return None

            return cast(JSONList, data)
    except FileNotFoundError:
        print(f"Error: File '{path}' not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{path}' is not a valid JSON file.")
        return None


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    raw_data = load_json_file(path)

    if raw_data is None:
        return []

    functions = []

    for item in raw_data:
        if not isinstance(item, dict):
            print("Invalid function definition.")
            return []

        try:
            function = FunctionDefinition(**item)
        except ValidationError:
            print("Invalid function definition.")
            return []
        functions.append(function)

    return functions


def load_function_calling_tests(path: str) -> list[FunctionCallTest]:
    raw_data = load_json_file(path)

    if raw_data is None:
        return []

    tests = []

    for item in raw_data:
        if not isinstance(item, dict):
            print("Invalid calling test.")
            return []

        try:
            function = FunctionCallTest(**item)
        except ValidationError:
            print("Invalid calling test.")
            return []
        tests.append(function)

    return tests
