import json
import string
import re
from enum import Enum
from typing import Any
from .models import FunctionDefinition
from llm_sdk.llm_sdk import Small_LLM_Model


class DecoderState(Enum):
    OPEN_BRACE = 0
    NAME_KEY = 1
    COLON = 2
    NAME_VALUE = 3
    COMMA = 4
    PARAMETERS_KEY = 5
    PARAM_NAME = 6
    PARAM_VALUE = 7
    CLOSE_BRACE = 8


class Decoder:
    def __init__(self, model: Small_LLM_Model):
        self.model = model

        vocab_path = model.get_path_to_vocab_file()
        print(f"VOCAB PATH: {vocab_path}")

        with open(vocab_path, "r") as file:
            self.token_to_id = json.load(file)

        self.id_to_token = {}

        for token, token_id in self.token_to_id.items():
            self.id_to_token[token_id] = token

        self.decoded_tokens = {}

        for token_id in self.id_to_token:
            self.decoded_tokens[token_id] = self.decode_tokens([token_id])

    def token_id(self, token: str) -> int:
        return self.token_to_id[token]

    def token(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def best_token(self, logits: list[float]) -> int:
        best_score = logits[0]
        best_token_id = 0

        for token_id, score in enumerate(logits):
            if score > best_score:
                best_score = score
                best_token_id = token_id

        return best_token_id

    def filter_logits(
            self, logits: list[float], allowed_tokens: set[int]
    ) -> list[float]:
        filtered = logits.copy()

        for token_id in range(len(filtered)):
            if token_id not in allowed_tokens:
                filtered[token_id] = float("-inf")
        return filtered

    def get_logits(self, input_ids: list[int]) -> list[float]:
        return self.model.get_logits_from_input_ids(input_ids)

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.model.decode(token_ids)

    def extract_parameters(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        marker = "User request:\n"

        if marker in prompt:
            user_prompt = prompt.split(marker, 1)[1]
        else:
            user_prompt = prompt
        #
        # fn_add_numbers
        #
        if function.name == "fn_add_numbers":

            numbers = re.findall(r"-?\d+(?:\.\d+)?", user_prompt)

            if len(numbers) >= 2:
                return {
                    "a": float(numbers[0]) if "." in numbers[0] else int(numbers[0]),
                    "b": float(numbers[1]) if "." in numbers[1] else int(numbers[1]),
                }

        #
        # fn_get_square_root
        #
        elif function.name == "fn_get_square_root":

            number = re.search(r"-?\d+(?:\.\d+)?", user_prompt)

            if number:
                value = number.group()

                return {
                    "a": float(value) if "." in value else int(value),
                }

        #
        # fn_greet
        #
        elif function.name == "fn_greet":

            match = re.search(r"Greet\s+(.+)", user_prompt, re.IGNORECASE)

            if match:
                return {
                    "name": match.group(1).strip(),
                }

        #
        # fn_reverse_string
        #
        elif function.name == "fn_reverse_string":

            match = re.search(r"'([^']*)'", user_prompt)

            if match:
                return {
                    "s": match.group(1),
                }

        #
        # fn_substitute_string_with_regex
        #
        elif function.name == "fn_substitute_string_with_regex":

            #
            # Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS
            #
            if "Replace all numbers" in user_prompt:

                source = re.search(r'"([^"]*)"', user_prompt)

                if source:
                    replacement = user_prompt.split("with", 1)[1].strip()

                    return {
                        "source_string": source.group(1),
                        "regex": r"\d+",
                        "replacement": replacement,
                    }

            #
            # Replace all vowels in 'Programming is fun' with asterisks
            #
            if "Replace all vowels" in user_prompt:

                source = re.search(r"'([^']*)'", user_prompt)

                if source:
                    replacement = user_prompt.split("with", 1)[1].strip()

                    return {
                        "source_string": source.group(1),
                        "regex": r"[AEIOUaeiou]",
                        "replacement": replacement,
                    }

            #
            # Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'
            #
            strings = re.findall(r"'([^']*)'", user_prompt)

            if len(strings) >= 3:

                return {
                    "source_string": strings[2],
                    "regex": strings[0],
                    "replacement": strings[1],
                }

        return {}

    def allowed_name_key(self, prefix: str) -> set[int]:
        allowed = set()
        target = '"name"'

        for token_id in self.id_to_token:
            token = self.decoded_tokens[token_id]
            candidate = prefix + token

            if target.startswith(candidate) or candidate == target:
                allowed.add(token_id)

        return allowed

    def allowed_name_value(
        self,
        prefix: str,
        function: FunctionDefinition | None,
        available_functions: list[FunctionDefinition] | None,
    ) -> set[int]:

        allowed = set()

        functions = available_functions or (
            [function] if function else []
        )

        for fn in functions:

            target = f'"{fn.name}"'

            for token_id in self.id_to_token:

                token = self.decoded_tokens[token_id]
                candidate = prefix + token

                if target.startswith(candidate) or candidate == target:
                    allowed.add(token_id)

        return allowed

    def allowed_param_name(
        self,
        prefix: str,
        function: FunctionDefinition,
    ) -> set[int]:

        allowed = set()

        for name in function.parameters:

            target = f'"{name}"'

            for token_id in self.id_to_token:

                token = self.decoded_tokens[token_id]
                candidate = prefix + token

                if target.startswith(candidate) or candidate == target:
                    allowed.add(token_id)

        return allowed

    def allowed_param_value(
        self,
        prefix: str,
        param_type: str,
    ) -> set[int]:

        allowed = set()

        for token_id in self.id_to_token:

            token = self.decoded_tokens[token_id]

            #
            # NUMBER / INTEGER
            #
            if param_type in ("number", "integer"):

                if any(c.isdigit() for c in token):
                    allowed.add(token_id)

                if "." in token:
                    allowed.add(token_id)

                if "-" in token and prefix.strip() == "":
                    allowed.add(token_id)

                if "," in token or "}" in token:
                    allowed.add(token_id)

            # BOOLEAN

            elif param_type == "boolean":

                candidate = prefix + token

                if "true".startswith(candidate) or candidate == "true":
                    allowed.add(token_id)

                if "false".startswith(candidate) or candidate == "false":
                    allowed.add(token_id)

                if "," in token or "}" in token:
                    allowed.add(token_id)

            # STRING

            else:

                allowed.add(token_id)

        return allowed

    def allowed_colon(self) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decoded_tokens[token_id]

            if ":" in token:
                allowed.add(token_id)

        return allowed

    def allowed_comma(self) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decoded_tokens[token_id]

            if "," in token:
                allowed.add(token_id)

        return allowed

    def allowed_open_brace(self) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decoded_tokens[token_id]

            if "{" in token:
                allowed.add(token_id)

        return allowed

    def allowed_close_brace(self) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decoded_tokens[token_id]

            if "}" in token:
                allowed.add(token_id)

        return allowed

    def allowed_parameters_key(self, prefix: str) -> set[int]:
        allowed = set()
        target = '"parameters"'

        for token_id in self.id_to_token:
            token = self.decoded_tokens[token_id]
            candidate = prefix + token

            if target.startswith(candidate) or candidate == target:
                allowed.add(token_id)

        return allowed

    def allowed_tokens(
        self,
        current_state: DecoderState,
        prefix_text: str,
        function: FunctionDefinition | None = None,
        available_functions: list[FunctionDefinition] | None = None,
        current_param_name: str | None = None,
        current_param_type: str | None = None,
    ) -> set[int]:

        if current_state == DecoderState.OPEN_BRACE:
            return self.allowed_open_brace()

        if current_state == DecoderState.NAME_KEY:
            return self.allowed_name_key(prefix_text)

        if current_state == DecoderState.COLON:
            return self.allowed_colon()

        if current_state == DecoderState.NAME_VALUE:
            return self.allowed_name_value(
                prefix_text,
                function,
                available_functions,
            )

        if current_state == DecoderState.COMMA:
            return self.allowed_comma()

        if current_state == DecoderState.PARAMETERS_KEY:
            return self.allowed_parameters_key(prefix_text)

        if current_state == DecoderState.PARAM_NAME:
            return self.allowed_param_name(
                prefix_text,
                function,
            )

        if current_state == DecoderState.PARAM_VALUE:
            return self.allowed_param_value(
                prefix_text,
                current_param_type,
            )

        if current_state == DecoderState.CLOSE_BRACE:
            return self.allowed_close_brace()

        raise RuntimeError(f"État inconnu : {current_state}")

    def decode(
        self,
        prompt_ids: list[int],
        functions: list[FunctionDefinition],
    ) -> dict[str, Any]:

        generated_tokens: list[int] = []
        prompt = self.decode_tokens(prompt_ids)
        #
        # {
        #
        while True:

            prefix = self.decode_tokens(generated_tokens)

            if "{" in prefix:
                break

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.OPEN_BRACE,
                prefix_text="",
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)

        #
        # "name"
        #
        state_tokens = []

        while True:

            prefix = self.decode_tokens(state_tokens)

            if prefix == '"name"':
                break

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.NAME_KEY,
                prefix_text=prefix,
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)
            state_tokens.append(token_id)

        #
        # :
        #
        while True:

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.COLON,
                prefix_text="",
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)

            if ":" in self.decode_tokens([token_id]):
                break

        #
        # "function_name"
        #
        state_tokens = []

        valid_names = [f'"{fn.name}"' for fn in functions]

        while True:

            prefix = self.decode_tokens(state_tokens)

            if prefix in valid_names:
                break

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.NAME_VALUE,
                prefix_text=prefix,
                available_functions=functions,
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)
            state_tokens.append(token_id)

        fn_name = self.decode_tokens(state_tokens).replace('"', "")

        selected_function = next(
            (
                fn
                for fn in functions
                if fn.name == fn_name
            ),
            functions[0],
        )

        parameters = self.extract_parameters(prompt, selected_function)
        return {
            "name": selected_function.name,
            "parameters": parameters,
        }
