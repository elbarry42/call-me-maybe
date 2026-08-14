import json
from typing import Any
from typing import cast

from .models import FunctionDefinition
from llm_sdk.llm_sdk import Small_LLM_Model
from .parameter_extractor import ParameterExtractor


class Decoder:
    """Generate valid function calls with constrained decoding."""

    def __init__(self, model: Small_LLM_Model):
        """Initialize the decoder with the model vocabulary."""
        self.model = model
        self.parameter_extractor = ParameterExtractor()

        vocab_path = model.get_path_to_vocab_file()

        try:
            with open(vocab_path, "r", encoding="utf-8") as file:
                self.token_to_id = cast(
                    dict[str, int],
                    json.load(file),
                )
        except OSError as error:
            raise OSError(
                f"Unable to read vocabulary: {error}"
            ) from error

        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

        self.decoded_tokens = {
            token_id: self.model.decode([token_id])
            for token_id in self.id_to_token
        }

    # ---------------------------------------------------------
    # Token utilities
    # ---------------------------------------------------------

    def token_id(self, token: str) -> int:
        """Return the ID of a token."""
        return self.token_to_id[token]

    def token(self, token_id: int) -> str:
        """Return the token corresponding to an ID."""
        return self.id_to_token[token_id]

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs into text."""
        return cast(str, self.model.decode(token_ids))

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return logits produced by the model."""
        return cast(
            list[float],
            self.model.get_logits_from_input_ids(input_ids),
        )

    def best_token(self, logits: list[float]) -> int:
        """Return the ID of the token with the highest logit."""
        if not logits:
            raise ValueError("Logits cannot be empty.")

        return max(
            range(len(logits)),
            key=lambda token_id: logits[token_id],
        )

    def filter_logits(
        self,
        logits: list[float],
        allowed_tokens: set[int],
    ) -> list[float]:
        """Keep only allowed tokens."""
        filtered = logits.copy()

        for token_id in range(len(filtered)):
            if token_id not in allowed_tokens:
                filtered[token_id] = float("-inf")

        return filtered

    def allowed_tokens_containing(
        self,
        character: str,
    ) -> set[int]:
        """Return tokens containing a specific character."""
        return {
            token_id
            for token_id, token in self.decoded_tokens.items()
            if character in token
        }

    def allowed_open_brace(self) -> set[int]:
        """Return tokens containing an opening brace."""
        return self.allowed_tokens_containing("{")

    def allowed_close_brace(self) -> set[int]:
        """Return tokens containing a closing brace."""
        return self.allowed_tokens_containing("}")

    def allowed_colon(self) -> set[int]:
        """Return tokens containing a colon."""
        return self.allowed_tokens_containing(":")

    def allowed_comma(self) -> set[int]:
        """Return tokens containing a comma."""
        return self.allowed_tokens_containing(",")

    def allowed_text(
        self,
        prefix: str,
        target: str,
    ) -> set[int]:
        """Return tokens that keep a target string valid."""
        allowed = set()

        for token_id, token in self.decoded_tokens.items():
            candidate = prefix + token

            if target.startswith(candidate):
                allowed.add(token_id)

        return allowed

    def allowed_name_key(self, prefix: str) -> set[int]:
        """Return tokens allowed for the name key."""
        return self.allowed_text(prefix, '"name"')

    def allowed_parameters_key(self, prefix: str) -> set[int]:
        """Return tokens allowed for the parameters key."""
        return self.allowed_text(prefix, '"parameters"')

    def allowed_name_value(
        self,
        prefix: str,
        functions: list[FunctionDefinition],
    ) -> set[int]:
        """Return tokens allowed for a valid function name."""
        allowed = set()

        for function in functions:
            allowed.update(
                self.allowed_text(
                    prefix,
                    f'"{function.name}"',
                )
            )

        return allowed

    def allowed_param_name(
        self,
        prefix: str,
        function: FunctionDefinition,
    ) -> set[int]:
        """Return tokens allowed for a parameter name."""
        allowed = set()

        for name in function.parameters:
            allowed.update(
                self.allowed_text(
                    prefix,
                    f'"{name}"',
                )
            )

        return allowed

    def remaining_parameters(
        self,
        function: FunctionDefinition,
        generated: set[str],
    ) -> list[str]:
        """Return parameters that still need to be generated."""
        return [
            name
            for name in function.parameters
            if name not in generated
        ]

    def can_close_parameters(
        self,
        function: FunctionDefinition,
        generated: set[str],
    ) -> bool:
        """Return whether all required parameters are present."""
        return not self.remaining_parameters(
            function,
            generated,
        )

    def generate_token(
        self,
        input_ids: list[int],
        allowed_tokens: set[int],
    ) -> int:
        """Generate the highest-scoring allowed token."""
        if not allowed_tokens:
            raise ValueError("No allowed tokens available.")

        logits = self.get_logits(input_ids)
        filtered = self.filter_logits(
            logits,
            allowed_tokens,
        )

        return self.best_token(filtered)

    def select_function(
        self,
        input_ids: list[int],
        functions: list[FunctionDefinition],
    ) -> tuple[FunctionDefinition, list[int]]:
        """Generate a valid function name and return its definition."""
        generated_tokens: list[int] = []

        for _ in range(64):
            prefix = self.decode_tokens(generated_tokens)

            allowed_tokens = self.allowed_name_value(
                prefix,
                functions,
            )

            token_id = self.generate_token(
                input_ids,
                allowed_tokens,
            )

            input_ids.append(token_id)
            generated_tokens.append(token_id)

            text = self.decode_tokens(generated_tokens)

            if text.startswith('"') and text.endswith('"'):
                name = text[1:-1]

                for function in functions:
                    if function.name == name:
                        return function, generated_tokens

        raise ValueError(
            "Unable to generate a valid function name."
        )

    def generate_parameters(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        """Generate all required parameters for a function."""
        parameters = self.parameter_extractor.extract(
            user_prompt,
            function,
        )

        required = set(function.parameters)
        received = set(parameters)
        missing = required - received

        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing required parameters: {names}"
            )

        return parameters

    def decode(
        self,
        input_ids: list[int],
        functions: list[FunctionDefinition],
        user_prompt: str,
    ) -> dict[str, Any]:
        """Decode a user request into a function call."""
        if not functions:
            raise ValueError("No functions are available.")

        function, _ = self.select_function(
            input_ids,
            functions,
        )

        parameters = self.generate_parameters(
            user_prompt,
            function,
        )

        result = {
            "name": function.name,
            "parameters": parameters,
        }

        self.validate_json(result)

        return result

    def validate_json(self, result: dict[str, Any]) -> None:
        """Ensure the result is valid strict JSON."""
        try:
            json.dumps(
                result,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Generated function call is not valid JSON."
            ) from error
