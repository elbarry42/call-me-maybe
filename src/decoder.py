from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model
from .json_builder import JsonBuilder
from .constraints import Constraints
from .models import FunctionDefinition
from .parameter_extractor import ParameterExtractor
from .token_utils import TokenUtils
from .generation import Generator
from .decoder_state import DecoderState
from .function_selector import FunctionSelector


class Decoder:
    """Decode user requests into structured function calls."""

    def __init__(self, model: Small_LLM_Model):
        """Initialize the decoder and its helper components."""
        self.model = model
        self.tokens = TokenUtils(model)
        self.constraints = Constraints(self.tokens)
        self.parameter_extractor = ParameterExtractor()
        self.generator = Generator(
            self.tokens,
            self.constraints,
        )
        self.json_builder = JsonBuilder()
        self.function_selector = FunctionSelector(
            self.tokens,
            self.constraints,
            self.generator,
        )

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return the logits produced by the model."""
        return self.tokens.get_logits(input_ids)

    def best_token(self, logits: list[float]) -> int:
        """Return the highest-scoring token ID."""
        return self.tokens.best_token(logits)

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs into text."""
        return self.tokens.decode_tokens(token_ids)

    def filter_logits(
        self,
        logits: list[float],
        allowed_tokens: set[int],
    ) -> list[float]:
        """Keep only logits corresponding to allowed tokens."""
        return self.tokens.filter_logits(
            logits,
            allowed_tokens,
        )

    def generate_one_token(
        self,
        input_ids: list[int],
        state: DecoderState,
        prefix: str,
        functions: list[FunctionDefinition],
        function: FunctionDefinition | None = None,
        parameter_type: str | None = None,
    ) -> int:
        """Generate one token according to the current JSON state."""
        return self.generator.generate_token(
            input_ids=input_ids,
            state=state,
            prefix=prefix,
            functions=functions,
            function=function,
            parameter_type=parameter_type,
        )

    def _append_token(
        self,
        input_ids: list[int],
        generated_tokens: list[int],
        token_id: int,
    ) -> None:
        """Append a generated token to the current sequences."""
        input_ids.append(token_id)
        generated_tokens.append(token_id)

    def _generated_text(
        self,
        generated_tokens: list[int],
    ) -> str:
        """Return the text generated so far."""
        return self.tokens.decode_tokens(generated_tokens)

    def _generate_until(
        self,
        input_ids: list[int],
        generated_tokens: list[int],
        state: DecoderState,
        functions: list[FunctionDefinition],
        function: FunctionDefinition | None = None,
        parameter_type: str | None = None,
        max_tokens: int = 256,
    ) -> str:
        """Generate tokens until the current JSON fragment is complete."""
        for _ in range(max_tokens):
            prefix = self._generated_text(generated_tokens)

            token_id = self.generate_one_token(
                input_ids=input_ids,
                state=state,
                prefix=prefix,
                functions=functions,
                function=function,
                parameter_type=parameter_type,
            )

            self._append_token(
                input_ids,
                generated_tokens,
                token_id,
            )

            text = self._generated_text(generated_tokens)

            if self._fragment_complete(text, state):
                return text

        raise ValueError(
            f"Maximum generation length reached in state {state}."
        )

    def _fragment_complete(
        self,
        text: str,
        state: DecoderState,
    ) -> bool:
        """Check whether the current JSON fragment is complete."""
        if state == DecoderState.OPEN_BRACE:
            return "{" in text

        if state == DecoderState.NAME_KEY:
            return text.endswith('"name"')

        if state == DecoderState.COLON:
            return text.endswith(":")

        if state == DecoderState.NAME_VALUE:
            return text.endswith('"')

        if state == DecoderState.PARAMETERS_KEY:
            return text.endswith('"parameters"')

        if state == DecoderState.PARAM_NAME:
            return text.endswith('"')

        if state == DecoderState.PARAM_VALUE:
            return text.endswith(",") or text.endswith("}")

        if state == DecoderState.COMMA:
            return text.endswith(",")

        if state == DecoderState.CLOSE_BRACE:
            return text.endswith("}")

        return False

    def _validate_parameters(
        self,
        function: FunctionDefinition,
        parameters: dict[str, Any],
    ) -> None:
        """Ensure that every required parameter is present."""
        required = set(function.parameters)
        received = set(parameters)

        missing = required - received

        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing required parameters: {missing_names}"
            )

    def extract_parameters(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        """Extract and validate all function parameters."""
        parameters = self.parameter_extractor.extract(
            user_prompt,
            function,
        )

        self._validate_parameters(
            function,
            parameters,
        )

        return parameters

    def build_result(
        self,
        function: FunctionDefinition,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate parameters and build the final function call."""
        self._validate_parameters(
            function,
            parameters,
        )

        return self.json_builder.build(
            name=function.name,
            parameters=parameters,
        )

