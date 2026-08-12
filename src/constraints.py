from .decoder_state import DecoderState
from .models import FunctionDefinition
from .token_utils import TokenUtils


class Constraints:
    """Define which tokens are allowed for each decoder state."""

    def __init__(self, tokens: TokenUtils):
        """Initialize constraints with token utilities."""
        self.tokens = tokens

    def allowed_name_key(self, prefix: str) -> set[int]:
        """Return tokens that can complete the name key."""
        return self._allowed_text(prefix, '"name"')

    def allowed_parameters_key(self, prefix: str) -> set[int]:
        """Return tokens that can complete the parameters key."""
        return self._allowed_text(prefix, '"parameters"')

    def allowed_name_value(
        self,
        prefix: str,
        functions: list[FunctionDefinition],
    ) -> set[int]:
        """Return tokens that can complete a valid function name."""
        allowed: set[int] = set()

        for function in functions:
            target = f'"{function.name}"'
            allowed.update(self._allowed_text(prefix, target))

        return allowed

    def allowed_param_name(
        self,
        prefix: str,
        function: FunctionDefinition,
    ) -> set[int]:
        """Return tokens that can complete a parameter name."""
        allowed: set[int] = set()

        for name in function.parameters:
            target = f'"{name}"'
            allowed.update(self._allowed_text(prefix, target))

        return allowed

    def allowed_param_value(
        self,
        prefix: str,
        parameter_type: str,
    ) -> set[int]:
        """Return tokens allowed for a parameter value."""
        allowed: set[int] = set()

        for token_id, token in self.tokens.decoded_tokens.items():
            if parameter_type in ("number", "integer"):
                self._allow_number_token(
                    allowed,
                    token_id,
                    token,
                    prefix,
                )

            elif parameter_type == "boolean":
                self._allow_boolean_token(
                    allowed,
                    token_id,
                    token,
                    prefix,
                )

            elif parameter_type == "string":
                allowed.add(token_id)

        return allowed

    def allowed_colon(self) -> set[int]:
        """Return tokens containing a JSON colon."""
        return self._tokens_containing(":")

    def allowed_comma(self) -> set[int]:
        """Return tokens containing a JSON comma."""
        return self._tokens_containing(",")

    def allowed_open_brace(self) -> set[int]:
        """Return tokens containing an opening brace."""
        return self._tokens_containing("{")

    def allowed_close_brace(self) -> set[int]:
        """Return tokens containing a closing brace."""
        return self._tokens_containing("}")

    def allowed_tokens(
        self,
        state: DecoderState,
        prefix: str,
        function: FunctionDefinition | None = None,
        functions: list[FunctionDefinition] | None = None,
        parameter_type: str | None = None,
    ) -> set[int]:
        """Return the allowed tokens for the current decoder state."""

        if state == DecoderState.OPEN_BRACE:
            return self.allowed_open_brace()

        if state == DecoderState.NAME_KEY:
            return self.allowed_name_key(prefix)

        if state == DecoderState.COLON:
            return self.allowed_colon()

        if state == DecoderState.NAME_VALUE:
            if functions is None:
                raise ValueError(
                    "Functions are required for NAME_VALUE state."
                )
            return self.allowed_name_value(prefix, functions)

        if state == DecoderState.PARAMETERS_KEY:
            return self.allowed_parameters_key(prefix)

        if state == DecoderState.PARAM_NAME:
            if function is None:
                raise ValueError(
                    "A function is required for PARAM_NAME state."
                )
            return self.allowed_param_name(prefix, function)

        if state == DecoderState.PARAM_VALUE:
            if parameter_type is None:
                raise ValueError(
                    "Parameter type is required for PARAM_VALUE state."
                )
            return self.allowed_param_value(prefix, parameter_type)

        if state == DecoderState.CLOSE_BRACE:
            return self.allowed_close_brace()

        if state == DecoderState.COMMA:
            return self.allowed_comma()

        raise ValueError(f"Unknown decoder state: {state}")

    def _allowed_text(
        self,
        prefix: str,
        target: str,
    ) -> set[int]:
        """Return tokens that keep a text prefix valid."""
        allowed: set[int] = set()

        for token_id, token in self.tokens.decoded_tokens.items():
            candidate = prefix + token

            if target.startswith(candidate):
                allowed.add(token_id)

        return allowed

    def _tokens_containing(self, character: str) -> set[int]:
        """Return token IDs containing a given character."""
        return {
            token_id
            for token_id, token in self.tokens.decoded_tokens.items()
            if character in token
        }

    def _allow_number_token(
        self,
        allowed: set[int],
        token_id: int,
        token: str,
        prefix: str,
    ) -> None:
        """Allow a token when it can be part of a number."""
        if any(char.isdigit() for char in token):
            allowed.add(token_id)

        if "." in token:
            allowed.add(token_id)

        if "-" in token and not prefix.strip():
            allowed.add(token_id)

        if "," in token or "}" in token:
            allowed.add(token_id)

    def _allow_boolean_token(
        self,
        allowed: set[int],
        token_id: int,
        token: str,
        prefix: str,
    ) -> None:
        """Allow a token when it can complete a boolean."""
        candidate = prefix + token

        if "true".startswith(candidate):
            allowed.add(token_id)

        if "false".startswith(candidate):
            allowed.add(token_id)

        if "," in token or "}" in token:
            allowed.add(token_id)

    def remaining_parameters(
        self,
        function: FunctionDefinition,
        generated_parameters: set[str],
    ) -> list[str]:
        """Return required parameters that have not been generated."""
        return [
            name
            for name in function.parameters
            if name not in generated_parameters
        ]

    def can_close_parameters(
        self,
        function: FunctionDefinition,
        generated_parameters: set[str],
    ) -> bool:
        """Return whether all required parameters are present."""
        return not self.remaining_parameters(
            function,
            generated_parameters,
        )
