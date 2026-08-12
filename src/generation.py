from .constraints import Constraints
from .decoder_state import DecoderState
from .models import FunctionDefinition
from .token_utils import TokenUtils


class Generator:
    """Generate tokens while respecting decoding constraints."""

    def __init__(
        self,
        tokens: TokenUtils,
        constraints: Constraints,
    ):
        """Initialize the generator."""
        self.tokens = tokens
        self.constraints = constraints

    def generate_token(
        self,
        input_ids: list[int],
        state: DecoderState,
        prefix: str,
        functions: list[FunctionDefinition],
        function: FunctionDefinition | None = None,
        parameter_type: str | None = None,
    ) -> int:
        """Generate the highest-scoring token allowed by the state."""
        allowed_tokens = self.constraints.allowed_tokens(
            state=state,
            prefix=prefix,
            functions=functions,
            function=function,
            parameter_type=parameter_type,
        )

        if not allowed_tokens:
            raise ValueError(
                f"No valid tokens for decoder state {state}."
            )

        logits = self.tokens.get_logits(input_ids)

        filtered_logits = self.tokens.filter_logits(
            logits,
            allowed_tokens,
        )

        return self.tokens.best_token(filtered_logits)
