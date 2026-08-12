import json
from typing import cast

from llm_sdk.llm_sdk import Small_LLM_Model


class TokenUtils:
    """Provide utilities for token and model operations."""

    def __init__(self, model: Small_LLM_Model):
        """Initialize token mappings from the model vocabulary."""
        self.model = model

        vocab_path = model.get_path_to_vocab_file()

        try:
            with open(vocab_path, "r", encoding="utf-8") as file:
                self.token_to_id: dict[str, int] = cast(
                    dict[str, int],
                    json.load(file),
                )
        except OSError as error:
            raise OSError(
                f"Unable to read vocabulary file '{vocab_path}': {error}"
            ) from error

        self.id_to_token: dict[int, str] = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

        self.decoded_tokens: dict[int, str] = {
            token_id: self.decode_tokens([token_id])
            for token_id in self.id_to_token
        }

    def token_id(self, token: str) -> int:
        """Return the ID associated with a token."""
        return self.token_to_id[token]

    def token(self, token_id: int) -> str:
        """Return the token associated with an ID."""
        return self.id_to_token[token_id]

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
        """Disable every token that is not allowed."""
        filtered = logits.copy()

        for token_id in range(len(filtered)):
            if token_id not in allowed_tokens:
                filtered[token_id] = float("-inf")

        return filtered

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return model logits for the given input token IDs."""
        return cast(
            list[float],
            self.model.get_logits_from_input_ids(input_ids),
        )

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs into text."""
        return cast(str, self.model.decode(token_ids))