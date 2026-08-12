import json
from typing import Any


class JsonBuilder:
    """Build and validate function call JSON objects."""

    def build(
        self,
        name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a function call object."""
        result: dict[str, Any] = {
            "name": name,
            "parameters": parameters,
        }

        self.validate(result)

        return result

    def validate(self, result: dict[str, Any]) -> None:
        """Validate that the result can be represented as strict JSON."""
        try:
            json.dumps(
                result,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Function call cannot be represented as valid JSON."
            ) from error
