import re

from .models import FunctionDefinition


class ParameterExtractor:
    """Extract function parameters from a user request."""

    def extract(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, object]:
        """Extract all parameters required by a function."""
        parameters = function.parameters

        if not parameters:
            return {}

        result: dict[str, object] = {}

        if self._is_regex_substitution(function):
            result = self._extract_regex_parameters(user_prompt)

        else:
            result = self._extract_generic_parameters(
                user_prompt,
                function,
            )

        return self._complete_parameters(
            user_prompt,
            function,
            result,
        )

    def _is_regex_substitution(
        self,
        function: FunctionDefinition,
    ) -> bool:
        """Check whether the function performs regex substitution."""
        return function.name == "fn_substitute_string_with_regex"

    def _extract_regex_parameters(
        self,
        user_prompt: str,
    ) -> dict[str, object]:
        """Extract parameters for regex substitution."""
        result: dict[str, object] = {}

        number_match = re.search(
            r"""numbers?\s+in\s+["'](.+?)["']\s+with\s+(.+)$""",
            user_prompt,
            re.IGNORECASE,
        )

        if number_match:
            result["source_string"] = number_match.group(1)
            result["regex"] = r"\d+"
            result["replacement"] = self._normalize_replacement(
                number_match.group(2),
            )
            return result

        vowel_match = re.search(
            r"""vowels?\s+in\s+["'](.+?)["']\s+with\s+(.+)$""",
            user_prompt,
            re.IGNORECASE,
        )

        if vowel_match:
            result["source_string"] = vowel_match.group(1)
            result["regex"] = "[aeiouAEIOU]"
            result["replacement"] = self._normalize_replacement(
                vowel_match.group(2),
            )
            return result

        substitution_match = re.search(
            r"""word\s+['"]([^'"]+)['"]\s+with\s+['"]([^'"]+)['"]"""
            r"""\s+in\s+['"](.+)['"]$""",
            user_prompt,
            re.IGNORECASE,
        )

        if substitution_match:
            result["source_string"] = substitution_match.group(3)
            result["regex"] = substitution_match.group(1)
            result["replacement"] = substitution_match.group(2)
            return result

        return result

    def _normalize_replacement(self, value: str) -> str:
        """Convert natural language replacement descriptions."""
        value = value.strip().strip("\"'")

        replacements = {
            "asterisks": "*",
            "asterisk": "*",
            "hash": "#",
            "number": "NUMBER",
            "numbers": "NUMBERS",
        }

        return replacements.get(
            value.lower(),
            value,
        )

    def _find_source_string(
        self,
        user_prompt: str,
        quoted: list[str],
    ) -> str:
        """Find the string that should be modified."""
        if "in " in user_prompt.lower():
            match = re.search(
                r"\bin\s+['\"]([^'\"]*)['\"]",
                user_prompt,
                re.IGNORECASE,
            )

            if match:
                return match.group(1)

        if quoted:
            return quoted[0]

        return user_prompt

    def _extract_replacement(
        self,
        user_prompt: str,
        candidates: list[str],
    ) -> str | None:
        """Extract a replacement value from the request."""
        lower_prompt = user_prompt.lower()

        for candidate in candidates:
            if candidate.lower() in lower_prompt:
                return candidate

        match = re.search(
            r"\bwith\s+['\"]?([^'\"]+)['\"]?",
            user_prompt,
            re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

        return None

    def _extract_generic_parameters(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, object]:
        """Extract common numeric and string parameters."""
        result: dict[str, object] = {}

        numbers = re.findall(
            r"-?\d+(?:\.\d+)?",
            user_prompt,
        )

        integers = re.findall(
            r"-?\d+",
            user_prompt,
        )

        strings = re.findall(
            r"""['"]([^'"]*)['"]""",
            user_prompt,
        )

        number_index = 0
        integer_index = 0
        string_index = 0

        for name, parameter in function.parameters.items():
            if parameter.type == "number":
                if number_index < len(numbers):
                    result[name] = float(numbers[number_index])
                    number_index += 1

            elif parameter.type == "integer":
                if integer_index < len(integers):
                    result[name] = int(integers[integer_index])
                    integer_index += 1

            elif parameter.type == "string":
                if string_index < len(strings):
                    result[name] = strings[string_index]
                    string_index += 1
                elif name == "name":
                    match = re.search(
                        r"^\s*greet\s+(.+?)\s*$",
                        user_prompt,
                        re.IGNORECASE,
                    )

                    if match:
                        result[name] = match.group(1).strip()

            elif parameter.type == "boolean":
                match = re.search(
                    rf"\b{name}\b\s+(true|false)\b",
                    user_prompt,
                    re.IGNORECASE,
                )

                if match:
                    result[name] = (
                        match.group(1).lower() == "true"
                    )

        return result

    def _complete_parameters(
        self,
        user_prompt: str,
        function: FunctionDefinition,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        """Ensure every required parameter has a value."""
        for name, parameter in function.parameters.items():
            if name in parameters:
                continue

            value = self._fallback_value(
                user_prompt,
                name,
                parameter.type,
            )

            parameters[name] = value

        return parameters

    def _fallback_value(
        self,
        user_prompt: str,
        name: str,
        parameter_type: str,
    ) -> object:
        """Raise when a required parameter cannot be extracted."""
        raise ValueError(
            f"Unable to extract required parameter '{name}' "
            f"from prompt: {user_prompt}"
        )
