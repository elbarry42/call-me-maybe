from enum import Enum


class DecoderState(Enum):
    """Represent the current state of JSON generation."""

    OPEN_BRACE = 0
    NAME_KEY = 1
    COLON = 2
    NAME_VALUE = 3
    COMMA = 4
    PARAMETERS_KEY = 5
    PARAM_NAME = 6
    PARAM_VALUE = 7
    CLOSE_BRACE = 8
