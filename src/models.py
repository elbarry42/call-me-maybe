from typing import Literal
from pydantic import BaseModel, Field


AllowedType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
]


class ParameterDefinition(BaseModel):
    type: AllowedType


class ReturnDefinition(BaseModel):
    type: AllowedType


class FunctionDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class FunctionCallTest(BaseModel):
    prompt: str = Field(min_length=1)
