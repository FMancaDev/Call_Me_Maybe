from typing import Any
from pydantic import BaseModel, ConfigDict


class ParameterDefinition(BaseModel):
    """Represent a function parameter definition"""

    type: str
    model_config = ConfigDict(extra="forbid")


class ReturnDefinition(BaseModel):
    """Represent a function return type definition"""

    type: str
    model_config = ConfigDict(extra="forbid")


class FunctionDefinition(BaseModel):
    """Represent one available function"""

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition

    model_config = ConfigDict(extra="forbid")


class PromptInput(BaseModel):
    """Represent one natural-language prompt from the input file"""

    prompt: str
    model_config = ConfigDict(extra="forbid")


class FunctionCallResult(BaseModel):
    """Represent one generated function call"""

    prompt: str
    name: str
    parameters: dict[str, Any]

    model_config = ConfigDict(extra="forbid")
