"""JSON Schema generation from Python type hints."""

from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from dataclasses import fields, is_dataclass
from enum import Enum


def get_json_type(python_type: Type) -> Dict[str, Any]:
    """Convert Python type to JSON Schema type.

    Args:
        python_type: The Python type to convert.

    Returns:
        A dictionary representing the JSON Schema type.
    """
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "contentEncoding": "base64"},
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle Any type
    if python_type is Any:
        return {}

    # Handle origin types (List, Dict, Optional, Union, etc.)
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Optional (Union with None)
    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            # This is Optional[X]
            schema = get_json_type(non_none_args[0])
            if "type" in schema:
                if isinstance(schema["type"], list):
                    schema["type"].append("null")
                else:
                    schema["type"] = [schema["type"], "null"]
            return schema
        else:
            # This is a Union of multiple types
            return {"oneOf": [get_json_type(arg) for arg in args]}

    # Handle List
    if origin is list:
        if args:
            return {"type": "array", "items": get_json_type(args[0])}
        return {"type": "array"}

    # Handle Dict
    if origin is dict:
        schema: Dict[str, Any] = {"type": "object"}
        if len(args) >= 2:
            schema["additionalProperties"] = get_json_type(args[1])
        return schema

    # Handle Enum
    if isinstance(python_type, type) and issubclass(python_type, Enum):
        return {
            "type": "string",
            "enum": [e.value for e in python_type]
        }

    # Handle dataclasses
    if is_dataclass(python_type):
        return generate_dataclass_schema(python_type)

    # Default to object for unknown types
    return {"type": "object"}


def generate_dataclass_schema(dataclass_type: Type) -> Dict[str, Any]:
    """Generate JSON Schema from a dataclass.

    Args:
        dataclass_type: The dataclass type to generate schema for.

    Returns:
        A dictionary representing the JSON Schema.
    """
    properties = {}
    required = []

    try:
        hints = get_type_hints(dataclass_type)
    except Exception:
        hints = {}

    for field in fields(dataclass_type):
        field_type = hints.get(field.name, Any)
        field_schema = get_json_type(field_type)

        # Add description if available
        if field.metadata.get("description"):
            field_schema["description"] = field.metadata["description"]

        properties[field.name] = field_schema

        # Check if field is required (no default value)
        if (field.default is field.default_factory
            and field.default_factory is type(field).default_factory):
            required.append(field.name)

    schema = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required

    return schema


def generate_function_schema(
    func: Callable,
    description: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate JSON Schema from a function's signature.

    Args:
        func: The function to generate schema for.
        description: Optional description override.
        name: Optional name override.

    Returns:
        A dictionary representing the JSON Schema for the function.
    """
    sig = inspect.signature(func)

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        # Skip self, cls, and special parameters
        if param_name in ("self", "cls", "agent", "context"):
            continue

        param_type = hints.get(param_name, Any)
        param_schema = get_json_type(param_type)

        # Extract description from docstring if available
        param_description = extract_param_description(func, param_name)
        if param_description:
            param_schema["description"] = param_description

        # Handle default values
        if param.default is not inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = param_schema

    # Build the schema
    func_name = name or func.__name__
    func_description = description or func.__doc__ or f"Execute {func_name}"

    # Clean up description
    func_description = func_description.strip().split("\n")[0].strip()

    schema = {
        "name": func_name,
        "description": func_description,
        "parameters": {
            "type": "object",
            "properties": properties,
        }
    }

    if required:
        schema["parameters"]["required"] = required

    return schema


def extract_param_description(func: Callable, param_name: str) -> Optional[str]:
    """Extract parameter description from function docstring.

    Args:
        func: The function to extract from.
        param_name: The parameter name to find.

    Returns:
        The parameter description if found, None otherwise.
    """
    docstring = func.__doc__
    if not docstring:
        return None

    # Parse Google-style docstring
    lines = docstring.split("\n")
    in_args_section = False

    for line in lines:
        stripped = line.strip()

        # Check for Args section
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue

        # Check for end of Args section
        if in_args_section and stripped and not stripped.startswith(" ") and ":" in stripped:
            if not stripped.startswith(param_name):
                if stripped.lower() in ("returns:", "raises:", "yields:", "examples:"):
                    in_args_section = False
                    continue

        if in_args_section:
            # Look for parameter definition
            if stripped.startswith(f"{param_name}:") or stripped.startswith(f"{param_name} "):
                # Extract description after the colon
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    if len(parts) > 1:
                        return parts[1].strip()

    return None


def validate_against_schema(value: Any, schema: Dict[str, Any]) -> List[str]:
    """Validate a value against a JSON Schema.

    Args:
        value: The value to validate.
        schema: The JSON Schema to validate against.

    Returns:
        A list of validation errors (empty if valid).
    """
    errors: List[str] = []

    schema_type = schema.get("type")

    if schema_type is None:
        return errors

    # Handle type arrays (e.g., ["string", "null"])
    if isinstance(schema_type, list):
        type_valid = False
        for t in schema_type:
            if _check_type(value, t):
                type_valid = True
                break
        if not type_valid:
            errors.append(f"Expected one of types {schema_type}, got {type(value).__name__}")
    else:
        if not _check_type(value, schema_type):
            errors.append(f"Expected type {schema_type}, got {type(value).__name__}")

    # Validate object properties
    if schema_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for req in required:
            if req not in value:
                errors.append(f"Missing required property: {req}")

        for prop_name, prop_value in value.items():
            if prop_name in properties:
                prop_errors = validate_against_schema(prop_value, properties[prop_name])
                errors.extend([f"{prop_name}.{e}" for e in prop_errors])

    # Validate array items
    if schema_type == "array" and isinstance(value, list):
        items_schema = schema.get("items", {})
        for i, item in enumerate(value):
            item_errors = validate_against_schema(item, items_schema)
            errors.extend([f"[{i}].{e}" for e in item_errors])

    # Validate enum
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"Value must be one of: {schema['enum']}")

    return errors


def _check_type(value: Any, type_name: str) -> bool:
    """Check if a value matches a JSON Schema type name."""
    if type_name == "null":
        return value is None
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return True
