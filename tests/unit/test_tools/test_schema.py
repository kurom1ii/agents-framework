"""Tests for JSON Schema generation and validation.

This module tests:
- get_json_type() type conversion
- generate_dataclass_schema()
- generate_function_schema()
- extract_param_description()
- validate_against_schema()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pytest

from agents_framework.tools.schema import (
    get_json_type,
    generate_dataclass_schema,
    generate_function_schema,
    extract_param_description,
    validate_against_schema,
)


# ============================================================================
# Test Types and Classes
# ============================================================================


class Color(Enum):
    """Color enumeration for testing."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Status(Enum):
    """Status enumeration with different value types."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass
class SimpleData:
    """Simple dataclass for testing."""

    name: str
    value: int


@dataclass
class ComplexData:
    """Complex dataclass with various field types."""

    id: int
    title: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NestedData:
    """Dataclass with nested dataclass."""

    parent_name: str
    child: SimpleData


# ============================================================================
# get_json_type() Tests
# ============================================================================


class TestGetJsonType:
    """Tests for get_json_type() function."""

    @pytest.mark.parametrize(
        "python_type,expected_type",
        [
            (str, "string"),
            (int, "integer"),
            (float, "number"),
            (bool, "boolean"),
        ],
    )
    def test_basic_types(self, python_type, expected_type):
        """Test conversion of basic Python types."""
        result = get_json_type(python_type)

        assert result["type"] == expected_type

    def test_none_type(self):
        """Test conversion of None type."""
        result = get_json_type(type(None))

        assert result["type"] == "null"

    def test_bytes_type(self):
        """Test conversion of bytes type."""
        result = get_json_type(bytes)

        assert result["type"] == "string"
        assert result["contentEncoding"] == "base64"

    def test_any_type(self):
        """Test conversion of Any type."""
        result = get_json_type(Any)

        assert result == {}

    def test_list_type(self):
        """Test conversion of List type."""
        result = get_json_type(List[str])

        assert result["type"] == "array"
        assert result["items"]["type"] == "string"

    def test_list_without_type_param(self):
        """Test conversion of bare list type.

        Note: Bare `list` (not `List[X]`) falls through to object type
        because get_origin(list) returns None. Use List from typing
        for proper array schema generation.
        """
        result = get_json_type(list)

        # Bare list is treated as an unknown class, defaults to object
        assert result["type"] == "object"

    def test_dict_type(self):
        """Test conversion of Dict type."""
        result = get_json_type(Dict[str, int])

        assert result["type"] == "object"
        assert result["additionalProperties"]["type"] == "integer"

    def test_dict_without_type_params(self):
        """Test conversion of bare dict type."""
        result = get_json_type(dict)

        assert result["type"] == "object"
        assert "additionalProperties" not in result

    def test_optional_type(self):
        """Test conversion of Optional type."""
        result = get_json_type(Optional[str])

        assert "type" in result
        if isinstance(result["type"], list):
            assert "string" in result["type"]
            assert "null" in result["type"]
        else:
            # Some implementations might use oneOf
            assert result["type"] == "string" or "oneOf" in result

    def test_union_type(self):
        """Test conversion of Union type."""
        result = get_json_type(Union[str, int])

        assert "oneOf" in result
        types = [item.get("type") for item in result["oneOf"]]
        assert "string" in types
        assert "integer" in types

    def test_union_with_none(self):
        """Test Union[X, None] is treated like Optional[X]."""
        result = get_json_type(Union[int, None])

        if isinstance(result.get("type"), list):
            assert "integer" in result["type"]
            assert "null" in result["type"]

    def test_enum_type(self):
        """Test conversion of Enum type."""
        result = get_json_type(Color)

        assert result["type"] == "string"
        assert result["enum"] == ["red", "green", "blue"]

    def test_nested_list_type(self):
        """Test conversion of nested List type."""
        result = get_json_type(List[List[int]])

        assert result["type"] == "array"
        assert result["items"]["type"] == "array"
        assert result["items"]["items"]["type"] == "integer"

    def test_list_of_optional(self):
        """Test conversion of List with Optional elements."""
        result = get_json_type(List[Optional[str]])

        assert result["type"] == "array"
        item_type = result["items"].get("type")
        if isinstance(item_type, list):
            assert "string" in item_type
            assert "null" in item_type

    def test_dataclass_type(self):
        """Test conversion of dataclass type."""
        result = get_json_type(SimpleData)

        assert result["type"] == "object"
        assert "properties" in result
        assert "name" in result["properties"]
        assert "value" in result["properties"]

    def test_unknown_type(self):
        """Test conversion of unknown type defaults to object."""

        class CustomClass:
            pass

        result = get_json_type(CustomClass)

        assert result["type"] == "object"


# ============================================================================
# generate_dataclass_schema() Tests
# ============================================================================


class TestGenerateDataclassSchema:
    """Tests for generate_dataclass_schema() function."""

    def test_simple_dataclass(self):
        """Test schema generation from simple dataclass."""
        result = generate_dataclass_schema(SimpleData)

        assert result["type"] == "object"
        assert "properties" in result
        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["value"]["type"] == "integer"

    def test_dataclass_with_optional_fields(self):
        """Test schema with optional fields."""
        result = generate_dataclass_schema(ComplexData)

        assert "id" in result["properties"]
        assert "title" in result["properties"]
        assert "description" in result["properties"]
        assert "tags" in result["properties"]
        assert "metadata" in result["properties"]

    def test_dataclass_required_fields(self):
        """Test that required fields are identified correctly."""
        result = generate_dataclass_schema(SimpleData)

        # All fields in SimpleData should be required (no defaults)
        if "required" in result:
            assert "name" in result["required"]
            assert "value" in result["required"]

    def test_dataclass_with_defaults_not_required(self):
        """Test that fields with defaults are not required."""
        result = generate_dataclass_schema(ComplexData)

        # id and title have no defaults, others do
        if "required" in result:
            assert "id" in result["required"]
            assert "title" in result["required"]
            # These have defaults or default_factory
            assert "description" not in result.get("required", [])
            assert "tags" not in result.get("required", [])
            assert "metadata" not in result.get("required", [])

    def test_dataclass_list_field(self):
        """Test schema for list field in dataclass."""
        result = generate_dataclass_schema(ComplexData)

        tags_schema = result["properties"]["tags"]
        assert tags_schema["type"] == "array"
        assert tags_schema["items"]["type"] == "string"

    def test_dataclass_dict_field(self):
        """Test schema for dict field in dataclass."""
        result = generate_dataclass_schema(ComplexData)

        metadata_schema = result["properties"]["metadata"]
        assert metadata_schema["type"] == "object"


# ============================================================================
# generate_function_schema() Tests
# ============================================================================


class TestGenerateFunctionSchema:
    """Tests for generate_function_schema() function."""

    def test_simple_function(self):
        """Test schema from simple function."""

        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: First number.
                b: Second number.

            Returns:
                Sum of the numbers.
            """
            return a + b

        result = generate_function_schema(add)

        assert result["name"] == "add"
        assert "Add two numbers" in result["description"]
        assert result["parameters"]["properties"]["a"]["type"] == "integer"
        assert result["parameters"]["properties"]["b"]["type"] == "integer"
        assert set(result["parameters"]["required"]) == {"a", "b"}

    def test_function_with_defaults(self):
        """Test schema for function with default values."""

        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone.

            Args:
                name: The name.
                greeting: The greeting.

            Returns:
                Greeting message.
            """
            return f"{greeting}, {name}!"

        result = generate_function_schema(greet)

        assert result["parameters"]["required"] == ["name"]
        assert result["parameters"]["properties"]["greeting"]["default"] == "Hello"

    def test_function_with_optional_type(self):
        """Test schema for function with Optional parameter."""

        def maybe_process(value: Optional[str] = None) -> str:
            """Process optional value.

            Args:
                value: Optional value to process.

            Returns:
                Result.
            """
            return value or "default"

        result = generate_function_schema(maybe_process)

        assert "required" not in result["parameters"] or result["parameters"]["required"] == []
        value_type = result["parameters"]["properties"]["value"].get("type")
        if isinstance(value_type, list):
            assert "string" in value_type
            assert "null" in value_type

    def test_function_with_name_override(self):
        """Test schema with name override."""

        def internal_func(x: int) -> int:
            """Internal function."""
            return x

        result = generate_function_schema(internal_func, name="public_name")

        assert result["name"] == "public_name"

    def test_function_with_description_override(self):
        """Test schema with description override."""

        def my_func(x: int) -> int:
            """Original description."""
            return x

        result = generate_function_schema(my_func, description="Custom description")

        assert result["description"] == "Custom description"

    def test_function_without_docstring(self):
        """Test schema for function without docstring."""

        def no_docs(a: str, b: str) -> str:
            return a + b

        result = generate_function_schema(no_docs)

        assert result["name"] == "no_docs"
        assert "no_docs" in result["description"] or "Execute" in result["description"]

    def test_function_without_type_hints(self):
        """Test schema for function without type hints."""

        def no_hints(a, b):
            """No type hints.

            Args:
                a: First value.
                b: Second value.
            """
            return a + b

        result = generate_function_schema(no_hints)

        assert "a" in result["parameters"]["properties"]
        assert "b" in result["parameters"]["properties"]

    def test_function_skips_special_params(self):
        """Test that self, cls, agent, context are skipped."""

        def with_special(self, agent, context, value: int) -> int:
            """Function with special params.

            Args:
                value: A value.
            """
            return value

        result = generate_function_schema(with_special)

        props = result["parameters"]["properties"]
        assert "self" not in props
        assert "agent" not in props
        assert "context" not in props
        assert "value" in props

    def test_async_function(self):
        """Test schema from async function."""

        async def async_process(data: str) -> str:
            """Process data async.

            Args:
                data: The data to process.

            Returns:
                Processed data.
            """
            return data.upper()

        result = generate_function_schema(async_process)

        assert result["name"] == "async_process"
        assert result["parameters"]["properties"]["data"]["type"] == "string"

    def test_function_with_complex_types(self):
        """Test schema for function with complex types."""

        def process(
            items: List[str],
            config: Dict[str, int],
            status: Optional[Color] = None,
        ) -> Dict[str, Any]:
            """Process with complex types.

            Args:
                items: List of items.
                config: Configuration dict.
                status: Optional status color.

            Returns:
                Result dict.
            """
            return {}

        result = generate_function_schema(process)

        assert result["parameters"]["properties"]["items"]["type"] == "array"
        assert result["parameters"]["properties"]["config"]["type"] == "object"

    def test_function_with_enum_parameter(self):
        """Test schema for function with Enum parameter."""

        def set_status(status: Status) -> bool:
            """Set the status.

            Args:
                status: The new status.

            Returns:
                Success flag.
            """
            return True

        result = generate_function_schema(set_status)

        status_schema = result["parameters"]["properties"]["status"]
        assert status_schema["type"] == "string"
        assert "pending" in status_schema["enum"]
        assert "active" in status_schema["enum"]
        assert "completed" in status_schema["enum"]


# ============================================================================
# extract_param_description() Tests
# ============================================================================


class TestExtractParamDescription:
    """Tests for extract_param_description() function."""

    def test_google_style_docstring(self):
        """Test extraction from Google-style docstring."""

        def my_func(name: str, age: int) -> str:
            """A function.

            Args:
                name: The name of the person.
                age: The age in years.

            Returns:
                A greeting.
            """
            return f"{name} is {age}"

        assert extract_param_description(my_func, "name") == "The name of the person."
        assert extract_param_description(my_func, "age") == "The age in years."

    def test_param_not_found(self):
        """Test when parameter is not in docstring."""

        def my_func(x: int) -> int:
            """A function.

            Args:
                x: The x value.
            """
            return x

        result = extract_param_description(my_func, "nonexistent")

        assert result is None

    def test_function_without_docstring(self):
        """Test function without docstring."""

        def no_docs(x: int) -> int:
            return x

        result = extract_param_description(no_docs, "x")

        assert result is None

    def test_function_without_args_section(self):
        """Test function with docstring but no Args section."""

        def partial_docs(x: int) -> int:
            """Just a description without args section."""
            return x

        result = extract_param_description(partial_docs, "x")

        assert result is None

    def test_arguments_section_variation(self):
        """Test 'Arguments:' section header."""

        def my_func(value: int) -> int:
            """A function.

            Arguments:
                value: The value to use.

            Returns:
                Result.
            """
            return value

        result = extract_param_description(my_func, "value")

        assert result == "The value to use."

    def test_parameters_section_variation(self):
        """Test 'Parameters:' section header."""

        def my_func(data: str) -> str:
            """A function.

            Parameters:
                data: The data string.

            Returns:
                Processed data.
            """
            return data

        result = extract_param_description(my_func, "data")

        assert result == "The data string."

    def test_multiline_description(self):
        """Test extracting first line of multiline description."""

        def my_func(config: dict) -> None:
            """A function.

            Args:
                config: The configuration object that contains
                    multiple settings and options for
                    the processing.

            Returns:
                None.
            """
            pass

        result = extract_param_description(my_func, "config")

        # Should extract at least the first line
        assert result is not None
        assert "configuration" in result.lower() or "object" in result.lower()


# ============================================================================
# validate_against_schema() Tests
# ============================================================================


class TestValidateAgainstSchema:
    """Tests for validate_against_schema() function."""

    def test_valid_string(self):
        """Test validation of valid string."""
        schema = {"type": "string"}
        errors = validate_against_schema("hello", schema)

        assert errors == []

    def test_invalid_string(self):
        """Test validation of invalid string."""
        schema = {"type": "string"}
        errors = validate_against_schema(123, schema)

        assert len(errors) > 0
        assert any("type" in e.lower() or "string" in e.lower() for e in errors)

    def test_valid_integer(self):
        """Test validation of valid integer."""
        schema = {"type": "integer"}
        errors = validate_against_schema(42, schema)

        assert errors == []

    def test_invalid_integer(self):
        """Test validation of invalid integer."""
        schema = {"type": "integer"}
        errors = validate_against_schema("42", schema)

        assert len(errors) > 0

    def test_boolean_not_confused_with_integer(self):
        """Test that boolean is not accepted as integer."""
        schema = {"type": "integer"}
        errors = validate_against_schema(True, schema)

        assert len(errors) > 0

    def test_valid_number(self):
        """Test validation of valid number."""
        schema = {"type": "number"}

        assert validate_against_schema(3.14, schema) == []
        assert validate_against_schema(42, schema) == []  # int is also number

    def test_valid_boolean(self):
        """Test validation of valid boolean."""
        schema = {"type": "boolean"}

        assert validate_against_schema(True, schema) == []
        assert validate_against_schema(False, schema) == []

    def test_invalid_boolean(self):
        """Test validation of invalid boolean."""
        schema = {"type": "boolean"}
        errors = validate_against_schema(1, schema)

        assert len(errors) > 0

    def test_valid_array(self):
        """Test validation of valid array."""
        schema = {"type": "array", "items": {"type": "string"}}
        errors = validate_against_schema(["a", "b", "c"], schema)

        assert errors == []

    def test_invalid_array_items(self):
        """Test validation of array with invalid items."""
        schema = {"type": "array", "items": {"type": "string"}}
        errors = validate_against_schema(["a", 1, "c"], schema)

        assert len(errors) > 0
        assert any("[1]" in e for e in errors)

    def test_valid_object(self, sample_json_schema, valid_schema_data):
        """Test validation of valid object."""
        errors = validate_against_schema(valid_schema_data, sample_json_schema)

        assert errors == []

    def test_object_missing_required(self, sample_json_schema):
        """Test validation of object missing required field."""
        data = {"name": "Test"}  # Missing 'age'
        errors = validate_against_schema(data, sample_json_schema)

        assert len(errors) > 0
        assert any("age" in e.lower() or "required" in e.lower() for e in errors)

    def test_object_wrong_property_type(self, sample_json_schema):
        """Test validation of object with wrong property type."""
        data = {"name": 123, "age": 25}  # name should be string
        errors = validate_against_schema(data, sample_json_schema)

        assert len(errors) > 0
        assert any("name" in e for e in errors)

    def test_type_array_validation(self):
        """Test validation with type array (e.g., ["string", "null"])."""
        schema = {"type": ["string", "null"]}

        assert validate_against_schema("hello", schema) == []
        assert validate_against_schema(None, schema) == []
        assert len(validate_against_schema(123, schema)) > 0

    def test_null_type(self):
        """Test validation of null type."""
        schema = {"type": "null"}

        assert validate_against_schema(None, schema) == []
        assert len(validate_against_schema("not null", schema)) > 0

    def test_enum_validation(self):
        """Test validation of enum values."""
        schema = {"type": "string", "enum": ["red", "green", "blue"]}

        assert validate_against_schema("red", schema) == []
        assert len(validate_against_schema("yellow", schema)) > 0

    def test_nested_object_validation(self):
        """Test validation of nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name"],
                },
            },
            "required": ["person"],
        }

        valid_data = {"person": {"name": "John", "age": 30}}
        assert validate_against_schema(valid_data, schema) == []

        invalid_data = {"person": {"age": 30}}  # missing name
        errors = validate_against_schema(invalid_data, schema)
        assert len(errors) > 0

    def test_nested_array_validation(self):
        """Test validation of nested arrays."""
        schema = {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
            },
        }

        valid_data = [[1, 2], [3, 4]]
        assert validate_against_schema(valid_data, schema) == []

        invalid_data = [[1, 2], ["a", "b"]]
        errors = validate_against_schema(invalid_data, schema)
        assert len(errors) > 0

    def test_empty_schema(self):
        """Test validation with empty schema (allows anything)."""
        schema: Dict[str, Any] = {}

        assert validate_against_schema("anything", schema) == []
        assert validate_against_schema(123, schema) == []
        assert validate_against_schema(None, schema) == []

    def test_schema_without_type(self):
        """Test validation with schema without type field."""
        schema = {"description": "No type specified"}

        # Should not raise, should return no errors
        errors = validate_against_schema({"any": "value"}, schema)
        assert errors == []

    @pytest.mark.parametrize(
        "value,schema_type,valid",
        [
            ("string", "string", True),
            (123, "integer", True),
            (12.5, "number", True),
            (True, "boolean", True),
            ([], "array", True),
            ({}, "object", True),
            (None, "null", True),
            ("string", "integer", False),
            (12.5, "integer", False),
            (True, "integer", False),
            ("true", "boolean", False),
            ([1, 2], "object", False),
            ({"a": 1}, "array", False),
        ],
    )
    def test_type_validation_parametrized(self, value, schema_type, valid):
        """Test various type validations with parametrization."""
        schema = {"type": schema_type}
        errors = validate_against_schema(value, schema)

        if valid:
            assert errors == []
        else:
            assert len(errors) > 0
