# Tools API Reference

API reference cho Tool System.

## @tool Decorator

Decorator để định nghĩa tools.

### Usage

```python
from agents_framework.tools.base import tool

@tool(name="calculator", description="Thực hiện phép tính")
def calculator(expression: str) -> str:
    """
    Tính toán biểu thức số học.

    Args:
        expression: Biểu thức cần tính (vd: "2 + 3 * 4")

    Returns:
        Kết quả dạng string
    """
    return str(eval(expression))
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tên tool (unique) |
| `description` | `str` | Mô tả tool cho LLM |

### Auto Schema Generation

Decorator tự động generate JSON schema từ type hints:

```python
@tool(name="search", description="Search the web")
def search(query: str, max_results: int = 10) -> str:
    ...

# Generated schema:
# {
#   "type": "object",
#   "properties": {
#     "query": {"type": "string"},
#     "max_results": {"type": "integer", "default": 10}
#   },
#   "required": ["query"]
# }
```

## BaseTool

Base class cho tools.

### Class Definition

```python
from agents_framework.tools.base import BaseTool, ToolResult

class BaseTool:
    name: str
    description: str
    parameters: Dict[str, Any]

    async def run(self, **kwargs) -> ToolResult:
        ...
```

### ToolResult

```python
@dataclass
class ToolResult:
    output: Any           # Kết quả
    success: bool = True  # Thành công?
    error: str = None     # Error message nếu có
```

### Custom Tool Class

```python
from agents_framework.tools.base import BaseTool, ToolResult

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get weather for a location"
    parameters = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    }

    async def run(self, location: str, unit: str = "celsius") -> ToolResult:
        # Call weather API
        weather = await self.fetch_weather(location, unit)
        return ToolResult(output=weather)
```

## ToolRegistry

Quản lý collection of tools.

### Class Definition

```python
from agents_framework.tools.registry import ToolRegistry

class ToolRegistry:
    def __init__(self):
        ...
```

### Methods

#### register(tool: BaseTool)

Đăng ký tool.

```python
registry = ToolRegistry()
registry.register(calculator)
registry.register(weather_tool)
```

#### get(name: str) -> BaseTool

Lấy tool theo tên.

```python
calc = registry.get("calculator")
```

#### list() -> List[str]

Liệt kê tất cả tool names.

```python
names = registry.list()
# ["calculator", "weather"]
```

#### to_definitions() -> List[ToolDefinition]

Export sang format cho LLM.

```python
definitions = registry.to_definitions()
response = await provider.generate(messages, tools=definitions)
```

#### unregister(name: str)

Xóa tool.

```python
registry.unregister("calculator")
```

## ToolExecutor

Thực thi tools với error handling.

### Class Definition

```python
from agents_framework.tools.executor import ToolExecutor

class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        ...
```

### Methods

#### execute(name: str, **kwargs) -> ToolResult

Thực thi tool.

```python
executor = ToolExecutor(registry, timeout=10.0)
result = await executor.execute("calculator", expression="2 + 3")

if result.success:
    print(f"Result: {result.output}")
else:
    print(f"Error: {result.error}")
```

#### execute_batch(calls: List[ToolCall]) -> List[ToolResult]

Thực thi nhiều tools.

```python
results = await executor.execute_batch([
    ToolCall(id="1", name="calc", arguments={"expression": "2+2"}),
    ToolCall(id="2", name="weather", arguments={"location": "Hanoi"}),
])
```

## Built-in Tools

### FileTools

```python
from agents_framework.tools.builtin.file import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)

registry.register(ReadFileTool())
registry.register(WriteFileTool())
```

### WebTools

```python
from agents_framework.tools.builtin.web import (
    FetchURLTool,
    SearchTool,
)

registry.register(FetchURLTool())
registry.register(SearchTool(api_key="..."))
```

### ShellTools

```python
from agents_framework.tools.builtin.shell import ShellTool

registry.register(ShellTool(
    allowed_commands=["ls", "cat", "grep"],
    working_directory="/safe/path",
))
```

## Tool with Async Support

```python
@tool(name="fetch_data", description="Fetch data from API")
async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

## Tool with Validation

```python
from pydantic import BaseModel, validator

class SearchParams(BaseModel):
    query: str
    limit: int = 10

    @validator("limit")
    def limit_range(cls, v):
        if v < 1 or v > 100:
            raise ValueError("limit must be 1-100")
        return v

@tool(name="search", description="Search with validation")
def search(params: SearchParams) -> str:
    return f"Searching: {params.query}, limit: {params.limit}"
```

## Example: Complete Tool Setup

```python
from agents_framework.tools.base import tool, BaseTool, ToolResult
from agents_framework.tools.registry import ToolRegistry
from agents_framework.tools.executor import ToolExecutor

# Decorator-based tool
@tool(name="greet", description="Greet someone")
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Class-based tool
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform calculations"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
        },
        "required": ["expression"],
    }

    async def run(self, expression: str) -> ToolResult:
        try:
            result = eval(expression)
            return ToolResult(output=str(result))
        except Exception as e:
            return ToolResult(output=None, success=False, error=str(e))

# Setup
registry = ToolRegistry()
registry.register(greet)
registry.register(CalculatorTool())

executor = ToolExecutor(registry, timeout=5.0)

# Execute
result = await executor.execute("greet", name="World")
print(result.output)  # Hello, World!

result = await executor.execute("calculator", expression="2 ** 10")
print(result.output)  # 1024
```
