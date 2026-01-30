"""Tool executor for managing tool execution with error handling and logging."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for tool execution.

    Attributes:
        timeout: Maximum execution time in seconds (None for no timeout).
        max_concurrent: Maximum concurrent tool executions.
        retry_on_error: Whether to retry failed executions.
        max_retries: Maximum number of retries.
        validate_args: Whether to validate arguments before execution.
    """

    timeout: Optional[float] = 30.0
    max_concurrent: int = 10
    retry_on_error: bool = False
    max_retries: int = 3
    validate_args: bool = True


@dataclass
class ExecutionResult:
    """Result of a tool execution with metadata.

    Attributes:
        tool_name: Name of the executed tool.
        tool_call_id: ID of the tool call (for LLM response tracking).
        result: The ToolResult from execution.
        execution_time: Time taken to execute in seconds.
        retries: Number of retries performed.
    """

    tool_name: str
    tool_call_id: str
    result: ToolResult
    execution_time: float = 0.0
    retries: int = 0


class ToolExecutor:
    """Executor for running tools with error handling and concurrency control.

    Provides a managed way to execute tools with:
    - Timeout handling
    - Concurrency limits
    - Retry logic
    - Execution logging
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        config: Optional[ExecutionConfig] = None,
    ):
        """Initialize the executor.

        Args:
            registry: Tool registry to use. Creates a new one if not provided.
            config: Execution configuration. Uses defaults if not provided.
        """
        self.registry = registry or ToolRegistry()
        self.config = config or ExecutionConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def execute(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Dict[str, Any],
    ) -> ExecutionResult:
        """Execute a single tool.

        Args:
            tool_name: Name of the tool to execute.
            tool_call_id: ID for tracking the tool call.
            arguments: Arguments to pass to the tool.

        Returns:
            ExecutionResult with the result and metadata.
        """
        import time

        start_time = time.monotonic()
        retries = 0

        async with self._semaphore:
            while True:
                try:
                    result = await self._execute_with_timeout(tool_name, arguments)
                    execution_time = time.monotonic() - start_time

                    logger.debug(
                        f"Tool '{tool_name}' executed successfully "
                        f"in {execution_time:.3f}s"
                    )

                    return ExecutionResult(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        result=result,
                        execution_time=execution_time,
                        retries=retries,
                    )

                except asyncio.TimeoutError:
                    execution_time = time.monotonic() - start_time
                    logger.warning(
                        f"Tool '{tool_name}' timed out after {self.config.timeout}s"
                    )
                    return ExecutionResult(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        result=ToolResult(
                            success=False,
                            error=f"Tool execution timed out after {self.config.timeout}s",
                        ),
                        execution_time=execution_time,
                        retries=retries,
                    )

                except Exception as e:
                    if self.config.retry_on_error and retries < self.config.max_retries:
                        retries += 1
                        logger.warning(
                            f"Tool '{tool_name}' failed, retrying "
                            f"({retries}/{self.config.max_retries}): {e}"
                        )
                        await asyncio.sleep(0.5 * retries)  # Exponential backoff
                        continue

                    execution_time = time.monotonic() - start_time
                    logger.error(f"Tool '{tool_name}' failed: {e}")
                    return ExecutionResult(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        result=ToolResult(success=False, error=str(e)),
                        execution_time=execution_time,
                        retries=retries,
                    )

    async def _execute_with_timeout(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool with timeout handling.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments.

        Returns:
            ToolResult from the tool execution.

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout.
        """
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found in registry",
            )

        if self.config.timeout:
            return await asyncio.wait_for(
                tool.run(**arguments),
                timeout=self.config.timeout,
            )
        else:
            return await tool.run(**arguments)

    async def execute_many(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[ExecutionResult]:
        """Execute multiple tool calls concurrently.

        Args:
            tool_calls: List of tool call dicts with keys:
                - name: Tool name
                - id: Tool call ID
                - arguments: Dict of arguments

        Returns:
            List of ExecutionResults in the same order as inputs.
        """
        tasks = [
            self.execute(
                tool_name=call["name"],
                tool_call_id=call["id"],
                arguments=call.get("arguments", {}),
            )
            for call in tool_calls
        ]

        return await asyncio.gather(*tasks)

    async def execute_sequential(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[ExecutionResult]:
        """Execute tool calls sequentially (one at a time).

        Args:
            tool_calls: List of tool call dicts.

        Returns:
            List of ExecutionResults in order.
        """
        results = []
        for call in tool_calls:
            result = await self.execute(
                tool_name=call["name"],
                tool_call_id=call["id"],
                arguments=call.get("arguments", {}),
            )
            results.append(result)
        return results

    def register(self, tool: BaseTool) -> BaseTool:
        """Register a tool with the executor's registry.

        Args:
            tool: The tool to register.

        Returns:
            The registered tool.
        """
        return self.registry.register(tool)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool from the registry.

        Args:
            name: Tool name.

        Returns:
            The tool if found, None otherwise.
        """
        return self.registry.get(name)
