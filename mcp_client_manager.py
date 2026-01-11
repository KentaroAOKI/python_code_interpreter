from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Mapping, Sequence

import anyio
from contextlib import asynccontextmanager

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version


logger = logging.getLogger("pycodei.mcp")

MAX_TOOL_NAME_LENGTH = 64
DEFAULT_TRANSPORT = "stdio"
DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERRORS = "strict"


def _load_distribution_version() -> str:
    """Load the version of the 'pycodei' distribution, or return a default if not found."""
    try:
        return version("pycodei")
    except PackageNotFoundError:
        return "0.0.0"


def _looks_like_path(value: str) -> bool:
    """Heuristic to determine if a string looks like a file system path."""
    # Unix absolute path
    if value.startswith(("~", ".", "/")):
        return True
    # Windows absolute path
    if os.name == "nt" and (value.startswith("\\") or (len(value) > 1 and value[1] == ":")):
        return True
    # Contains path separators
    return os.path.sep in value or (os.path.altsep and os.path.altsep in value)


def _expand_path(value: str, base_dir: Path | None) -> str:
    """Expand a file system path, resolving relative paths against a base directory if provided."""
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded) or base_dir is None:
        return expanded
    if _looks_like_path(expanded):
        return str((base_dir / expanded).resolve())
    return expanded


def _extract_env(raw_env: Mapping[str, Any] | None) -> dict[str, str]:
    """Extract environment variables from a raw mapping, ensuring all values are strings."""
    if not raw_env:
        return {}
    # Extract and convert to strings
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        if not isinstance(key, str):
            continue
        if value is None:
            continue
        env[key] = str(value)
    return env


def _safe_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    """Ensure the schema is a mapping; otherwise, return a default empty object schema."""
    if isinstance(schema, Mapping):
        return dict(schema)
    return {"type": "object", "properties": {}}


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    transport: str = DEFAULT_TRANSPORT
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    encoding: str = DEFAULT_ENCODING
    encoding_errors: str = DEFAULT_ENCODING_ERRORS
    disabled: bool = False

    def stdio_parameters(self) -> StdioServerParameters:
        if not self.command:
            raise ValueError(f"MCP server '{self.name}' is missing a command for stdio transport.")
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env or None,
            cwd=self.cwd,
            encoding=self.encoding,
            encoding_error_handler=self.encoding_errors,  # type: ignore[arg-type]
        )


@dataclass
class MCPToolBinding:
    server: MCPServerConfig
    tool_name: str
    description: str | None
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    function_name: str = ""


class MCPClientManager:
    """Loads MCP server definitions and exposes them as OpenAI-style tools."""

    def __init__(
        self,
        raw_servers: Mapping[str, Any] | None,
        base_dir: Path | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.client_info = types.Implementation(name="pycodei", version=_load_distribution_version())
        self._servers = self._parse_servers(raw_servers or {})
        self._openai_tools: list[dict[str, Any]] | None = None
        self._function_map: dict[str, Callable[[Any, Any], str]] | None = None
        self._bindings: dict[str, MCPToolBinding] = {}

    @property
    def has_servers(self) -> bool:
        return any(not cfg.disabled for cfg in self._servers.values())

    def get_openai_tools(self) -> tuple[list[dict[str, Any]], dict[str, Callable[[Any, Any], str]]]:
        if not self.has_servers:
            return [], {}
        if self._openai_tools is None or self._function_map is None:
            self._build_tool_cache()
        return self._openai_tools or [], self._function_map or {}

    def _parse_servers(self, raw_servers: Mapping[str, Any]) -> dict[str, MCPServerConfig]:
        servers: dict[str, MCPServerConfig] = {}
        for name, raw_config in raw_servers.items():
            if not isinstance(raw_config, Mapping):
                logger.warning("Skipping MCP server '%s': expected an object.", name)
                continue

            transport = str(raw_config.get("transport") or raw_config.get("type") or DEFAULT_TRANSPORT).lower()
            disabled = bool(raw_config.get("disabled", False))
            command = raw_config.get("command")
            url = raw_config.get("url")

            cwd_value = raw_config.get("cwd")
            args_value = raw_config.get("args")
            env_value = raw_config.get("env")
            headers_value = raw_config.get("headers")
            encoding = raw_config.get("encoding") or DEFAULT_ENCODING
            encoding_errors = raw_config.get("encoding_error_handler") or raw_config.get("encodingErrorHandler") or DEFAULT_ENCODING_ERRORS

            if isinstance(command, str) and _looks_like_path(command):
                command = _expand_path(command, self.base_dir)
            if isinstance(cwd_value, str):
                cwd_value = _expand_path(cwd_value, self.base_dir)
            if isinstance(url, str):
                url = url.strip()

            args_list: list[str] = []
            if isinstance(args_value, Sequence) and not isinstance(args_value, (str, bytes)):
                args_list = [str(item) for item in args_value]

            headers = _extract_env(headers_value if isinstance(headers_value, Mapping) else None)
            env = _extract_env(env_value if isinstance(env_value, Mapping) else None)

            server_config = MCPServerConfig(
                name=name,
                transport=transport,
                command=command if isinstance(command, str) else None,
                args=args_list,
                env=env,
                cwd=cwd_value if isinstance(cwd_value, str) else None,
                url=url if isinstance(url, str) and url else None,
                headers=headers,
                encoding=str(encoding),
                encoding_errors=str(encoding_errors),
                disabled=disabled,
            )

            if server_config.disabled:
                logger.info("MCP server '%s' is disabled in configuration.", name)
                servers[name] = server_config
                continue

            if server_config.transport == "stdio" and not server_config.command:
                logger.warning("Skipping MCP server '%s': stdio transport requires a command.", name)
                continue

            if server_config.transport in {"sse", "http", "https"} and not server_config.url:
                logger.warning("Skipping MCP server '%s': SSE transport requires a URL.", name)
                continue

            if server_config.transport in {"websocket", "ws", "wss"} and not server_config.url:
                logger.warning("Skipping MCP server '%s': WebSocket transport requires a URL.", name)
                continue

            servers[name] = server_config
        return servers

    def _build_tool_cache(self) -> None:
        bindings: list[MCPToolBinding] = []
        for server_config in self._servers.values():
            if server_config.disabled:
                continue
            try:
                tools = anyio.run(partial(self._list_tools_async, server_config))
            except Exception as exc:  # pragma: no cover - network/process errors
                logger.warning("Unable to load tools from MCP server '%s': %s", server_config.name, exc)
                continue

            for tool in tools:
                binding = MCPToolBinding(
                    server=server_config,
                    tool_name=tool.name,
                    description=tool.description,
                    input_schema=_safe_schema(tool.inputSchema),
                    output_schema=tool.outputSchema if isinstance(tool.outputSchema, Mapping) else None,
                )
                bindings.append(binding)

        self._assign_function_names(bindings)
        self._bindings = {binding.function_name: binding for binding in bindings}
        self._openai_tools = [
            self._tool_binding_to_openai_spec(binding) for binding in bindings
        ]
        self._function_map = {
            binding.function_name: self._create_callable(binding) for binding in bindings
        }

    def _assign_function_names(self, bindings: list[MCPToolBinding]) -> None:
        used: set[str] = set()
        for binding in bindings:
            base = self._sanitize_name(f"{binding.server.name}__{binding.tool_name}")
            if not base:
                base = "mcp_tool"
            if len(base) > MAX_TOOL_NAME_LENGTH:
                base = base[:MAX_TOOL_NAME_LENGTH]

            name = base
            counter = 2
            while name in used:
                suffix = f"_{counter}"
                name = f"{base[: MAX_TOOL_NAME_LENGTH - len(suffix)]}{suffix}"
                counter += 1
            binding.function_name = name
            used.add(name)

    @staticmethod
    def _sanitize_name(value: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", value)
        return sanitized.strip("_")

    def _tool_binding_to_openai_spec(self, binding: MCPToolBinding) -> dict[str, Any]:
        description = binding.description or (
            f"Tool '{binding.tool_name}' exposed by MCP server '{binding.server.name}'."
        )
        return {
            "type": "function",
            "function": {
                "name": binding.function_name,
                "description": description,
                "parameters": binding.input_schema,
            },
        }

    def _create_callable(self, binding: MCPToolBinding) -> Callable[[str, Any], str]:
        def _call(function_arguments: str | dict[str, Any], _messages: Any) -> str:
            try:
                arguments = self._parse_arguments(function_arguments)
            except ValueError as exc:
                return f"Invalid arguments for MCP tool '{binding.function_name}': {exc}"

            try:
                result = anyio.run(
                    partial(self._call_tool_async, binding.server, binding.tool_name, arguments)
                )
            except Exception as exc:  # pragma: no cover - depends on external servers
                logger.error(
                    "Error invoking MCP tool '%s' on server '%s': %s",
                    binding.tool_name,
                    binding.server.name,
                    exc,
                )
                return f"Failed to invoke MCP tool '{binding.function_name}': {exc}"

            return self._format_tool_result(binding, result)

        return _call

    @staticmethod
    def _parse_arguments(raw_arguments: str | dict[str, Any]) -> dict[str, Any] | None:
        if raw_arguments is None:
            return None
        if isinstance(raw_arguments, Mapping):
            return dict(raw_arguments)
        if not isinstance(raw_arguments, str):
            raise ValueError("expected JSON object string or dict")
        if not raw_arguments.strip():
            return None
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not decode JSON: {exc}") from exc
        if parsed is None:
            return None
        if not isinstance(parsed, Mapping):
            raise ValueError("tool arguments must decode to an object")
        return dict(parsed)

    def _format_tool_result(self, binding: MCPToolBinding, result: types.CallToolResult) -> str:
        payload: dict[str, Any] = {
            "server": binding.server.name,
            "tool": binding.tool_name,
            "is_error": result.isError,
        }
        if result.content:
            payload["content"] = [self._content_block_to_python(block) for block in result.content]
        if result.structuredContent is not None:
            payload["structured_content"] = result.structuredContent
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _content_block_to_python(block: Any) -> Any:
        if hasattr(block, "model_dump"):
            return block.model_dump(exclude_none=True)
        if isinstance(block, Mapping):
            return dict(block)
        return block

    async def _list_tools_async(self, server_config: MCPServerConfig) -> list[types.Tool]:
        async with self._session(server_config) as session:
            tools: list[types.Tool] = []
            cursor: str | None = None
            while True:
                params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
                request = types.ClientRequest(types.ListToolsRequest(params=params))
                result = await session.send_request(request, types.ListToolsResult)
                tools.extend(result.tools)
                cursor = result.nextCursor
                if not cursor:
                    break
            return tools

    async def _call_tool_async(
        self,
        server_config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> types.CallToolResult:
        async with self._session(server_config) as session:
            request = types.ClientRequest(
                types.CallToolRequest(
                    params=types.CallToolRequestParams(name=tool_name, arguments=arguments),
                )
            )
            return await session.send_request(request, types.CallToolResult)

    @asynccontextmanager
    async def _session(
        self,
        server_config: MCPServerConfig,
    ) -> AsyncGenerator[
        ClientSession,
        None,
    ]:
        async with self._transport_stream(server_config) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                client_info=self.client_info,
            ) as session:
                await session.initialize()
                yield session

    @asynccontextmanager
    async def _transport_stream(
        self,
        server_config: MCPServerConfig,
    ) -> Any:
        transport = server_config.transport.lower()
        if transport in {"stdio"}:
            stdio_params = server_config.stdio_parameters()
            async with stdio_client(stdio_params) as streams:
                yield streams
        elif transport in {"sse", "http", "https"}:
            if not server_config.url:
                raise ValueError(f"MCP server '{server_config.name}' is missing a URL for SSE transport.")
            async with sse_client(server_config.url, headers=server_config.headers) as streams:
                yield streams
        elif transport in {"websocket", "ws", "wss"}:
            if not server_config.url:
                raise ValueError(f"MCP server '{server_config.name}' is missing a URL for WebSocket transport.")
            from mcp.client.websocket import websocket_client  # Imported lazily

            async with websocket_client(server_config.url) as streams:
                yield streams
        else:
            raise ValueError(f"Unsupported MCP transport '{transport}' for server '{server_config.name}'.")
