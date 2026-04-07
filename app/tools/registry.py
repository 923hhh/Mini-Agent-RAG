from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.retrievers.local_kb import search_local_knowledge_base
from app.schemas.chat import RetrievedReference, ToolDefinition
from app.services.settings import AppSettings
from app.storage.filters import MetadataFilters


@dataclass(frozen=True)
class ToolExecutionResult:
    output: str
    references: list[RetrievedReference]


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    executor: Callable[[AppSettings, dict[str, Any]], ToolExecutionResult]


def list_tools() -> list[ToolDefinition]:
    return [tool.definition for tool in _TOOL_REGISTRY.values()]


def tool_names() -> set[str]:
    return set(_TOOL_REGISTRY.keys())


def execute_tool(
    name: str,
    settings: AppSettings,
    arguments: dict[str, Any],
) -> ToolExecutionResult:
    tool = _TOOL_REGISTRY.get(name)
    if tool is None:
        raise ValueError(f"不支持的工具: {name}")
    return tool.executor(settings, arguments)


def _search_local_knowledge(
    settings: AppSettings,
    arguments: dict[str, Any],
) -> ToolExecutionResult:
    knowledge_base_name = str(arguments.get("knowledge_base_name", "")).strip()
    query = str(arguments.get("query", "")).strip()
    top_k = int(arguments.get("top_k", settings.kb.VECTOR_SEARCH_TOP_K))
    score_threshold = float(arguments.get("score_threshold", settings.kb.SCORE_THRESHOLD))
    raw_metadata_filters = arguments.get("metadata_filters")
    metadata_filters = (
        MetadataFilters.model_validate(raw_metadata_filters)
        if raw_metadata_filters
        else None
    )

    if not knowledge_base_name:
        raise ValueError("search_local_knowledge 需要 knowledge_base_name。")
    if not query:
        raise ValueError("search_local_knowledge 需要 query。")

    references = search_local_knowledge_base(
        settings=settings,
        knowledge_base_name=knowledge_base_name,
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        metadata_filters=metadata_filters,
    )
    if references:
        top_ref = references[0]
        preview = top_ref.content_preview.replace("\n", " ").strip()
        output = (
            f"命中 {len(references)} 条知识片段。"
            f"首条来源: {top_ref.source}。"
            f"预览: {preview}"
        )
    else:
        output = "未检索到满足阈值的知识片段。"
    return ToolExecutionResult(output=output, references=references)


def _calculate(
    settings: AppSettings,
    arguments: dict[str, Any],
) -> ToolExecutionResult:
    del settings

    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        raise ValueError("calculate 需要 expression。")
    if len(expression) > 120:
        raise ValueError("表达式过长，请缩短后重试。")

    value = _safe_eval(expression)
    if isinstance(value, float) and value.is_integer():
        value = int(value)

    return ToolExecutionResult(
        output=f"{expression} = {value}",
        references=[],
    )


def _current_time(
    settings: AppSettings,
    arguments: dict[str, Any],
) -> ToolExecutionResult:
    del settings

    timezone_name = str(arguments.get("timezone", "")).strip()
    if timezone_name:
        try:
            current = datetime.now(ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"不支持的时区: {timezone_name}") from exc
    else:
        current = datetime.now().astimezone()

    formatted = current.strftime("%Y-%m-%d %H:%M:%S %Z%z").strip()
    return ToolExecutionResult(output=formatted, references=[])


def _safe_eval(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _eval_node(node.operand)
        return _UNARY_OPERATORS[type(node.op)](operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise ValueError("表达式中包含不支持的语法。")


_BINARY_OPERATORS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.AST], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_TOOL_REGISTRY: dict[str, RegisteredTool] = {
    "search_local_knowledge": RegisteredTool(
        definition=ToolDefinition(
            name="search_local_knowledge",
            description="在指定 knowledge_base_name 内检索本地知识库，适合回答文档内容、概念说明和项目资料问题。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要在知识库中检索的问题。",
                    },
                    "knowledge_base_name": {
                        "type": "string",
                        "description": "目标知识库名称，例如 phase2_demo。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的候选片段数量。",
                        "default": 4,
                    },
                    "score_threshold": {
                        "type": "number",
                        "description": "相关度阈值，范围 0 到 1。",
                        "default": 0.5,
                    },
                    "metadata_filters": {
                        "type": "object",
                        "description": "可选元数据过滤条件，用于按 source、page、title、Header1 等字段过滤。",
                    },
                },
                "required": ["query", "knowledge_base_name"],
            },
        ),
        executor=_search_local_knowledge,
    ),
    "calculate": RegisteredTool(
        definition=ToolDefinition(
            name="calculate",
            description="计算数学表达式，支持括号、加减乘除、整除、取模和幂运算。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "例如 12*(3+4) 或 (18-6)/3。",
                    },
                },
                "required": ["expression"],
            },
        ),
        executor=_calculate,
    ),
    "current_time": RegisteredTool(
        definition=ToolDefinition(
            name="current_time",
            description="返回当前时间，可选指定 IANA 时区名称；未指定时使用当前系统时区。",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "可选时区，如 Asia/Shanghai。",
                    },
                },
                "required": [],
            },
        ),
        executor=_current_time,
    ),
}
