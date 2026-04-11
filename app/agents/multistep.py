from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.api.streaming import iter_text_chunks
from app.chains.rag import convert_history, generate_rag_answer, stream_rag_answer
from app.schemas.chat import (
    AgentChatRequest,
    AgentChatResponse,
    AgentStepRecord,
    MemoryOverview,
    RetrievedReference,
    ToolCallRecord,
)
from app.services.llm_service import build_chat_model
from app.services.memory_service import (
    agent_memory_enabled,
    persist_agent_turns,
    retrieve_agent_memory,
    sanitize_session_id,
)
from app.services.reference_overview import build_reference_overview
from app.services.settings import AppSettings
from app.services.streaming_llm import stream_prompt_output
from app.tools.registry import (
    ToolExecutionResult,
    build_langchain_tool_schemas,
    execute_tool,
    resolve_tool_definitions,
    tool_names,
)
from app.utils.text import deduplicate_strings


AGENT_DIRECT_SYSTEM_PROMPT = """你是一个简洁、准确的中文智能体助手。
当问题不需要调用工具时，直接回答即可。
若问题信息不足，明确说明缺少什么信息，不要编造。"""

AGENT_TOOL_PLANNING_SYSTEM_PROMPT = """你是一个负责规划下一步工具调用的中文智能体。
你的任务不是直接展开长答案，而是判断“下一步是否需要调用一个工具”。

请严格遵守以下规则：
1. 每一轮最多调用一个工具；如果当前已足够回答，就不要调用工具。
2. 如果 `knowledge_base_name` 已提供，且问题属于概念说明、文档问答、配置参数、项目资料检索，优先调用 `search_local_knowledge`，不要直接用常识跳过知识库。
3. 如果问题明显是在求数学结果，且已经有明确表达式或已有检索结果里给出了足够数字，再调用 `calculate`。
4. 如果问题明确要求“当前时间 / 今天日期 / 现在几点”，调用 `current_time`。
5. 不要重复调用与已执行记录完全相同的工具和参数。
6. 如果缺少某个工具所需参数，就不要硬调用。
7. 如果用户要求“先查再算”，在检索出足够数字后，下一步应优先调用 `calculate`，而不是停在检索结果。

如果需要工具，请只返回工具调用。
如果不需要继续调用工具，请直接返回一小句说明。"""

_TIME_KEYWORDS = (
    "当前时间",
    "现在时间",
    "现在几点",
    "几点了",
    "时间是多少",
    "今天几号",
    "当前日期",
    "今天日期",
    "date",
    "time",
)

_KNOWLEDGE_HINTS = (
    "知识库",
    "文档",
    "资料",
    "文件",
    "配置",
    "参数",
    "规则",
    "说明",
    "查",
    "检索",
    "根据",
)

_CALC_HINTS = (
    "计算",
    "算出",
    "求出",
    "结果",
    "总和",
    "相加",
    "加起来",
    "合计",
    "一共",
    "相减",
    "减去",
    "差值",
    "乘积",
    "相乘",
    "乘起来",
    "除以",
    "相除",
    "平均",
)


@dataclass(frozen=True)
class PlannedToolCall:
    tool_name: str
    arguments: dict[str, Any]
    reason: str


@dataclass
class ExecutedToolRecord:
    tool_name: str
    result: ToolExecutionResult


@dataclass(frozen=True)
class ToolPlanningDecision:
    plan: PlannedToolCall | None
    used_llm: bool


@dataclass
class AgentExecutionState:
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    steps: list[AgentStepRecord] = field(default_factory=list)
    references: list[RetrievedReference] = field(default_factory=list)
    executed_tools: list[ExecutedToolRecord] = field(default_factory=list)
    executed_signatures: set[str] = field(default_factory=set)
    stop_reason: str = ""


def run_agent(
    settings: AppSettings,
    request: AgentChatRequest,
) -> AgentChatResponse:
    validate_agent_request(request)
    session_id = sanitize_session_id(request.session_id)
    memory_context = ""
    memory_overview: MemoryOverview | None = None
    if session_id and agent_memory_enabled(settings):
        bundle = retrieve_agent_memory(
            settings,
            session_id=session_id,
            query=request.query,
        )
        memory_context = bundle.text
        memory_overview = bundle.overview

    state = execute_agent_plan(
        settings=settings,
        request=request,
        memory_context=memory_context,
    )
    answer = build_agent_answer(
        settings=settings,
        request=request,
        state=state,
        memory_context=memory_context,
    )
    append_final_step(state, answer)
    if session_id and agent_memory_enabled(settings):
        persist_agent_turns(
            settings,
            session_id=session_id,
            user_text=request.query,
            assistant_text=answer,
            tools_used=[
                item.tool_name for item in state.tool_calls if item.status == "success"
            ],
        )
    return AgentChatResponse(
        answer=answer,
        tool_calls=state.tool_calls,
        steps=state.steps,
        references=state.references,
        reference_overview=build_reference_overview(state.references),
        knowledge_base_name=request.knowledge_base_name,
        used_tools=bool(state.tool_calls),
        stream=request.stream,
        session_id=session_id,
        memory_overview=memory_overview,
    )


def stream_agent_events(
    settings: AppSettings,
    request: AgentChatRequest,
) -> Iterator[dict[str, Any]]:
    validate_agent_request(request)
    session_id = sanitize_session_id(request.session_id)
    memory_context = ""
    memory_overview: MemoryOverview | None = None
    if session_id and agent_memory_enabled(settings):
        bundle = retrieve_agent_memory(
            settings,
            session_id=session_id,
            query=request.query,
        )
        memory_context = bundle.text
        memory_overview = bundle.overview

    emitted_events: list[dict[str, Any]] = []
    state = execute_agent_plan(
        settings=settings,
        request=request,
        emit=emitted_events.append,
        memory_context=memory_context,
    )
    for payload in emitted_events:
        yield payload

    answer_parts: list[str] = []
    for delta in stream_agent_answer(
        settings=settings,
        request=request,
        state=state,
        memory_context=memory_context,
    ):
        answer_parts.append(delta)
        yield {"type": "token", "delta": delta}

    answer = "".join(answer_parts)
    append_final_step(state, answer)
    if session_id and agent_memory_enabled(settings):
        persist_agent_turns(
            settings,
            session_id=session_id,
            user_text=request.query,
            assistant_text=answer,
            tools_used=[
                item.tool_name for item in state.tool_calls if item.status == "success"
            ],
        )
    done_body: dict[str, Any] = {
        "answer": answer,
        "tool_calls": [item.model_dump() for item in state.tool_calls],
        "steps": [item.model_dump() for item in state.steps],
        "references": [item.model_dump() for item in state.references],
        "reference_overview": build_reference_overview(state.references).model_dump(),
        "knowledge_base_name": request.knowledge_base_name,
        "used_tools": bool(state.tool_calls),
        "stream": True,
        "session_id": session_id,
    }
    if memory_overview is not None:
        done_body["memory_overview"] = memory_overview.model_dump()
    yield {"type": "done", **done_body}


def execute_agent_plan(
    settings: AppSettings,
    request: AgentChatRequest,
    emit: Callable[[dict[str, Any]], None] | None = None,
    memory_context: str = "",
) -> AgentExecutionState:
    state = AgentExecutionState()

    for step_index in range(1, request.max_steps + 1):
        plan = select_next_tool_call(settings, request, state, memory_context=memory_context)
        if plan is None:
            break

        signature = build_plan_signature(plan)
        if signature in state.executed_signatures:
            stop_step = AgentStepRecord(
                step_index=step_index,
                kind="stop",
                status="stopped",
                summary=f"检测到重复工具调用 `{plan.tool_name}`，已提前终止以避免死循环。",
                tool_name=plan.tool_name,
                arguments=plan.arguments,
            )
            state.steps.append(stop_step)
            state.stop_reason = stop_step.summary
            emit_step(emit, stop_step)
            return state

        state.executed_signatures.add(signature)
        try:
            tool_result = execute_tool(
                name=plan.tool_name,
                settings=settings,
                arguments=plan.arguments,
            )
            tool_call = ToolCallRecord(
                step_index=step_index,
                tool_name=plan.tool_name,
                arguments=plan.arguments,
                output=tool_result.output,
                status="success",
            )
            state.tool_calls.append(tool_call)
            state.executed_tools.append(
                ExecutedToolRecord(tool_name=plan.tool_name, result=tool_result)
            )
            state.references = merge_references(state.references, tool_result.references)

            step = AgentStepRecord(
                step_index=step_index,
                kind="tool",
                status="success",
                summary=build_tool_step_summary(plan, tool_result),
                tool_name=plan.tool_name,
                arguments=plan.arguments,
                output=tool_result.output,
            )
            state.steps.append(step)
            emit_tool_call(emit, tool_call)
            emit_step(emit, step)
            emit_references(emit, tool_result.references)
        except Exception as exc:
            tool_call = ToolCallRecord(
                step_index=step_index,
                tool_name=plan.tool_name,
                arguments=plan.arguments,
                output="",
                status="error",
                error_message=str(exc),
            )
            state.tool_calls.append(tool_call)
            step = AgentStepRecord(
                step_index=step_index,
                kind="tool",
                status="error",
                summary=f"工具 `{plan.tool_name}` 执行失败：{exc}",
                tool_name=plan.tool_name,
                arguments=plan.arguments,
                output="",
            )
            state.steps.append(step)
            state.stop_reason = step.summary
            emit_tool_call(emit, tool_call)
            emit_step(emit, step)
            return state

    if len(state.tool_calls) >= request.max_steps:
        next_plan = select_next_tool_call(settings, request, state, memory_context=memory_context)
        if next_plan is not None:
            stop_step = AgentStepRecord(
                step_index=request.max_steps + 1,
                kind="stop",
                status="stopped",
                summary=f"已达到最大工具步数 {request.max_steps}，终止后续工具调用。",
                tool_name=next_plan.tool_name,
                arguments=next_plan.arguments,
            )
            state.steps.append(stop_step)
            state.stop_reason = stop_step.summary
            emit_step(emit, stop_step)

    return state


def select_next_tool_call(
    settings: AppSettings,
    request: AgentChatRequest,
    state: AgentExecutionState,
    *,
    memory_context: str = "",
) -> PlannedToolCall | None:
    planning = select_next_tool_call_with_llm(
        settings, request, state, memory_context=memory_context
    )
    if planning.used_llm and planning.plan is not None:
        return planning.plan
    return select_next_tool_call_heuristic(request, state)


def select_next_tool_call_heuristic(
    request: AgentChatRequest,
    state: AgentExecutionState,
) -> PlannedToolCall | None:
    allowed = resolve_allowed_tools(request)
    query = request.query.strip()

    if (
        "search_local_knowledge" in allowed
        and request.knowledge_base_name
        and should_search_knowledge(query, state)
    ):
        return PlannedToolCall(
            tool_name="search_local_knowledge",
            arguments={
                "query": query,
                "knowledge_base_name": request.knowledge_base_name,
                "top_k": request.top_k,
                "score_threshold": request.score_threshold,
                **(
                    {"metadata_filters": request.metadata_filters.model_dump()}
                    if request.metadata_filters is not None
                    else {}
                ),
            },
            reason="需要先检索知识库上下文。",
        )

    if "current_time" in allowed and should_call_current_time(query, state):
        arguments: dict[str, Any] = {}
        timezone_name = extract_timezone_name(query)
        if timezone_name:
            arguments["timezone"] = timezone_name
        return PlannedToolCall(
            tool_name="current_time",
            arguments=arguments,
            reason="问题涉及当前时间信息。",
        )

    if "calculate" in allowed:
        expression = derive_calculate_expression(query, state.references)
        if expression and should_call_calculate(state):
            return PlannedToolCall(
                tool_name="calculate",
                arguments={"expression": expression},
                reason="已具备计算表达式，继续执行数学计算。",
            )

    return None


def select_next_tool_call_with_llm(
    settings: AppSettings,
    request: AgentChatRequest,
    state: AgentExecutionState,
    *,
    memory_context: str = "",
) -> ToolPlanningDecision:
    allowed = resolve_allowed_tools(request)
    plannable_tools = resolve_plannable_tools(request, allowed)
    if not plannable_tools:
        return ToolPlanningDecision(plan=None, used_llm=True)

    try:
        llm = build_chat_model(
            settings,
            model_name=settings.model.AGENT_MODEL,
            temperature=0.0,
        )
        bind_tools = getattr(llm, "bind_tools", None)
        if not callable(bind_tools):
            return ToolPlanningDecision(plan=None, used_llm=False)

        prompt = build_agent_tool_planning_prompt()
        variables = build_agent_tool_planning_variables(
            request=request,
            state=state,
            plannable_tools=plannable_tools,
            memory_context=memory_context,
        )
        response = bind_tools(plannable_tools, tool_choice="auto").invoke(prompt.invoke(variables))
    except Exception:
        return ToolPlanningDecision(plan=None, used_llm=False)

    tool_calls = extract_tool_calls_from_response(response)
    if not tool_calls:
        return ToolPlanningDecision(plan=None, used_llm=True)

    first_call = tool_calls[0]
    tool_name = str(first_call.get("name", "")).strip()
    arguments = coerce_tool_call_arguments(first_call.get("args"))
    if not tool_name or tool_name not in allowed:
        return ToolPlanningDecision(plan=None, used_llm=False)
    if tool_name == "search_local_knowledge" and not request.knowledge_base_name.strip():
        return ToolPlanningDecision(plan=None, used_llm=False)

    return ToolPlanningDecision(
        plan=PlannedToolCall(
            tool_name=tool_name,
            arguments=arguments,
            reason="由 LLM tool calling 规划得出下一步工具调用。",
        ),
        used_llm=True,
    )


def resolve_plannable_tools(
    request: AgentChatRequest,
    allowed: set[str],
) -> list[dict[str, Any]]:
    names = set(allowed)
    if not request.knowledge_base_name.strip():
        names.discard("search_local_knowledge")
    return build_langchain_tool_schemas(names)


def build_agent_tool_planning_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_TOOL_PLANNING_SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            (
                "human",
                "长期记忆摘要（可能为空）：\n{memory_context}\n\n"
                "用户问题：\n{query}\n\n"
                "knowledge_base_name：{knowledge_base_name}\n\n"
                "可用工具：\n{available_tools}\n\n"
                "已执行工具记录：\n{tool_history}\n\n"
                "当前已获得的上下文：\n{observation_context}\n\n"
                "如果还需要工具，请只调用一个最合适的工具；"
                "如果已经足够回答或当前无法继续获取有效信息，就直接返回一小句说明，不要再调用工具。",
            ),
        ]
    )


def build_agent_tool_planning_variables(
    *,
    request: AgentChatRequest,
    state: AgentExecutionState,
    plannable_tools: list[dict[str, Any]],
    memory_context: str = "",
) -> dict[str, object]:
    memory_block = memory_context.strip() if memory_context.strip() else "（无）"
    return {
        "history": convert_history(request.history),
        "memory_context": memory_block,
        "query": request.query,
        "knowledge_base_name": request.knowledge_base_name.strip() or "（未提供）",
        "available_tools": format_available_tools_for_planning(plannable_tools),
        "tool_history": format_tool_history_for_planning(state),
        "observation_context": build_agent_observation_context(state),
    }


def format_available_tools_for_planning(plannable_tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for tool in plannable_tools:
        function = tool.get("function", {})
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        description = str(function.get("description", "")).strip()
        parameters = function.get("parameters", {})
        required = parameters.get("required", []) if isinstance(parameters, dict) else []
        required_text = ", ".join(str(item) for item in required) if isinstance(required, list) else ""
        lines.append(
            f"- {name}: {description}"
            + (f" 必填参数: {required_text}" if required_text else "")
        )
    return "\n".join(lines) if lines else "（无可用工具）"


def format_tool_history_for_planning(state: AgentExecutionState) -> str:
    if not state.tool_calls:
        return "（尚未执行任何工具）"

    lines: list[str] = []
    for item in state.tool_calls[-4:]:
        lines.append(
            f"- step={item.step_index} tool={item.tool_name} status={item.status} "
            f"args={json.dumps(item.arguments, ensure_ascii=False, sort_keys=True)} "
            f"output={item.output or item.error_message or ''}"
        )
    return "\n".join(lines)


def build_agent_observation_context(state: AgentExecutionState) -> str:
    parts: list[str] = []
    if state.references:
        parts.append("知识库证据：")
        for reference in state.references[:4]:
            metadata_bits = [f"source={reference.source}"]
            if reference.section_title:
                metadata_bits.append(f"section={reference.section_title}")
            if reference.page is not None:
                metadata_bits.append(f"page={reference.page}")
            snippet = " ".join(reference.content.split())[:320]
            parts.append(f"- {' | '.join(metadata_bits)} | content={snippet}")
    if state.executed_tools:
        parts.append("工具输出：")
        for record in state.executed_tools[-4:]:
            parts.append(f"- {record.tool_name}: {record.result.output}")
    return "\n".join(parts) if parts else "（当前没有额外上下文）"


def extract_tool_calls_from_response(response: Any) -> list[dict[str, Any]]:
    raw_tool_calls = getattr(response, "tool_calls", None)
    if isinstance(raw_tool_calls, list):
        return [item for item in raw_tool_calls if isinstance(item, dict)]

    additional_kwargs = getattr(response, "additional_kwargs", None)
    if not isinstance(additional_kwargs, dict):
        return []
    tool_calls = additional_kwargs.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        function = item.get("function", {})
        if not isinstance(function, dict):
            continue
        normalized.append(
            {
                "name": function.get("name"),
                "args": function.get("arguments"),
            }
        )
    return normalized


def coerce_tool_call_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return dict(raw_arguments)
    if isinstance(raw_arguments, str):
        normalized = raw_arguments.strip()
        if not normalized:
            return {}
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return payload
    return {}


def resolve_allowed_tools(request: AgentChatRequest) -> set[str]:
    if request.allowed_tools:
        return set(request.allowed_tools)
    return tool_names()


def validate_agent_request(request: AgentChatRequest) -> None:
    if request.session_id is not None:
        sanitize_session_id(request.session_id)

    available = tool_names()
    if not request.allowed_tools:
        return

    allowed = set(request.allowed_tools)
    invalid_tools = sorted(allowed - available)
    if invalid_tools:
        invalid_text = ", ".join(invalid_tools)
        raise ValueError(f"allowed_tools 中包含未注册工具: {invalid_text}")


def build_agent_answer(
    settings: AppSettings,
    request: AgentChatRequest,
    state: AgentExecutionState,
    *,
    memory_context: str = "",
) -> str:
    if not state.tool_calls:
        return generate_direct_answer(settings, request, memory_context=memory_context)

    failed_tool = next((item for item in state.tool_calls if item.status == "error"), None)
    if failed_tool is not None:
        return f"执行过程中工具 `{failed_tool.tool_name}` 失败：{failed_tool.error_message or '未知错误'}。"

    executed_names = [item.tool_name for item in state.executed_tools]
    if executed_names == ["search_local_knowledge"]:
        first_result = state.executed_tools[0].result
        if first_result.references:
            answer = generate_rag_answer(
                settings=settings,
                query=request.query,
                references=first_result.references,
                history=request.history,
                agent_memory_context=memory_context,
            )
        else:
            answer = "我已调用 `search_local_knowledge`，但当前知识库没有检索到满足阈值的内容。"
        if state.stop_reason:
            return f"{answer} {state.stop_reason}"
        return answer

    if "calculate" in executed_names:
        calculate_output = latest_tool_output(state, "calculate")
        if "search_local_knowledge" in executed_names:
            answer = f"我先检索了知识库中的相关内容，再完成计算。计算结果：{calculate_output}。"
        elif "current_time" in executed_names:
            time_output = latest_tool_output(state, "current_time")
            answer = f"我先获取了当前时间（{time_output}），再完成计算。计算结果：{calculate_output}。"
        else:
            answer = f"计算结果：{calculate_output}。"
        if state.stop_reason:
            return f"{answer} {state.stop_reason}"
        return answer

    if executed_names == ["current_time"]:
        answer = f"当前时间：{latest_tool_output(state, 'current_time')}。"
        if state.stop_reason:
            return f"{answer} {state.stop_reason}"
        return answer

    latest_output = state.executed_tools[-1].result.output
    answer = f"我已按顺序执行工具：{', '.join(executed_names)}。最后一步输出：{latest_output}。"
    if state.stop_reason:
        return f"{answer} {state.stop_reason}"
    return answer


def append_final_step(state: AgentExecutionState, answer: str) -> None:
    state.steps.append(
        AgentStepRecord(
            step_index=len(state.steps) + 1,
            kind="final",
            status="completed",
            summary="已生成最终回答。",
            output=answer,
        )
    )


def stream_agent_answer(
    settings: AppSettings,
    request: AgentChatRequest,
    state: AgentExecutionState,
    *,
    memory_context: str = "",
) -> Iterator[str]:
    if not state.tool_calls:
        yield from stream_direct_answer(settings, request, memory_context=memory_context)
        return

    failed_tool = next((item for item in state.tool_calls if item.status == "error"), None)
    if failed_tool is not None:
        yield from iter_text_chunks(
            f"执行过程中工具 `{failed_tool.tool_name}` 失败：{failed_tool.error_message or '未知错误'}。"
        )
        return

    executed_names = [item.tool_name for item in state.executed_tools]
    if executed_names == ["search_local_knowledge"]:
        first_result = state.executed_tools[0].result
        if state.stop_reason:
            yield from iter_text_chunks(
                build_agent_answer(
                    settings, request, state, memory_context=memory_context
                )
            )
            return
        if first_result.references:
            yield from stream_rag_answer(
                settings=settings,
                query=request.query,
                references=first_result.references,
                history=request.history,
                agent_memory_context=memory_context,
            )
            return
        yield from iter_text_chunks("我已调用 `search_local_knowledge`，但当前知识库没有检索到满足阈值的内容。")
        return

    if executed_names == ["current_time"] and state.stop_reason:
        yield from iter_text_chunks(
            build_agent_answer(settings, request, state, memory_context=memory_context)
        )
        return

    yield from iter_text_chunks(
        build_agent_answer(settings, request, state, memory_context=memory_context)
    )


def emit_tool_call(emit: Callable[[dict[str, Any]], None] | None, tool_call: ToolCallRecord) -> None:
    if emit is None:
        return
    emit({"type": "tool_call", "tool_call": tool_call.model_dump()})


def emit_step(emit: Callable[[dict[str, Any]], None] | None, step: AgentStepRecord) -> None:
    if emit is None:
        return
    emit({"type": "step", "step": step.model_dump()})


def emit_references(
    emit: Callable[[dict[str, Any]], None] | None,
    references: list[RetrievedReference],
) -> None:
    if emit is None:
        return
    for reference in references:
        emit({"type": "reference", "reference": reference.model_dump()})


def build_plan_signature(plan: PlannedToolCall) -> str:
    return f"{plan.tool_name}:{json.dumps(plan.arguments, ensure_ascii=False, sort_keys=True)}"


def should_search_knowledge(query: str, state: AgentExecutionState) -> bool:
    if has_successful_tool(state, "search_local_knowledge"):
        return False
    if is_time_query(query) and not mentions_knowledge(query):
        return False
    if extract_math_expression(query) and not mentions_knowledge(query):
        return False
    return True


def should_call_current_time(query: str, state: AgentExecutionState) -> bool:
    return is_time_query(query) and not has_successful_tool(state, "current_time")


def should_call_calculate(state: AgentExecutionState) -> bool:
    return not has_successful_tool(state, "calculate")


def has_successful_tool(state: AgentExecutionState, tool_name: str) -> bool:
    return any(
        item.tool_name == tool_name and item.status == "success"
        for item in state.tool_calls
    )


def latest_tool_output(state: AgentExecutionState, tool_name: str) -> str:
    for item in reversed(state.executed_tools):
        if item.tool_name == tool_name:
            return item.result.output
    return ""


def merge_references(
    existing: list[RetrievedReference],
    incoming: list[RetrievedReference],
) -> list[RetrievedReference]:
    merged: list[RetrievedReference] = list(existing)
    seen = {item.chunk_id for item in existing}
    for reference in incoming:
        if reference.chunk_id in seen:
            continue
        seen.add(reference.chunk_id)
        merged.append(reference)
    return merged


def build_tool_step_summary(
    plan: PlannedToolCall,
    tool_result: ToolExecutionResult,
) -> str:
    if plan.tool_name == "search_local_knowledge":
        return f"已检索知识库，命中 {len(tool_result.references)} 条相关片段。"
    if plan.tool_name == "calculate":
        return f"已完成计算：{tool_result.output}。"
    if plan.tool_name == "current_time":
        return f"已获取当前时间：{tool_result.output}。"
    return f"已执行工具 `{plan.tool_name}`。"


def generate_direct_answer(
    settings: AppSettings,
    request: AgentChatRequest,
    *,
    memory_context: str = "",
) -> str:
    prompt = build_agent_direct_prompt()
    variables = {
        "history": convert_history(request.history),
        "memory_section": format_agent_memory_section(memory_context),
        "query": request.query,
    }
    llm = build_chat_model(
        settings,
        model_name=settings.model.AGENT_MODEL,
        temperature=settings.model.TEMPERATURE,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(variables)


def stream_direct_answer(
    settings: AppSettings,
    request: AgentChatRequest,
    *,
    memory_context: str = "",
) -> Iterator[str]:
    prompt = build_agent_direct_prompt()
    variables = {
        "history": convert_history(request.history),
        "memory_section": format_agent_memory_section(memory_context),
        "query": request.query,
    }
    yield from stream_prompt_output(
        settings,
        prompt,
        variables,
        model_name=settings.model.AGENT_MODEL,
        temperature=settings.model.TEMPERATURE,
    )


def format_agent_memory_section(memory_context: str) -> str:
    text = memory_context.strip()
    if not text:
        return ""
    return f"【长期记忆】\n{text}\n\n"


def build_agent_direct_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_DIRECT_SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{memory_section}{query}"),
        ]
    )


def is_time_query(query: str) -> bool:
    normalized = query.lower()
    return any(keyword in normalized for keyword in _TIME_KEYWORDS)


def mentions_knowledge(query: str) -> bool:
    return any(keyword in query for keyword in _KNOWLEDGE_HINTS)


def has_calculation_intent(query: str) -> bool:
    if extract_math_expression(query):
        return True
    return any(keyword in query for keyword in _CALC_HINTS)


def extract_timezone_name(query: str) -> str:
    match = re.search(r"\b[A-Za-z]+/[A-Za-z_]+\b", query)
    if match:
        return match.group(0)
    return ""


def derive_calculate_expression(
    query: str,
    references: list[RetrievedReference],
) -> str:
    direct_expression = extract_math_expression(query)
    if direct_expression:
        return direct_expression
    if not references or not has_calculation_intent(query):
        return ""

    numbers = extract_numbers_from_references(query, references)
    if len(numbers) < 2:
        return ""

    operation = detect_math_operation(query)
    if operation == "average":
        return f"({'+'.join(numbers)})/{len(numbers)}"
    if operation == "multiply":
        return "*".join(numbers[:2])
    if operation == "divide":
        return f"{numbers[0]}/{numbers[1]}"
    if operation == "subtract":
        return f"{numbers[0]}-{numbers[1]}"
    return "+".join(numbers[:2])


def extract_numbers_from_references(
    query: str,
    references: list[RetrievedReference],
) -> list[str]:
    combined = "\n".join(reference.content for reference in references)
    tokens: list[str] = []

    for term in extract_identifier_terms(query):
        pattern = re.compile(rf"{re.escape(term)}[^\d\-]{{0,20}}(-?\d+(?:\.\d+)?)", re.IGNORECASE)
        for match in pattern.findall(combined):
            tokens.append(match)

    if len(tokens) >= 2:
        return deduplicate_strings(tokens)

    fallback = re.findall(r"-?\d+(?:\.\d+)?", combined)
    return deduplicate_strings(tokens + fallback)


def extract_identifier_terms(query: str) -> list[str]:
    return deduplicate_strings(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", query))


def detect_math_operation(query: str) -> str:
    if "平均" in query:
        return "average"
    if any(keyword in query for keyword in ("乘积", "相乘", "乘起来", "乘以")):
        return "multiply"
    if any(keyword in query for keyword in ("除以", "相除", "比值")):
        return "divide"
    if any(keyword in query for keyword in ("差值", "相减", "减去", "差")):
        return "subtract"
    return "sum"


def extract_math_expression(query: str) -> str:
    normalized = (
        query.replace("×", "*")
        .replace("x", "*")
        .replace("X", "*")
        .replace("÷", "/")
        .replace("（", "(")
        .replace("）", ")")
    )
    stripped = normalized.strip()
    if re.fullmatch(r"[\d\.\+\-\*\/%\(\)\s]+", stripped) and re.search(r"\d", stripped):
        return normalize_expression(stripped)

    matches = re.findall(r"[\d\.\+\-\*\/%\(\)\s]{3,}", normalized)
    candidates = [
        normalize_expression(match)
        for match in matches
        if re.search(r"\d", match)
    ]
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return ""
    return max(candidates, key=len)


def normalize_expression(expression: str) -> str:
    normalized = re.sub(r"\s+", "", expression)
    if re.fullmatch(r"[\d\.\+\-\*\/%\(\)]+", normalized) and re.search(r"\d", normalized):
        return normalized
    return ""
