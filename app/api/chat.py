from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.multistep import run_agent, stream_agent_events, validate_agent_request
from app.api.dependencies import SettingsDep
from app.api.errors import error_payload
from app.api.streaming import SSE_MEDIA_TYPE, sse_event
from app.chains.rag import generate_rag_answer, maybe_run_corrective_retrieval, stream_rag_answer
from app.retrievers.local_kb import (
    search_local_knowledge_base,
    search_local_knowledge_base_second_pass,
    search_temp_knowledge_base,
    search_temp_knowledge_base_second_pass,
)
from app.schemas.chat import AgentChatRequest, AgentChatResponse, ChatRequest, ChatResponse
from app.services.reference_overview import build_reference_overview
from app.services.temp_kb_service import TempKnowledgeBaseExpiredError
from app.services.web_search_service import search_corrective_web_references


router = APIRouter(prefix="/chat", tags=["chat"])


def resolve_rag_request(
    settings,
    request: ChatRequest,
) -> tuple[list, str]:
    try:
        if request.source_type == "local_kb":
            if not request.knowledge_base_name:
                raise HTTPException(
                    status_code=422,
                    detail=error_payload(code="missing_knowledge_base_name", message="knowledge_base_name 不能为空。"),
                )
            references = search_local_knowledge_base(
                settings=settings,
                knowledge_base_name=request.knowledge_base_name,
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                history=request.history,
                metadata_filters=request.metadata_filters,
            )
            target_name = request.knowledge_base_name
            references = maybe_run_corrective_retrieval(
                settings=settings,
                query=request.query,
                references=references,
                history=request.history,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                retrieve=lambda corrective_query, corrective_top_k, corrective_threshold: search_local_knowledge_base_second_pass(
                    settings=settings,
                    knowledge_base_name=request.knowledge_base_name,
                    query=corrective_query,
                    top_k=corrective_top_k,
                    score_threshold=corrective_threshold,
                    history=request.history,
                    metadata_filters=request.metadata_filters,
                ),
                search_web=lambda corrective_query: search_corrective_web_references(
                    settings=settings,
                    query=corrective_query,
                ),
                source_type=request.source_type,
                target_name=target_name,
            )
        elif request.source_type == "temp_kb":
            if not request.knowledge_id:
                raise HTTPException(
                    status_code=422,
                    detail=error_payload(code="missing_knowledge_id", message="knowledge_id 不能为空。"),
                )
            references = search_temp_knowledge_base(
                settings=settings,
                knowledge_id=request.knowledge_id,
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                history=request.history,
                metadata_filters=request.metadata_filters,
            )
            target_name = request.knowledge_id
            references = maybe_run_corrective_retrieval(
                settings=settings,
                query=request.query,
                references=references,
                history=request.history,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                retrieve=lambda corrective_query, corrective_top_k, corrective_threshold: search_temp_knowledge_base_second_pass(
                    settings=settings,
                    knowledge_id=request.knowledge_id,
                    query=corrective_query,
                    top_k=corrective_top_k,
                    score_threshold=corrective_threshold,
                    history=request.history,
                    metadata_filters=request.metadata_filters,
                ),
                search_web=lambda corrective_query: search_corrective_web_references(
                    settings=settings,
                    query=corrective_query,
                ),
                source_type=request.source_type,
                target_name=target_name,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=error_payload(
                    code="unsupported_source_type",
                    message=f"暂不支持的 source_type: {request.source_type}",
                ),
            )
    except HTTPException:
        raise
    except TempKnowledgeBaseExpiredError as exc:
        raise HTTPException(
            status_code=410,
            detail=error_payload(code="temp_knowledge_expired", message=str(exc)),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=error_payload(code="knowledge_base_not_found", message=str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=error_payload(code="retrieval_failed", message=f"知识库检索失败: {exc}"),
        ) from exc

    return references, target_name


def rag_stream_response(
    settings,
    request: ChatRequest,
    references,
    target_name: str,
) -> StreamingResponse:
    reference_overview = build_reference_overview(references)

    def event_stream():
        answer_parts: list[str] = []
        try:
            for reference in references:
                yield sse_event("reference", {"reference": reference.model_dump()})

            for delta in stream_rag_answer(
                settings=settings,
                query=request.query,
                references=references,
                history=request.history,
            ):
                answer_parts.append(delta)
                yield sse_event("token", {"delta": delta})

            yield sse_event(
                "done",
                {
                    "answer": "".join(answer_parts),
                    "references": [item.model_dump() for item in references],
                    "reference_overview": reference_overview.model_dump(),
                    "source_type": request.source_type,
                    "knowledge_base_name": target_name,
                    "used_context": bool(references),
                    "stream": True,
                },
            )
        except Exception as exc:
            yield sse_event("error", {"message": f"RAG 生成失败: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type=SSE_MEDIA_TYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def agent_stream_response(
    settings,
    request: AgentChatRequest,
) -> StreamingResponse:
    def event_stream():
        for payload in stream_agent_events(settings=settings, request=request):
            event_type = str(payload.get("type", "message"))
            body = {key: value for key, value in payload.items() if key != "type"}
            yield sse_event(event_type, body)

    return StreamingResponse(
        event_stream(),
        media_type=SSE_MEDIA_TYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/rag", response_model=ChatResponse)
def rag_chat(
    request: ChatRequest,
    settings: SettingsDep,
):
    references, target_name = resolve_rag_request(settings, request)

    if request.stream:
        return rag_stream_response(settings, request, references, target_name)

    try:
        answer = generate_rag_answer(
            settings=settings,
            query=request.query,
            references=references,
            history=request.history,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=error_payload(code="rag_generation_failed", message=f"RAG 生成失败: {exc}"),
        ) from exc

    return ChatResponse(
        answer=answer,
        references=references,
        reference_overview=build_reference_overview(references),
        source_type=request.source_type,
        knowledge_base_name=target_name,
        used_context=bool(references),
        stream=request.stream,
    )


@router.post("/agent", response_model=AgentChatResponse)
def agent_chat(
    request: AgentChatRequest,
    settings: SettingsDep,
):
    if request.stream:
        try:
            validate_agent_request(request)
            return agent_stream_response(settings, request)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=error_payload(code="agent_validation_error", message=str(exc)),
            ) from exc

    try:
        return run_agent(settings=settings, request=request)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=error_payload(code="agent_validation_error", message=str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=error_payload(code="agent_execution_failed", message=f"Agent 执行失败: {exc}"),
        ) from exc
