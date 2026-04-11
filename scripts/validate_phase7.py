from __future__ import annotations

from contextlib import ExitStack
from datetime import timedelta
from hashlib import sha256
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import requests
import uvicorn
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from langchain_core.embeddings import Embeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.main import app
from app.chains.rag import RetrievalCoverageGrade, maybe_run_corrective_retrieval
from app.retrievers.local_kb import build_dense_query_bundle, build_query_bundle, search_local_knowledge_base
from app.storage.bm25_index import load_bm25_index
from app.services.kb_incremental_rebuild import chunk_cache_embedding_path, load_chunk_cache
from app.services.llm_service import build_chat_model, normalize_llm_provider, resolve_openai_compatible_api_key
from app.services.query_rewrite_service import generate_hypothetical_doc, generate_multi_queries
from app.services.rerank_service import RerankTextInput, rerank_texts
from app.services.settings import load_settings
from app.services.temp_kb_service import (
    cleanup_temp_knowledge_bases,
    load_temp_manifest,
    maybe_run_startup_cleanup,
    write_temp_manifest,
)
from app.services.web_search_service import parse_duckduckgo_html_results


def main() -> int:
    summary: dict[str, object] = {
        "api_checks": {},
        "ui_checks": {},
    }

    settings = load_settings(PROJECT_ROOT)
    with offline_validation_patches():
        with TestClient(app) as client:
            summary["api_checks"] = run_api_checks(settings, client)
        summary["ui_checks"] = run_ui_checks()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


class OfflineDeterministicEmbeddings(Embeddings):
    def __init__(self, dimension: int = 64) -> None:
        self.dimension = max(8, dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        tokens = _tokenize_validation_text(text)
        vector = [0.0] * self.dimension
        if not tokens:
            vector[0] = 1.0
            return vector

        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            for offset in range(0, min(16, len(digest)), 4):
                bucket = digest[offset] % self.dimension
                sign = 1.0 if digest[offset + 1] % 2 == 0 else -1.0
                weight = 1.0 + (digest[offset + 2] / 255.0)
                vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def __call__(self, value):
        if isinstance(value, str):
            return self.embed_query(value)
        return self.embed_documents(list(value))


class OfflineCrossEncoder:
    def predict(self, pairs, show_progress_bar: bool = False):
        return [_score_rerank_pair(query, text) for query, text in pairs]


def _tokenize_validation_text(text: str) -> list[str]:
    normalized = (text or "").lower()
    return re.findall(r"[\u4e00-\u9fff]{1,}|[a-z0-9_]+", normalized)


def _score_rerank_pair(query: str, text: str) -> float:
    query_tokens = set(_tokenize_validation_text(query))
    text_tokens = set(_tokenize_validation_text(text))
    if not query_tokens or not text_tokens:
        return 0.0

    overlap = len(query_tokens & text_tokens) / len(query_tokens)
    score = overlap
    has_duration_value = bool(re.search(r"\d+\s*(?:个?月|年|天|日|小时)", text))
    if any(keyword in query for keyword in ("多久", "多少", "几", "期限", "保修")):
        if has_duration_value:
            score += 0.4
        else:
            score -= 0.2
    if any(keyword in text for keyword in ("个月", "工单", "采购单")):
        score += 0.15
    if "什么是" in query and "是" in text:
        score += 0.15
    if any(token in text_tokens for token in ("rag", "检索增强生成")):
        score += 0.1
    if any(phrase in text for phrase in ("不给出任何", "不回答用户问题", "无法回答", "没有给出答案")):
        score -= 0.35
    return score


def _offline_build_embeddings(settings, model_name: str | None = None):
    return OfflineDeterministicEmbeddings()


def _offline_single_query_rewrite(settings, query: str, history=None) -> str:
    return query.strip()


def _offline_multi_query_rewrite(*, settings, query: str, history=None, max_queries: int) -> list[str]:
    return []


def _offline_hypothetical_doc_generation(settings, query: str, history=None) -> str:
    return ""


def _offline_generate_rag_answer(
    settings, query: str, references, history, agent_memory_context: str = "", **_: object
) -> str:
    if not references:
        return "根据当前检索到的内容，无法确定。"

    query_tokens = _tokenize_validation_text(query)
    ranked_contents = sorted(
        (
            (sum(1 for token in query_tokens if token and token in reference.content.lower()), reference.content.strip())
            for reference in references
            if reference.content.strip()
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    best_content = ranked_contents[0][1] if ranked_contents else references[0].content.strip()
    first_sentence = re.split(r"[。！？!?]\s*", best_content, maxsplit=1)[0].strip()
    if first_sentence:
        return first_sentence
    return best_content[:120] or "根据当前检索到的内容，无法确定。"


def _offline_stream_rag_answer(
    settings, query: str, references, history, agent_memory_context: str = "", **_: object
):
    answer = _offline_generate_rag_answer(
        settings, query, references, history, agent_memory_context=agent_memory_context
    )
    chunk_size = 12
    for start in range(0, len(answer), chunk_size):
        yield answer[start : start + chunk_size]


def _offline_select_next_tool_call_with_llm(settings, request, state, memory_context: str = "", **_: object):
    from app.agents.multistep import ToolPlanningDecision

    return ToolPlanningDecision(plan=None, used_llm=False)


def offline_validation_patches() -> ExitStack:
    stack = ExitStack()
    stack.enter_context(patch("app.services.embedding_assembler.build_embeddings", new=_offline_build_embeddings))
    stack.enter_context(patch("app.retrievers.local_kb.build_embeddings", new=_offline_build_embeddings))
    stack.enter_context(
        patch("app.services.query_rewrite_service._invoke_single_query_rewrite", new=_offline_single_query_rewrite)
    )
    stack.enter_context(
        patch("app.services.query_rewrite_service._invoke_multi_query_rewrite", new=_offline_multi_query_rewrite)
    )
    stack.enter_context(
        patch(
            "app.services.query_rewrite_service._invoke_hypothetical_doc_generation",
            new=_offline_hypothetical_doc_generation,
        )
    )
    stack.enter_context(patch("app.services.rerank_service.load_cross_encoder", return_value=OfflineCrossEncoder()))
    stack.enter_context(patch("app.chains.rag.generate_rag_answer", new=_offline_generate_rag_answer))
    stack.enter_context(patch("app.chains.rag.stream_rag_answer", new=_offline_stream_rag_answer))
    stack.enter_context(patch("app.api.chat.generate_rag_answer", new=_offline_generate_rag_answer))
    stack.enter_context(patch("app.api.chat.stream_rag_answer", new=_offline_stream_rag_answer))
    stack.enter_context(patch("app.agents.multistep.generate_rag_answer", new=_offline_generate_rag_answer))
    stack.enter_context(patch("app.agents.multistep.stream_rag_answer", new=_offline_stream_rag_answer))
    stack.enter_context(
        patch("app.agents.multistep.select_next_tool_call_with_llm", new=_offline_select_next_tool_call_with_llm)
    )
    return stack


def run_agent_memory_offline_block(settings) -> dict[str, object]:
    from unittest.mock import patch

    from app.services.memory_service import (
        persist_agent_turns,
        retrieve_agent_memory,
        sanitize_session_id,
    )

    rejected = False
    try:
        sanitize_session_id("bad id")
    except ValueError:
        rejected = True
    assert rejected, "sanitize_session_id 应拒绝含空格的 session_id"

    result: dict[str, object] = {"invalid_session_rejected": True}
    tmp_root = (PROJECT_ROOT / "data" / "_validate_tmp" / f"agent_memory_{int(time.time() * 1000)}").resolve()
    if tmp_root.exists():
        shutil.rmtree(tmp_root, onerror=_handle_rmtree_error)
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        mem_root = (tmp_root / "am").resolve()
        trial = settings.model_copy(
            update={
                "basic": settings.basic.model_copy(
                    update={
                        "ENABLE_AGENT_MEMORY": True,
                        "AGENT_MEMORY_ROOT": str(mem_root),
                        "AGENT_MEMORY_EPISODE_MAX_TURNS": 100,
                    }
                )
            }
        )
        sid = "phase7_mem_demo"
        persist_agent_turns(
            trial,
            session_id=sid,
            user_text="偏好深色主题",
            assistant_text="已记录。",
            tools_used=[],
        )
        sem_path = trial.agent_memory_session_dir(sid) / "semantics.jsonl"
        record = {
            "semantic_id": "sem-000001",
            "fact": "用户偏好深色主题 UI",
            "episode_id": "ep-000001",
            "category": "preference",
            "confidence": 0.9,
            "source_ids": [],
            "parent_ids": [],
            "children_ids": [],
        }
        with sem_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        with patch("app.services.memory_service.build_embeddings", new=_offline_build_embeddings):
            bundle = retrieve_agent_memory(
                trial,
                session_id=sid,
                query="深色主题 偏好",
            )
        result["semantic_hits"] = bundle.overview.semantic_hits
        result["used_memory"] = bundle.overview.used_memory
        assert bundle.overview.semantic_hits >= 1, "应命中注入的语义记忆"
        assert bundle.overview.used_memory, "记忆上下文应非空"
    finally:
        if tmp_root.exists():
            try:
                shutil.rmtree(tmp_root, onerror=_handle_rmtree_error)
            except OSError:
                pass
    return result


def submit_rebuild_and_wait(
    client: TestClient,
    payload: dict[str, object],
    *,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 0.1,
) -> tuple[int, dict[str, object]]:
    response = client.post("/knowledge_base/rebuild", json=payload)
    assert response.status_code == 202, "knowledge_base/rebuild task submit failed"
    accepted_payload = response.json()
    task_id = str(accepted_payload.get("task_id", "")).strip()
    assert task_id, "knowledge_base/rebuild did not return task_id"

    deadline = time.time() + timeout_seconds
    last_payload: dict[str, object] = accepted_payload
    while time.time() < deadline:
        status_response = client.get(f"/knowledge_base/rebuild/{task_id}")
        assert status_response.status_code == 200, "knowledge_base/rebuild status query failed"
        task_payload = status_response.json()
        last_payload = task_payload
        status = str(task_payload.get("status", ""))
        if status == "succeeded":
            result = task_payload.get("result")
            assert isinstance(result, dict), "rebuild task succeeded without result"
            return status_response.status_code, result
        if status == "failed":
            raise AssertionError(
                f"knowledge_base/rebuild failed: {task_payload.get('error_message', 'unknown error')}"
            )
        time.sleep(poll_interval_seconds)

    raise AssertionError(f"knowledge_base/rebuild timed out: {last_payload}")


def run_api_checks(settings, client: TestClient) -> dict[str, object]:
    checks: dict[str, object] = {}
    local_upload_kb_name = "phase7_local_upload_demo"
    multistep_kb_name = "phase7_multistep_demo"
    optimized_rag_kb_name = "phase7_rag_optimized_demo"
    incremental_kb_name = f"phaseg_incremental_demo_validation_{int(time.time())}"

    list_response = client.get("/knowledge_base/list")
    list_json = list_response.json()
    checks["knowledge_base_list"] = {
        "status_code": list_response.status_code,
        "knowledge_base_names": [item["knowledge_base_name"] for item in list_json],
    }
    assert list_response.status_code == 200, "knowledge_base/list failed"

    default_chat_model = build_chat_model(settings)
    api_provider_settings = settings.model_copy(
        update={
            "model": settings.model.model_copy(
                update={
                    "LLM_PROVIDER": "api",
                    "OPENAI_COMPATIBLE_BASE_URL": "https://example.com/v1",
                    "OPENAI_COMPATIBLE_API_KEY": "demo-key",
                }
            )
        }
    )
    checks["llm_provider_switching"] = {
        "default_provider": normalize_llm_provider(settings.model.LLM_PROVIDER),
        "default_model_class": default_chat_model.__class__.__name__,
        "api_provider_alias": normalize_llm_provider(api_provider_settings.model.LLM_PROVIDER),
        "api_key_resolved": resolve_openai_compatible_api_key(api_provider_settings),
    }
    expected_default_backend = {
        "ollama": "ChatOllama",
        "openai_compatible": "ChatOpenAI",
    }.get(checks["llm_provider_switching"]["default_provider"])
    assert expected_default_backend is not None, "default llm provider is unsupported"
    assert checks["llm_provider_switching"]["default_model_class"] == expected_default_backend, "default llm backend changed unexpectedly"
    assert checks["llm_provider_switching"]["api_provider_alias"] == "openai_compatible", "api provider alias mismatch"
    assert checks["llm_provider_switching"]["api_key_resolved"] == "demo-key", "api provider key resolve failed"

    multi_query_input = "电机额定电流和过载保护参数在说明书哪里看"
    with patch(
        "app.services.query_rewrite_service._invoke_multi_query_rewrite",
        return_value=[
            "电机 额定电流 过载保护 参数 说明书",
            "电机说明书 额定电流 过载保护 故障 排查",
        ],
    ):
        multi_queries = generate_multi_queries(settings, multi_query_input)
    multi_query_bundle = build_query_bundle(multi_queries)
    checks["multi_query_rewrite"] = {
        "query_count": len(multi_queries),
        "query_bundle_count": len(multi_query_bundle),
        "queries": multi_queries,
    }
    assert multi_queries[0] == multi_query_input, "multi query should keep original query first"
    assert len(multi_queries) == 3, "multi query should expand to original + 2 rewrites"
    assert len(set(multi_queries)) == len(multi_queries), "multi query should deduplicate rewritten queries"
    assert multi_query_bundle[: len(multi_queries)] == multi_queries, "query bundle should preserve multi query order"

    hyde_settings = settings.model_copy(
        update={
            "kb": settings.kb.model_copy(
                update={
                    "ENABLE_HYDE": True,
                }
            )
        }
    )
    with patch(
        "app.services.query_rewrite_service._invoke_hypothetical_doc_generation",
        return_value="The motor manual usually lists rated current, overload protection thresholds, and related troubleshooting notes in the electrical parameter section.",
    ):
        hypothetical_doc = generate_hypothetical_doc(hyde_settings, multi_query_input)
    dense_query_bundle = build_dense_query_bundle(multi_query_bundle, hypothetical_doc)
    checks["hyde_generation"] = {
        "hypothetical_doc": hypothetical_doc,
        "dense_query_bundle_count": len(dense_query_bundle),
        "dense_query_tail": dense_query_bundle[-1] if dense_query_bundle else "",
    }
    assert hypothetical_doc, "hyde should generate hypothetical document text when enabled"
    assert dense_query_bundle[-1] == hypothetical_doc, "hyde text should only append to dense retrieval bundle"
    assert dense_query_bundle[: len(multi_query_bundle)] == multi_query_bundle, "hyde should not reorder lexical query bundle"

    corrective_settings = settings.model_copy(
        update={
            "kb": settings.kb.model_copy(
                update={
                    "ENABLE_CORRECTIVE_RAG": True,
                    "CORRECTIVE_RAG_SECOND_PASS_TOP_K": 5,
                    "CORRECTIVE_RAG_SECOND_PASS_SCORE_THRESHOLD": 0.2,
                }
            )
        }
    )
    initial_corrective_refs = [
        _build_retrieved_reference(
            chunk_id="partial-1",
            source="partial.txt",
            content="ZX900 标准保修期为 24 个月。",
            relevance_score=0.58,
        )
    ]
    second_pass_calls: list[dict[str, object]] = []

    def _mock_second_pass(corrective_query: str, corrective_top_k: int, corrective_threshold: float):
        second_pass_calls.append(
            {
                "query": corrective_query,
                "top_k": corrective_top_k,
                "score_threshold": corrective_threshold,
            }
        )
        return [
            _build_retrieved_reference(
                chunk_id="better-1",
                source="after_sales.txt",
                content="ZX900 售后申请需要提供采购单、安装照片，并先联系技术支持创建工单。",
                relevance_score=0.91,
            )
        ]

    with patch(
        "app.chains.rag.grade_documents",
        return_value=RetrievalCoverageGrade(
            grade="partial",
            reason="已有保修信息，但缺少售后材料要求。",
            missing_aspects="缺少售后准备材料与提单流程。",
        ),
    ), patch(
        "app.chains.rag.generate_corrective_query",
        return_value="ZX900 保修 售后 材料 工单",
    ):
        corrective_refs = maybe_run_corrective_retrieval(
            settings=corrective_settings,
            query="ZX900 的保修多久，售后要准备什么？",
            references=initial_corrective_refs,
            history=[],
            top_k=2,
            score_threshold=0.5,
            retrieve=_mock_second_pass,
            source_type="local_kb",
            target_name="phase7_corrective_demo",
        )
    checks["corrective_rag_second_pass"] = {
        "call_count": len(second_pass_calls),
        "follow_up_query": second_pass_calls[0]["query"] if second_pass_calls else "",
        "follow_up_top_k": second_pass_calls[0]["top_k"] if second_pass_calls else 0,
        "follow_up_score_threshold": second_pass_calls[0]["score_threshold"] if second_pass_calls else 0.0,
        "final_sources": [item.source for item in corrective_refs],
    }
    assert second_pass_calls, "corrective rag did not trigger second retrieval pass"
    assert second_pass_calls[0]["query"] == "ZX900 保修 售后 材料 工单", "corrective rag follow-up query mismatch"
    assert second_pass_calls[0]["top_k"] == 5, "corrective rag second pass top_k mismatch"
    assert abs(float(second_pass_calls[0]["score_threshold"]) - 0.2) < 1e-9, "corrective rag second pass threshold mismatch"
    assert [item.source for item in corrective_refs][:2] == ["after_sales.txt", "partial.txt"], "corrective rag should merge second-pass evidence first"

    sample_web_html = """
    <div class="results">
      <div class="result">
        <a class="result__a" href="https://support.example.com/zx900-warranty">ZX900 售后工单流程</a>
        <a class="result__snippet">ZX900 售后申请需要提供采购单和安装照片，并先联系技术支持创建工单。</a>
      </div>
      <div class="result">
        <a class="result__a" href="https://docs.example.com/zx900-guide">ZX900 维保说明</a>
        <a class="result__snippet">标准保修期为 24 个月，自验收日期开始计算。</a>
      </div>
    </div>
    """
    parsed_web_results = parse_duckduckgo_html_results(sample_web_html, limit=2)
    checks["corrective_web_search_parser"] = {
        "count": len(parsed_web_results),
        "first_url": parsed_web_results[0].url if parsed_web_results else "",
        "first_domain": parsed_web_results[0].source_domain if parsed_web_results else "",
    }
    assert len(parsed_web_results) == 2, "corrective web search parser should return two snippets"
    assert parsed_web_results[0].source_domain == "support.example.com", "corrective web search parser domain mismatch"

    web_corrective_settings = settings.model_copy(
        update={
            "kb": settings.kb.model_copy(
                update={
                    "ENABLE_CORRECTIVE_RAG": True,
                    "ENABLE_CORRECTIVE_WEB_SEARCH": True,
                    "CORRECTIVE_RAG_SECOND_PASS_TOP_K": 5,
                    "CORRECTIVE_RAG_SECOND_PASS_SCORE_THRESHOLD": 0.2,
                    "CORRECTIVE_WEB_SEARCH_TOP_K": 2,
                }
            )
        }
    )
    web_search_calls: list[str] = []

    def _mock_web_search(corrective_query: str):
        web_search_calls.append(corrective_query)
        return [
            _build_retrieved_reference(
                chunk_id="web-1",
                source="support.example.com",
                content="ZX900 售后申请需要提供采购单和安装照片，并先联系技术支持创建工单。",
                relevance_score=0.67,
            )
        ]

    with patch(
        "app.chains.rag.grade_documents",
        return_value=RetrievalCoverageGrade(
            grade="partial",
            reason="已有保修信息，但缺少外部售后流程补充。",
            missing_aspects="缺少售后申请材料和工单流程。",
        ),
    ), patch(
        "app.chains.rag.generate_corrective_query",
        return_value="ZX900 保修 售后 材料 工单",
    ):
        web_supplement_refs = maybe_run_corrective_retrieval(
            settings=web_corrective_settings,
            query="ZX900 的保修多久，售后要准备什么？",
            references=initial_corrective_refs,
            history=[],
            top_k=2,
            score_threshold=0.5,
            retrieve=_mock_second_pass,
            search_web=_mock_web_search,
            source_type="local_kb",
            target_name="phase7_corrective_demo",
        )
    checks["corrective_rag_web_supplement"] = {
        "call_count": len(web_search_calls),
        "query": web_search_calls[0] if web_search_calls else "",
        "sources": [item.source for item in web_supplement_refs],
    }
    assert web_search_calls, "corrective rag did not trigger web supplement search"
    assert web_search_calls[0] == "ZX900 保修 售后 材料 工单", "corrective rag web query mismatch"
    assert "support.example.com" in checks["corrective_rag_web_supplement"]["sources"], "corrective rag web result was not merged"

    write_local_knowledge_base_files(
        settings,
        "phase2_demo",
        {
            "phase7_validation_rag.txt": (
                "RAG 是检索增强生成，先从知识库检索相关资料，再把命中的上下文交给大模型生成答案。"
                "这样可以减少幻觉，并让回答引用企业文档中的事实。"
            )
        },
    )
    rebuild_status_code, rebuild_json = submit_rebuild_and_wait(
        client,
        {"knowledge_base_name": "phase2_demo"},
    )
    checks["rebuild_phase2_demo"] = {
        "status_code": rebuild_status_code,
        "chunks": rebuild_json.get("chunks"),
        "files_processed": rebuild_json.get("files_processed"),
    }
    assert rebuild_status_code == 200, "knowledge_base/rebuild failed"

    write_local_knowledge_base_files(
        settings,
        incremental_kb_name,
        {
            "intro.txt": "RAG 会先检索知识，再将相关片段交给大模型生成答案。",
            "faq.txt": "如果文件没有变化，重建时应该优先复用已有切片和向量缓存。",
        },
    )
    incremental_full_status_code, incremental_full_json = submit_rebuild_and_wait(
        client,
        {"knowledge_base_name": incremental_kb_name},
    )
    checks["incremental_rebuild_full"] = {
        "status_code": incremental_full_status_code,
        "index_mode": incremental_full_json.get("index_mode"),
        "files_rebuilt": incremental_full_json.get("files_rebuilt"),
        "chunks_embedded": incremental_full_json.get("chunks_embedded"),
        "build_manifest_path": incremental_full_json.get("build_manifest_path"),
    }
    assert incremental_full_status_code == 200, "incremental full rebuild failed"
    assert incremental_full_json.get("index_mode") == "full", "initial incremental rebuild should be full"
    assert incremental_full_json.get("files_rebuilt") == 2, "initial incremental rebuild file count mismatch"
    assert incremental_full_json.get("chunks_embedded", 0) >= 2, "initial incremental rebuild did not embed chunks"

    chunk_cache_dir = settings.vector_store_chunk_cache_dir(incremental_kb_name)
    chunk_cache_files = sorted(chunk_cache_dir.glob("*.json"))
    assert chunk_cache_files, "incremental rebuild did not write chunk cache metadata"
    first_chunk_cache = chunk_cache_files[0]
    first_embedding_cache = chunk_cache_embedding_path(first_chunk_cache)
    loaded_chunk_cache = load_chunk_cache(first_chunk_cache)
    raw_chunk_cache_payload = json.loads(first_chunk_cache.read_text(encoding="utf-8"))
    first_raw_entry = raw_chunk_cache_payload.get("chunk_entries", [{}])[0]
    checks["incremental_chunk_cache_numpy"] = {
        "metadata_cache_file": str(first_chunk_cache),
        "embedding_cache_file": str(first_embedding_cache),
        "embedding_cache_exists": first_embedding_cache.exists(),
        "chunk_entry_count": len(loaded_chunk_cache.chunk_entries),
        "embedding_loaded": bool(
            loaded_chunk_cache.chunk_entries and loaded_chunk_cache.chunk_entries[0].embedding
        ),
        "embedding_in_metadata": bool(
            isinstance(first_raw_entry, dict) and "embedding" in first_raw_entry
        ),
    }
    assert first_embedding_cache.exists(), "chunk cache embedding npy file missing"
    assert loaded_chunk_cache.chunk_entries, "chunk cache metadata contains no chunk entries"
    assert loaded_chunk_cache.chunk_entries[0].embedding, "chunk cache did not load embedding from cache"
    assert not checks["incremental_chunk_cache_numpy"]["embedding_in_metadata"], "chunk cache metadata still stores embedding inline"

    bm25_index_path = settings.vector_store_bm25_index_path(incremental_kb_name)
    loaded_bm25_index = load_bm25_index(bm25_index_path)
    checks["incremental_bm25_persisted_index"] = {
        "bm25_index_file": str(bm25_index_path),
        "exists": bm25_index_path.exists(),
        "backend": loaded_bm25_index.backend if loaded_bm25_index is not None else "missing",
        "chunk_count": len(loaded_bm25_index.chunk_ids) if loaded_bm25_index is not None else 0,
    }
    assert bm25_index_path.exists(), "bm25 persisted index file missing"
    assert loaded_bm25_index is not None, "bm25 persisted index failed to load"
    assert loaded_bm25_index.chunk_ids, "bm25 persisted index contains no chunks"

    incremental_reuse_status_code, incremental_reuse_json = submit_rebuild_and_wait(
        client,
        {"knowledge_base_name": incremental_kb_name},
    )
    checks["incremental_rebuild_reuse"] = {
        "status_code": incremental_reuse_status_code,
        "index_mode": incremental_reuse_json.get("index_mode"),
        "files_reused": incremental_reuse_json.get("files_reused"),
        "files_rebuilt": incremental_reuse_json.get("files_rebuilt"),
        "chunks_reused": incremental_reuse_json.get("chunks_reused"),
        "chunks_embedded": incremental_reuse_json.get("chunks_embedded"),
    }
    assert incremental_reuse_status_code == 200, "incremental reuse rebuild failed"
    assert incremental_reuse_json.get("index_mode") == "reuse", "incremental rebuild did not enter reuse mode"
    assert incremental_reuse_json.get("files_rebuilt") == 0, "reuse rebuild unexpectedly rebuilt files"
    assert incremental_reuse_json.get("chunks_embedded") == 0, "reuse rebuild unexpectedly embedded chunks"

    write_local_knowledge_base_files(
        settings,
        incremental_kb_name,
        {
            "append.txt": "新增文件时，索引应优先采用追加方式，而不是每次推翻整个向量库。",
        },
    )
    incremental_append_status_code, incremental_append_json = submit_rebuild_and_wait(
        client,
        {"knowledge_base_name": incremental_kb_name},
    )
    checks["incremental_rebuild_append"] = {
        "status_code": incremental_append_status_code,
        "index_mode": incremental_append_json.get("index_mode"),
        "files_reused": incremental_append_json.get("files_reused"),
        "files_rebuilt": incremental_append_json.get("files_rebuilt"),
        "chunks_reused": incremental_append_json.get("chunks_reused"),
        "chunks_embedded": incremental_append_json.get("chunks_embedded"),
    }
    assert incremental_append_status_code == 200, "incremental append rebuild failed"
    assert incremental_append_json.get("index_mode") == "append", "incremental rebuild did not enter append mode"
    assert incremental_append_json.get("files_reused", 0) >= 2, "append rebuild did not reuse existing files"
    assert incremental_append_json.get("files_rebuilt") == 1, "append rebuild should rebuild exactly one file"

    incremental_append_rag_response = client.post(
        "/chat/rag",
        json={
            "query": "新增文件时应该优先采用什么方式？",
            "source_type": "local_kb",
            "knowledge_base_name": incremental_kb_name,
        },
    )
    incremental_append_rag_json = incremental_append_rag_response.json()
    checks["incremental_rebuild_append_rag"] = {
        "status_code": incremental_append_rag_response.status_code,
        "used_context": incremental_append_rag_json.get("used_context"),
        "first_source": _first_source(incremental_append_rag_json),
        "sources": [
            item.get("source", "")
            for item in incremental_append_rag_json.get("references", [])
            if isinstance(item, dict)
        ],
    }
    assert incremental_append_rag_response.status_code == 200, "incremental append rag failed"
    assert "append.txt" in checks["incremental_rebuild_append_rag"]["sources"], "append rebuild result was not searchable"

    write_local_knowledge_base_files(
        settings,
        incremental_kb_name,
        {
            "faq.txt": "如果文件内容发生变化，重建流程应安全回退，但其他未变化文件仍应复用缓存。",
        },
    )
    incremental_modified_status_code, incremental_modified_json = submit_rebuild_and_wait(
        client,
        {"knowledge_base_name": incremental_kb_name},
    )
    checks["incremental_rebuild_modified_full"] = {
        "status_code": incremental_modified_status_code,
        "index_mode": incremental_modified_json.get("index_mode"),
        "files_reused": incremental_modified_json.get("files_reused"),
        "files_rebuilt": incremental_modified_json.get("files_rebuilt"),
        "chunks_reused": incremental_modified_json.get("chunks_reused"),
        "chunks_embedded": incremental_modified_json.get("chunks_embedded"),
    }
    assert incremental_modified_status_code == 200, "incremental modified rebuild failed"
    assert incremental_modified_json.get("index_mode") == "full", "modified file should trigger full rebuild"
    assert incremental_modified_json.get("files_reused", 0) >= 2, "modified rebuild should still reuse unchanged files"
    assert incremental_modified_json.get("files_rebuilt") == 1, "modified rebuild should rebuild one changed file"

    rag_response = client.post(
        "/chat/rag",
        json={
            "query": "\u4ec0\u4e48\u662f RAG\uff1f",
            "source_type": "local_kb",
            "knowledge_base_name": "phase2_demo",
        },
    )
    rag_json = rag_response.json()
    checks["local_kb_rag"] = {
        "status_code": rag_response.status_code,
        "used_context": rag_json.get("used_context"),
        "reference_count": len(rag_json.get("references", [])),
        "first_source": _first_source(rag_json),
    }
    assert rag_response.status_code == 200, "local_kb rag failed"
    assert rag_json.get("used_context"), "local_kb rag did not use context"

    rag_stream_status, rag_stream_events = parse_stream_events(
        client,
        "/chat/rag",
        {
            "query": "\u4ec0\u4e48\u662f RAG\uff1f",
            "source_type": "local_kb",
            "knowledge_base_name": "phase2_demo",
            "stream": True,
        },
    )
    checks["local_kb_rag_stream"] = {
        "status_code": rag_stream_status,
        "event_types": [item["type"] for item in rag_stream_events[:6]],
        "last_event_type": rag_stream_events[-1]["type"] if rag_stream_events else "",
    }
    assert rag_stream_status == 200, "local_kb rag stream failed"
    assert any(item["type"] == "token" for item in rag_stream_events), "rag stream has no token event"
    assert rag_stream_events and rag_stream_events[-1]["type"] == "done", "rag stream missing done event"

    rerank_enabled_settings = settings.model_copy(
        update={
            "kb": settings.kb.model_copy(update={"ENABLE_MODEL_RERANK": True}),
        }
    )
    with patch("app.services.rerank_service.load_cross_encoder", return_value=OfflineCrossEncoder()):
        rerank_probe = rerank_texts(
            settings=rerank_enabled_settings,
            query="什么是 RAG？",
            items=[
                RerankTextInput(
                    candidate_id="probe-intro",
                    text="RAG 是检索增强生成，通过先检索外部知识再调用大模型生成答案。",
                )
            ],
            top_n=1,
        )
    checks["model_rerank_probe"] = {
        "applied": rerank_probe.applied,
        "strategy": rerank_probe.strategy,
        "message": rerank_probe.message,
    }
    assert rerank_probe.applied, "model rerank probe did not apply"
    assert rerank_probe.strategy == "model", "unexpected model rerank probe strategy"

    with patch("app.services.rerank_service.load_cross_encoder", return_value=OfflineCrossEncoder()):
        rerank_pair_probe = rerank_texts(
            settings=rerank_enabled_settings,
            query="AlphaX2000 的保修期多久？",
            items=[
                RerankTextInput(
                    candidate_id="distractor",
                    text=(
                        "AlphaX2000 保修期多久 AlphaX2000 保修期多久 AlphaX2000 保修期多久。"
                        "这里反复提到保修期多久和 AlphaX2000，但不给出任何时长数字，也不回答用户问题。"
                    ),
                ),
                RerankTextInput(
                    candidate_id="warranty",
                    text="AlphaX2000 标准保修期为 36 个月，自验收日期开始计算。",
                ),
            ],
            top_n=2,
        )
    checks["model_rerank_pair_probe"] = {
        "scores": rerank_pair_probe.scores,
    }
    assert set(rerank_pair_probe.scores) == {"distractor", "warranty"}, "model rerank pair probe missing candidates"
    assert rerank_pair_probe.scores.get("warranty", 0.0) >= rerank_pair_probe.scores.get("distractor", 0.0), "model rerank pair probe should not rank the correct answer lower than distractor"

    with patch("app.services.rerank_service.load_cross_encoder", return_value=OfflineCrossEncoder()):
        rerank_fallback_references = search_local_knowledge_base(
            settings=rerank_enabled_settings,
            knowledge_base_name="phase2_demo",
            query="什么是 RAG？",
            top_k=4,
            score_threshold=0.5,
        )
    checks["model_rerank_fallback_search"] = {
        "reference_count": len(rerank_fallback_references),
        "first_source": rerank_fallback_references[0].source if rerank_fallback_references else "",
    }
    assert rerank_fallback_references, "model rerank fallback search returned no references"

    reranker_demo_kb_name = "phasef_reranker_demo"
    reranker_demo_upload = client.post(
        "/knowledge_base/upload",
        data={
            "scope": "local",
            "knowledge_base_name": reranker_demo_kb_name,
            "overwrite_existing": "true",
            "auto_rebuild": "true",
        },
        files=[
            (
                "files",
                (
                    "distractor.txt",
                    (
                        "AlphaX2000 保修期多久 AlphaX2000 保修期多久 AlphaX2000 保修期多久。"
                        "这里反复提到保修期多久和 AlphaX2000，但不给出任何时长数字，也不回答用户问题。"
                    ).encode("utf-8"),
                    "text/plain",
                ),
            ),
            (
                "files",
                (
                    "warranty.txt",
                    "AlphaX2000 标准保修期为 36 个月，自验收日期开始计算。".encode("utf-8"),
                    "text/plain",
                ),
            ),
        ],
    )
    assert reranker_demo_upload.status_code == 200, "reranker demo upload failed"
    heuristic_only_settings = settings.model_copy(
        update={
            "kb": settings.kb.model_copy(update={"ENABLE_MODEL_RERANK": False}),
        }
    )
    reranker_query = "AlphaX2000 的保修期多久？"
    heuristic_reranker_refs = search_local_knowledge_base(
        settings=heuristic_only_settings,
        knowledge_base_name=reranker_demo_kb_name,
        query=reranker_query,
        top_k=2,
        score_threshold=0.0,
    )
    with patch("app.services.rerank_service.load_cross_encoder", return_value=OfflineCrossEncoder()):
        model_reranker_refs = search_local_knowledge_base(
            settings=rerank_enabled_settings,
            knowledge_base_name=reranker_demo_kb_name,
            query=reranker_query,
            top_k=2,
            score_threshold=0.0,
        )
    checks["model_rerank_beats_heuristic"] = {
        "heuristic_sources": [item.source for item in heuristic_reranker_refs],
        "model_sources": [item.source for item in model_reranker_refs],
    }
    assert heuristic_reranker_refs, "heuristic reranker demo returned no references"
    assert model_reranker_refs, "model reranker demo returned no references"
    assert "distractor.txt" in checks["model_rerank_beats_heuristic"]["heuristic_sources"], "heuristic demo did not return distractor candidate"
    assert "warranty.txt" in checks["model_rerank_beats_heuristic"]["model_sources"], "model rerank demo did not return the correct answer candidate"

    local_upload_response = client.post(
        "/knowledge_base/upload",
        data={
            "scope": "local",
            "knowledge_base_name": local_upload_kb_name,
            "overwrite_existing": "true",
            "auto_rebuild": "true",
            "chunk_size": str(settings.kb.CHUNK_SIZE),
            "chunk_overlap": str(settings.kb.CHUNK_OVERLAP),
        },
        files=[
            (
                "files",
                (
                    "phase7_local_upload.txt",
                    (
                        "\u957f\u671f\u77e5\u8bc6\u5e93\u4e0a\u4f20\u73b0\u5728\u652f\u6301\u540c\u540d\u6587\u4ef6\u8986\u76d6\u63a7\u5236\uff0c"
                        "\u5e76\u4e14\u53ef\u4ee5\u5728\u4e0a\u4f20\u540e\u76f4\u63a5\u89e6\u53d1\u81ea\u52a8\u91cd\u5efa\u3002"
                    ).encode("utf-8"),
                    "text/plain",
                ),
            )
        ],
    )
    local_upload_json = local_upload_response.json()
    checks["local_kb_upload"] = {
        "status_code": local_upload_response.status_code,
        "knowledge_base_name": local_upload_json.get("knowledge_base_name"),
        "files_processed": local_upload_json.get("files_processed"),
        "auto_rebuild": local_upload_json.get("auto_rebuild"),
        "requires_rebuild": local_upload_json.get("requires_rebuild"),
    }
    assert local_upload_response.status_code == 200, "local knowledge base upload failed"
    assert local_upload_json.get("rebuild_result"), "local knowledge base upload did not rebuild automatically"

    optimized_rag_upload_response = client.post(
        "/knowledge_base/upload",
        data={
            "scope": "local",
            "knowledge_base_name": optimized_rag_kb_name,
            "overwrite_existing": "true",
            "auto_rebuild": "true",
            "chunk_size": "48",
            "chunk_overlap": "8",
        },
        files=[
            (
                "files",
                (
                    "phase7_rag_optimized.txt",
                    (
                        "ZX-900 工业传感器适用于高温环境。安装前需要保持电源关闭，并按照维护手册完成校准。"
                        "产品的标准保修期为 24 个月，保修从验收日期开始计算。若设备序列号缺失，则无法享受保修服务。"
                        "售后申请时需要提供采购单和安装照片，返修前需要先联系技术支持创建工单。"
                    ).encode("utf-8"),
                    "text/plain",
                ),
            )
        ],
    )
    optimized_rag_upload_json = optimized_rag_upload_response.json()
    checks["optimized_rag_upload"] = {
        "status_code": optimized_rag_upload_response.status_code,
        "knowledge_base_name": optimized_rag_upload_json.get("knowledge_base_name"),
        "files_processed": optimized_rag_upload_json.get("files_processed"),
        "chunks": optimized_rag_upload_json.get("chunks"),
    }
    assert optimized_rag_upload_response.status_code == 200, "optimized rag upload failed"
    assert optimized_rag_upload_json.get("rebuild_result"), "optimized rag upload did not rebuild automatically"

    optimized_rag_response = client.post(
        "/chat/rag",
        json={
            "query": "帮我看看 ZX900 这个型号的保修是多久，售后要准备什么？",
            "source_type": "local_kb",
            "knowledge_base_name": optimized_rag_kb_name,
        },
    )
    optimized_rag_json = optimized_rag_response.json()
    optimized_first_reference = _first_reference(optimized_rag_json)
    optimized_first_content = (
        str(optimized_first_reference.get("content", ""))
        if isinstance(optimized_first_reference, dict)
        else ""
    )
    checks["optimized_rag_hybrid_small_to_big"] = {
        "status_code": optimized_rag_response.status_code,
        "used_context": optimized_rag_json.get("used_context"),
        "reference_count": len(optimized_rag_json.get("references", [])),
        "first_source": _first_source(optimized_rag_json),
        "first_content_length": len(optimized_first_content),
        "contains_warranty": "24 个月" in optimized_first_content,
        "contains_after_sales": "工单" in optimized_first_content,
    }
    assert optimized_rag_response.status_code == 200, "optimized rag request failed"
    assert optimized_rag_json.get("used_context"), "optimized rag did not use context"
    assert "24 个月" in optimized_first_content, "optimized rag reference missing warranty text"
    assert "工单" in optimized_first_content, "optimized rag reference did not expand to adjacent context"
    assert len(optimized_first_content) > 80, "optimized rag reference did not return expanded context"

    local_upload_rag_response = client.post(
        "/chat/rag",
        json={
            "query": "\u957f\u671f\u77e5\u8bc6\u5e93\u4e0a\u4f20\u652f\u6301\u4ec0\u4e48\uff1f",
            "source_type": "local_kb",
            "knowledge_base_name": local_upload_kb_name,
            "score_threshold": 0.0,
        },
    )
    local_upload_rag_json = local_upload_rag_response.json()
    checks["local_kb_upload_rag"] = {
        "status_code": local_upload_rag_response.status_code,
        "used_context": local_upload_rag_json.get("used_context"),
        "reference_count": len(local_upload_rag_json.get("references", [])),
        "first_source": _first_source(local_upload_rag_json),
    }
    assert local_upload_rag_response.status_code == 200, "local uploaded knowledge base rag failed"
    assert local_upload_rag_json.get("used_context"), "local uploaded knowledge base rag did not use context"

    local_upload_duplicate_response = client.post(
        "/knowledge_base/upload",
        data={
            "scope": "local",
            "knowledge_base_name": local_upload_kb_name,
            "overwrite_existing": "false",
            "auto_rebuild": "false",
        },
        files=[
            (
                "files",
                (
                    "phase7_local_upload.txt",
                    "\u8fd9\u662f\u7528\u4e8e\u91cd\u540d\u8df3\u8fc7\u6821\u9a8c\u7684\u5185\u5bb9\u3002".encode("utf-8"),
                    "text/plain",
                ),
            )
        ],
    )
    local_upload_duplicate_json = local_upload_duplicate_response.json()
    checks["local_kb_upload_duplicate_skip"] = {
        "status_code": local_upload_duplicate_response.status_code,
        "files_processed": local_upload_duplicate_json.get("files_processed"),
        "skipped_files": local_upload_duplicate_json.get("skipped_files"),
        "requires_rebuild": local_upload_duplicate_json.get("requires_rebuild"),
    }
    assert local_upload_duplicate_response.status_code == 200, "local duplicate upload failed"
    assert local_upload_duplicate_json.get("files_processed") == 0, "duplicate local upload unexpectedly wrote files"

    multistep_upload_response = client.post(
        "/knowledge_base/upload",
        data={
            "scope": "local",
            "knowledge_base_name": multistep_kb_name,
            "overwrite_existing": "true",
            "auto_rebuild": "true",
        },
        files=[
            (
                "files",
                (
                    "phase7_multistep.txt",
                    (
                        "\u8fd9\u4e2a\u591a\u6b65\u77e5\u8bc6\u5e93\u793a\u4f8b\u4e2d\uff0cchunk_size = 800\uff0cchunk_overlap = 150\u3002"
                        "\u5982\u679c\u9700\u8981\u8ba1\u7b97\uff0c\u8bf7\u5148\u68c0\u7d22\u518d\u6c42\u503c\u3002"
                    ).encode("utf-8"),
                    "text/plain",
                ),
            )
        ],
    )
    multistep_upload_json = multistep_upload_response.json()
    checks["local_kb_multistep_upload"] = {
        "status_code": multistep_upload_response.status_code,
        "knowledge_base_name": multistep_upload_json.get("knowledge_base_name"),
        "files_processed": multistep_upload_json.get("files_processed"),
    }
    assert multistep_upload_response.status_code == 200, "multistep local knowledge base upload failed"

    multistep_agent_response = client.post(
        "/chat/agent",
        json={
            "query": "\u5148\u67e5\u77e5\u8bc6\u5e93\u91cc\u7684 chunk_size \u548c chunk_overlap\uff0c\u518d\u8ba1\u7b97\u5b83\u4eec\u7684\u548c\u3002",
            "knowledge_base_name": multistep_kb_name,
            "allowed_tools": ["search_local_knowledge", "calculate"],
            "max_steps": 4,
        },
    )
    multistep_agent_json = multistep_agent_response.json()
    checks["agent_multistep_search_then_calculate"] = {
        "status_code": multistep_agent_response.status_code,
        "tool_names": _tool_names_in_order(multistep_agent_json),
        "step_kinds": _step_kinds(multistep_agent_json),
        "answer": multistep_agent_json.get("answer"),
    }
    assert multistep_agent_response.status_code == 200, "multistep agent request failed"
    assert _tool_names_in_order(multistep_agent_json)[:2] == ["search_local_knowledge", "calculate"], "multistep agent did not execute search then calculate"
    assert "950" in str(multistep_agent_json.get("answer", "")), "multistep agent answer missing calculated result"

    multistep_guard_response = client.post(
        "/chat/agent",
        json={
            "query": "\u5148\u67e5\u77e5\u8bc6\u5e93\u91cc\u7684 chunk_size \u548c chunk_overlap\uff0c\u518d\u8ba1\u7b97\u5b83\u4eec\u7684\u548c\u3002",
            "knowledge_base_name": multistep_kb_name,
            "allowed_tools": ["search_local_knowledge", "calculate"],
            "max_steps": 1,
        },
    )
    multistep_guard_json = multistep_guard_response.json()
    checks["agent_multistep_max_steps_guard"] = {
        "status_code": multistep_guard_response.status_code,
        "tool_names": _tool_names_in_order(multistep_guard_json),
        "step_kinds": _step_kinds(multistep_guard_json),
        "answer": multistep_guard_json.get("answer"),
    }
    assert multistep_guard_response.status_code == 200, "multistep max_steps guard request failed"
    assert _tool_names_in_order(multistep_guard_json) == ["search_local_knowledge"], "max_steps guard unexpectedly executed more than one tool"
    assert "最大工具步数" in str(multistep_guard_json.get("answer", "")), "max_steps guard answer missing stop hint"

    upload_response = client.post(
        "/knowledge_base/upload",
        data={"scope": "temp", "knowledge_base_name": ""},
        files=[
            (
                "files",
                (
                    "phase7_temp.txt",
                    (
                        "\u4e34\u65f6\u6587\u4ef6\u95ee\u7b54\u80fd\u5728\u4e0d\u843d\u5730\u957f\u671f\u77e5\u8bc6\u5e93\u7684"
                        "\u60c5\u51b5\u4e0b\uff0c\u76f4\u63a5\u5bf9\u4e0a\u4f20\u6587\u4ef6\u6784\u5efa\u4e34\u65f6 FAISS \u7d22\u5f15\u3002"
                    ).encode("utf-8"),
                    "text/plain",
                ),
            )
        ],
    )
    upload_json = upload_response.json()
    knowledge_id = upload_json["knowledge_id"]
    checks["temp_upload"] = {
        "status_code": upload_response.status_code,
        "knowledge_id": knowledge_id,
        "files_processed": upload_json.get("files_processed"),
        "expires_at": upload_json.get("expires_at"),
    }
    assert upload_response.status_code == 200, "temp upload failed"

    temp_rag_response = client.post(
        "/chat/rag",
        json={
            "query": "\u4e34\u65f6\u6587\u4ef6\u95ee\u7b54\u662f\u600e\u4e48\u5de5\u4f5c\u7684\uff1f",
            "source_type": "temp_kb",
            "knowledge_id": knowledge_id,
            "score_threshold": 0.0,
        },
    )
    temp_rag_json = temp_rag_response.json()
    checks["temp_kb_rag"] = {
        "status_code": temp_rag_response.status_code,
        "used_context": temp_rag_json.get("used_context"),
        "reference_count": len(temp_rag_json.get("references", [])),
        "first_source": _first_source(temp_rag_json),
    }
    assert temp_rag_response.status_code == 200, "temp_kb rag failed"
    assert temp_rag_json.get("used_context"), "temp_kb rag did not use context"

    not_expired_cleanup = cleanup_temp_knowledge_bases(
        settings,
        knowledge_id=knowledge_id,
        expired_only=True,
        cleanup_reason="validation_not_expired",
    )
    checks["temp_cleanup_not_expired"] = {
        "scanned": not_expired_cleanup.scanned,
        "removed": not_expired_cleanup.removed,
        "entry_reason": _first_cleanup_reason(not_expired_cleanup.model_dump()),
    }
    assert not_expired_cleanup.removed == 0, "cleanup removed a non-expired temp knowledge base"

    manual_cleanup_upload = client.post(
        "/knowledge_base/upload",
        data={"scope": "temp", "knowledge_base_name": ""},
        files=[
            (
                "files",
                (
                    "phase7_cleanup.txt",
                    "\u8fd9\u4e2a\u4e34\u65f6\u77e5\u8bc6\u5e93\u7528\u4e8e\u9a8c\u8bc1\u624b\u52a8\u8fc7\u671f\u6e05\u7406\u3002".encode("utf-8"),
                    "text/plain",
                ),
            )
        ],
    )
    manual_cleanup_json = manual_cleanup_upload.json()
    manual_cleanup_knowledge_id = manual_cleanup_json["knowledge_id"]
    assert manual_cleanup_upload.status_code == 200, "manual cleanup temp upload failed"
    force_expire_temp_knowledge(settings, manual_cleanup_knowledge_id)
    manual_cleanup_result = cleanup_temp_knowledge_bases(
        settings,
        knowledge_id=manual_cleanup_knowledge_id,
        expired_only=True,
        cleanup_reason="validation_manual_cleanup",
    )
    checks["temp_cleanup_manual_expired"] = {
        "scanned": manual_cleanup_result.scanned,
        "removed": manual_cleanup_result.removed,
        "entry_reason": _first_cleanup_reason(manual_cleanup_result.model_dump()),
    }
    assert manual_cleanup_result.removed == 1, "expired temp knowledge base was not removed by manual cleanup"

    startup_cleanup_upload = client.post(
        "/knowledge_base/upload",
        data={"scope": "temp", "knowledge_base_name": ""},
        files=[
            (
                "files",
                (
                    "phase7_startup_cleanup.txt",
                    "\u8fd9\u4e2a\u4e34\u65f6\u77e5\u8bc6\u5e93\u7528\u4e8e\u9a8c\u8bc1\u542f\u52a8\u65f6\u81ea\u52a8\u6e05\u7406\u3002".encode("utf-8"),
                    "text/plain",
                ),
            )
        ],
    )
    startup_cleanup_json = startup_cleanup_upload.json()
    startup_cleanup_knowledge_id = startup_cleanup_json["knowledge_id"]
    assert startup_cleanup_upload.status_code == 200, "startup cleanup temp upload failed"
    force_expire_temp_knowledge(settings, startup_cleanup_knowledge_id)
    startup_cleanup_result = maybe_run_startup_cleanup(settings, startup_name="validation")
    assert startup_cleanup_result is not None, "startup cleanup did not run"
    checks["temp_cleanup_startup"] = {
        "scanned": startup_cleanup_result.scanned,
        "removed": startup_cleanup_result.removed,
        "target_removed": _cleanup_entry_removed(
            startup_cleanup_result.model_dump(),
            startup_cleanup_knowledge_id,
        ),
    }
    assert _cleanup_entry_removed(
        startup_cleanup_result.model_dump(),
        startup_cleanup_knowledge_id,
    ), "expired temp knowledge base was not removed by startup cleanup"

    expired_access_upload = client.post(
        "/knowledge_base/upload",
        data={"scope": "temp", "knowledge_base_name": ""},
        files=[
            (
                "files",
                (
                    "phase7_expired_access.txt",
                    "\u8fd9\u4e2a\u4e34\u65f6\u77e5\u8bc6\u5e93\u7528\u4e8e\u9a8c\u8bc1\u8fc7\u671f\u540e\u8bbf\u95ee\u62a5\u9519\u3002".encode("utf-8"),
                    "text/plain",
                ),
            )
        ],
    )
    expired_access_json = expired_access_upload.json()
    expired_access_knowledge_id = expired_access_json["knowledge_id"]
    assert expired_access_upload.status_code == 200, "expired access temp upload failed"
    force_expire_temp_knowledge(settings, expired_access_knowledge_id)
    expired_access_response = client.post(
        "/chat/rag",
        json={
            "query": "\u8fd9\u4e2a\u4e34\u65f6\u77e5\u8bc6\u5e93\u8fd8\u80fd\u7528\u5417\uff1f",
            "source_type": "temp_kb",
            "knowledge_id": expired_access_knowledge_id,
        },
    )
    expired_access_json = expired_access_response.json()
    checks["temp_kb_expired_error"] = {
        "status_code": expired_access_response.status_code,
        "payload": expired_access_json,
    }
    assert expired_access_response.status_code == 410, "expired temp knowledge access did not return 410"
    assert expired_access_json.get("code") == "temp_knowledge_expired", "expired temp knowledge error code mismatch"

    agent_response = client.post(
        "/chat/agent",
        json={
            "query": "\u4ec0\u4e48\u662f RAG\uff1f",
            "knowledge_base_name": "phase2_demo",
        },
    )
    agent_json = agent_response.json()
    checks["agent_search_local_knowledge"] = {
        "status_code": agent_response.status_code,
        "used_tools": agent_json.get("used_tools"),
        "tool_name": _first_tool(agent_json),
        "reference_count": len(agent_json.get("references", [])),
    }
    assert agent_response.status_code == 200, "agent knowledge search failed"
    assert agent_json.get("used_tools"), "agent did not use tools"

    time_agent_response = client.post(
        "/chat/agent",
        json={"query": "\u73b0\u5728\u51e0\u70b9\u4e86\uff1f"},
    )
    time_agent_json = time_agent_response.json()
    checks["agent_current_time"] = {
        "status_code": time_agent_response.status_code,
        "used_tools": time_agent_json.get("used_tools"),
        "tool_name": _first_tool(time_agent_json),
    }
    assert time_agent_response.status_code == 200, "agent current_time failed"

    multistep_stream_status, multistep_stream_events = parse_stream_events(
        client,
        "/chat/agent",
        {
            "query": "\u5148\u67e5\u77e5\u8bc6\u5e93\u91cc\u7684 chunk_size \u548c chunk_overlap\uff0c\u518d\u8ba1\u7b97\u5b83\u4eec\u7684\u548c\u3002",
            "knowledge_base_name": multistep_kb_name,
            "allowed_tools": ["search_local_knowledge", "calculate"],
            "max_steps": 4,
            "stream": True,
        },
    )
    checks["agent_multistep_stream"] = {
        "status_code": multistep_stream_status,
        "event_types": [item["type"] for item in multistep_stream_events[:10]],
        "tool_call_count": sum(1 for item in multistep_stream_events if item["type"] == "tool_call"),
        "step_count": sum(1 for item in multistep_stream_events if item["type"] == "step"),
        "last_event_type": multistep_stream_events[-1]["type"] if multistep_stream_events else "",
    }
    assert multistep_stream_status == 200, "multistep stream failed"
    assert sum(1 for item in multistep_stream_events if item["type"] == "tool_call") >= 2, "multistep stream did not emit two tool calls"
    assert multistep_stream_events and multistep_stream_events[-1]["type"] == "done", "multistep stream missing done event"

    agent_stream_status, agent_stream_events = parse_stream_events(
        client,
        "/chat/agent",
        {
            "query": "\u73b0\u5728\u51e0\u70b9\u4e86\uff1f",
            "stream": True,
        },
    )
    checks["agent_current_time_stream"] = {
        "status_code": agent_stream_status,
        "event_types": [item["type"] for item in agent_stream_events[:6]],
        "last_event_type": agent_stream_events[-1]["type"] if agent_stream_events else "",
    }
    assert agent_stream_status == 200, "agent current_time stream failed"
    assert any(item["type"] == "tool_call" for item in agent_stream_events), "agent stream has no tool_call event"
    assert agent_stream_events and agent_stream_events[-1]["type"] == "done", "agent stream missing done event"

    validation_error_response = client.post(
        "/chat/rag",
        json={"query": "\u6d4b\u8bd5", "source_type": "local_kb"},
    )
    checks["validation_error_shape"] = {
        "status_code": validation_error_response.status_code,
        "payload": validation_error_response.json(),
    }
    assert validation_error_response.status_code == 422, "validation error status mismatch"

    checks["agent_memory_offline"] = run_agent_memory_offline_block(settings)

    return checks


def run_ui_checks() -> dict[str, object]:
    port = 8012
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_health(base_url)

        at = AppTest.from_file(str(PROJECT_ROOT / "app" / "ui" / "app.py"))
        at.run(timeout=30)

        find_by_label(at.text_input, "API Base URL").input(base_url).run(timeout=30)
        find_by_label(at.button, "\u5237\u65b0\u77e5\u8bc6\u5e93\u5217\u8868").click().run(timeout=30)
        kb_selectbox = find_by_label(at.selectbox, "选择知识库")
        knowledge_names = list(getattr(kb_selectbox, "options", []))
        local_upload_button_present = any(
            getattr(button, "label", None) == "\u4e0a\u4f20\u957f\u671f\u77e5\u8bc6\u5e93\u6587\u4ef6"
            for button in at.button
        )
        agent_max_steps_present = any(
            getattr(slider, "label", None) == "Agent max_steps"
            for slider in at.slider
        )

        find_by_label(at.text_area, "\u95ee\u9898").input("\u4ec0\u4e48\u662f RAG\uff1f").run(timeout=30)
        find_by_label(at.button, "\u53d1\u9001 RAG \u8bf7\u6c42").click().run(timeout=120)
        rag_success_messages = [item.value for item in at.success]

        find_by_label(at.text_area, "Agent \u95ee\u9898").input("\u73b0\u5728\u51e0\u70b9\u4e86\uff1f").run(timeout=30)
        find_by_label(at.button, "\u53d1\u9001 Agent \u8bf7\u6c42").click().run(timeout=120)
        agent_success_messages = [item.value for item in at.success]
        assert local_upload_button_present, "local knowledge base upload button missing"
        assert agent_max_steps_present, "agent max_steps slider missing"

        return {
            "knowledge_base_names": knowledge_names,
            "local_upload_button_present": local_upload_button_present,
            "agent_max_steps_present": agent_max_steps_present,
            "tab_count": len(at.tabs),
            "rag_success_messages": rag_success_messages,
            "agent_success_messages": agent_success_messages,
        }
    finally:
        server.should_exit = True
        server.force_exit = True
        thread.join(timeout=10)


def wait_for_health(base_url: str) -> None:
    for _ in range(50):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
        except requests.RequestException:
            time.sleep(0.2)
            continue
        if response.ok:
            return
        time.sleep(0.2)
    raise RuntimeError("API server did not start in time")


def find_by_label(elements, label: str):
    for element in elements:
        if getattr(element, "label", None) == label:
            return element
    raise AssertionError(f"Element not found: {label}")


def _first_source(payload: dict[str, object]) -> str:
    references = payload.get("references", [])
    if not isinstance(references, list) or not references:
        return ""
    first = references[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("source", ""))


def _first_reference(payload: dict[str, object]) -> dict[str, object]:
    references = payload.get("references", [])
    if not isinstance(references, list) or not references:
        return {}
    first = references[0]
    if not isinstance(first, dict):
        return {}
    return first


def _build_retrieved_reference(
    *,
    chunk_id: str,
    source: str,
    content: str,
    relevance_score: float,
) -> object:
    from app.schemas.chat import RetrievedReference

    return RetrievedReference(
        chunk_id=chunk_id,
        source=source,
        source_path=source,
        extension=".txt",
        content=content,
        content_preview=content[:120],
        raw_score=max(0.0, 1.0 - relevance_score),
        relevance_score=relevance_score,
        evidence_type="text",
        source_modality="text",
    )


def _first_tool(payload: dict[str, object]) -> str:
    tool_calls = payload.get("tool_calls", [])
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    first = tool_calls[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("tool_name", ""))


def _tool_names_in_order(payload: dict[str, object]) -> list[str]:
    tool_calls = payload.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        names.append(str(item.get("tool_name", "")))
    return names


def _step_kinds(payload: dict[str, object]) -> list[str]:
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        return []
    kinds: list[str] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        kinds.append(str(item.get("kind", "")))
    return kinds


def _first_cleanup_reason(payload: dict[str, object]) -> str:
    entries = payload.get("entries", [])
    if not isinstance(entries, list) or not entries:
        return ""
    first = entries[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("reason", ""))


def _cleanup_entry_removed(payload: dict[str, object], knowledge_id: str) -> bool:
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("knowledge_id", "")) != knowledge_id:
            continue
        return bool(entry.get("removed", False))
    return False


def force_expire_temp_knowledge(settings, knowledge_id: str) -> None:
    manifest = load_temp_manifest(settings, knowledge_id)
    expired_at = manifest.created_at - timedelta(minutes=1)
    updated_manifest = manifest.model_copy(
        update={
            "last_accessed_at": expired_at,
            "expires_at": expired_at,
        }
    )
    write_temp_manifest(settings, updated_manifest)


def reset_local_knowledge_base(settings, knowledge_base_name: str) -> None:
    knowledge_base_dir = settings.knowledge_base_root / knowledge_base_name
    vector_store_dir = settings.vector_store_dir(knowledge_base_name)
    if knowledge_base_dir.exists():
        shutil.rmtree(knowledge_base_dir, onerror=_handle_rmtree_error)
    if vector_store_dir.exists():
        shutil.rmtree(vector_store_dir, onerror=_handle_rmtree_error)


def write_local_knowledge_base_files(settings, knowledge_base_name: str, files: dict[str, str]) -> None:
    content_dir = settings.knowledge_base_content_dir(knowledge_base_name)
    content_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (content_dir / filename).write_text(content, encoding="utf-8")


def _handle_rmtree_error(func, path, exc_info) -> None:
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass
    func(path)


def parse_stream_events(
    client: TestClient,
    path: str,
    payload: dict[str, object],
) -> tuple[int, list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    current_event = ""
    with client.stream("POST", path, json=payload) as response:
        status_code = response.status_code
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line[len("event: ") :]
                continue
            if line.startswith("data: "):
                payload_data = json.loads(line[len("data: ") :])
                if isinstance(payload_data, dict):
                    payload_data.setdefault("type", current_event)
                    events.append(payload_data)
    return status_code, events


if __name__ == "__main__":
    raise SystemExit(main())
