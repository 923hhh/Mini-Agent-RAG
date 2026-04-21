"""管理 Agent 记忆的写入、检索与压缩。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.schemas.chat import MemoryOverview
from app.services.models.embedding_service import build_embeddings, embed_texts_batched
from app.services.models.llm_service import build_chat_model
from app.services.core.settings import AppSettings

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

EPISODE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是会话摘要助手。根据对话摘录生成简短标题与摘要，便于后续检索。"
            "只输出一行 JSON："
            '{"title":"<=30字","summary":"<=200字"}',
        ),
        ("human", "对话摘录：\n{transcript}"),
    ]
)

SEMANTIC_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是长期记忆抽取器。只输出 JSON 数组；每项包含 "
            'fact（一句陈述）、category（preference|goal|constraint|identity|project|other）、'
            "confidence（0到1的小数）。若无值得长期保存的事实，输出 []。",
        ),
        ("human", "以下是一段会话摘要，请抽取可跨会话复用的长期事实：\n{summary}"),
    ]
)


@dataclass(frozen=True)
class AgentMemoryContext:
    """检索得到的注入文本与统计。"""

    text: str
    overview: MemoryOverview


def sanitize_session_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    if not _SESSION_ID_PATTERN.fullmatch(s):
        raise ValueError(
            "session_id 无效：应为 1–128 位，仅含字母、数字、点、下划线或连字符。"
        )
    return s


def agent_memory_enabled(settings: AppSettings) -> bool:
    return bool(settings.basic.ENABLE_AGENT_MEMORY)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _open_episode_id(meta: dict[str, Any]) -> str:
    return _next_id("ep", int(meta["episode_counter"]))


def _load_meta(session_dir: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "next_turn_seq": 0,
        "episode_counter": 1,
        "next_semantic_seq": 0,
        "turns_in_open_episode": 0,
    }
    path = session_dir / "meta.json"
    if not path.is_file():
        return dict(defaults)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError
    except (OSError, ValueError, json.JSONDecodeError):
        return dict(defaults)

    data.setdefault("next_turn_seq", 0)
    data.setdefault("next_semantic_seq", 0)
    data.setdefault("turns_in_open_episode", 0)
    if "episode_counter" not in data:
        oid = str(data.get("open_episode_id", "ep-000001"))
        match = re.match(r"^ep-(\d+)$", oid)
        data["episode_counter"] = int(match.group(1)) if match else 1
    return data


def _save_meta(session_dir: Path, meta: dict[str, Any]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_id(prefix: str, seq: int) -> str:
    return f"{prefix}-{seq:06d}"


def _memory_trace(settings: AppSettings, name: str, payload: dict[str, Any]) -> None:
    if not agent_memory_enabled(settings):
        return
    trace_path = settings.log_root / f"{name}.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cosine_top_k(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    k: int,
) -> list[int]:
    if matrix.size == 0 or k <= 0:
        return []
    q = query_vec.astype(np.float64)
    m = matrix.astype(np.float64)
    qn = q / (np.linalg.norm(q) + 1e-9)
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    scores = mn @ qn
    order = np.argsort(-scores)
    return [int(i) for i in order[:k]]


def _budget_trim(parts: list[str], budget: int) -> str:
    text = "\n\n".join(p for p in parts if p.strip())
    if len(text) <= budget:
        return text
    return text[: budget - 3].rstrip() + "..."


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_json_array(text: str) -> list[Any]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def retrieve_agent_memory(
    settings: AppSettings,
    *,
    session_id: str,
    query: str,
) -> AgentMemoryContext:
    overview = MemoryOverview()
    if not agent_memory_enabled(settings):
        return AgentMemoryContext(text="", overview=overview)

    session_dir = settings.agent_memory_session_dir(session_id)
    semantics = _read_jsonl(session_dir / "semantics.jsonl")
    episodes = _read_jsonl(session_dir / "episodes.jsonl")
    turns_path = session_dir / "turns.jsonl"

    if not semantics and not episodes and not turns_path.is_file():
        _memory_trace(
            settings,
            "memory_retrieval_trace",
            {
                "session_id": session_id,
                "semantic_hits": 0,
                "episode_hits": 0,
                "turn_hits": 0,
                "used_memory": False,
                "note": "empty_store",
            },
        )
        return AgentMemoryContext(text="", overview=overview)

    try:
        embeddings = build_embeddings(settings)
    except Exception as exc:
        _memory_trace(
            settings,
            "memory_retrieval_trace",
            {
                "session_id": session_id,
                "error": str(exc),
                "used_memory": False,
            },
        )
        return AgentMemoryContext(text="", overview=overview)

    q_vec = np.array(embeddings.embed_query(query), dtype=np.float32)
    batch_size = max(1, settings.kb.EMBEDDING_BATCH_SIZE)

    semantic_lines: list[str] = []
    episode_lines: list[str] = []
    turn_snippets: list[str] = []

    semantic_rows = [row for row in semantics if str(row.get("fact", "")).strip()]
    sem_texts = [str(row.get("fact", "")).strip() for row in semantic_rows]
    if sem_texts:
        sem_matrix = np.array(
            embed_texts_batched(embeddings, sem_texts, batch_size),
            dtype=np.float32,
        )
    else:
        sem_matrix = np.zeros((0, q_vec.shape[0]), dtype=np.float32)
    sem_indices = _cosine_top_k(q_vec, sem_matrix, settings.basic.AGENT_MEMORY_SEMANTIC_TOP_K)
    hit_episode_ids: set[str] = set()
    for idx in sem_indices:
        if 0 <= idx < len(semantic_rows):
            row = semantic_rows[idx]
            fact = str(row.get("fact", "")).strip()
            semantic_lines.append(f"- {fact}")
            eid = str(row.get("episode_id", "")).strip()
            if eid:
                hit_episode_ids.add(eid)

    ep_summaries = [
        (str(row.get("episode_id", "")).strip(), str(row.get("summary", "")).strip())
        for row in episodes
        if str(row.get("summary", "")).strip()
    ]
    ep_ids = [pair[0] for pair in ep_summaries]
    ep_texts = [pair[1] for pair in ep_summaries]
    if ep_texts:
        ep_matrix = np.array(embed_texts_batched(embeddings, ep_texts, batch_size), dtype=np.float32)
        qn = q_vec.astype(np.float64)
        qn = qn / (np.linalg.norm(qn) + 1e-9)
        mn = ep_matrix.astype(np.float64)
        mn = mn / (np.linalg.norm(mn, axis=1, keepdims=True) + 1e-9)
        ep_scores = (mn @ qn).tolist()
        order = sorted(
            range(len(ep_ids)),
            key=lambda i: (0 if ep_ids[i] in hit_episode_ids else 1, -ep_scores[i]),
        )
        k_ep = settings.basic.AGENT_MEMORY_EPISODE_TOP_K
        seen_ep: set[str] = set()
        for idx in order:
            if len(episode_lines) >= k_ep:
                break
            eid, summary = ep_ids[idx], ep_texts[idx]
            if not eid or eid in seen_ep:
                continue
            seen_ep.add(eid)
            title = ""
            for row in episodes:
                if str(row.get("episode_id", "")).strip() == eid:
                    title = str(row.get("title", "")).strip()
                    break
            label = f"{title} ({eid})" if title else eid
            episode_lines.append(f"- [{label}] {summary}")

    if settings.basic.AGENT_MEMORY_ENABLE_TURN_EXPANSION and turns_path.is_file():
        recent = _read_jsonl(turns_path)[-6:]
        for row in recent:
            role = str(row.get("role", ""))
            content = str(row.get("content", "")).strip()
            if content:
                turn_snippets.append(f"{role}: {content[:400]}")

    parts: list[str] = []
    if semantic_lines:
        parts.append("语义层事实：\n" + "\n".join(semantic_lines))
    if episode_lines:
        parts.append("相关会话摘要：\n" + "\n".join(episode_lines))
    if turn_snippets:
        parts.append("近期原话摘录：\n" + "\n".join(turn_snippets))

    text = _budget_trim(parts, settings.basic.AGENT_MEMORY_CONTEXT_CHAR_BUDGET)
    used = bool(text.strip())
    overview = MemoryOverview(
        semantic_hits=len(semantic_lines),
        episode_hits=len(episode_lines),
        turn_hits=len(turn_snippets),
        used_memory=used,
    )
    _memory_trace(
        settings,
        "memory_retrieval_trace",
        {
            "session_id": session_id,
            "semantic_hits": overview.semantic_hits,
            "episode_hits": overview.episode_hits,
            "turn_hits": overview.turn_hits,
            "used_memory": used,
            "memory_chars": len(text),
        },
    )
    return AgentMemoryContext(text=text, overview=overview)


def _run_episode_llm(settings: AppSettings, transcript: str) -> tuple[str, str]:
    llm = build_chat_model(
        settings,
        model_name=settings.model.AGENT_MODEL,
        temperature=0.0,
    )
    chain = EPISODE_SUMMARY_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"transcript": transcript[:6000]})
    data = _extract_json_object(raw)
    if data:
        title = str(data.get("title", "")).strip() or "会话摘要"
        summary = str(data.get("summary", "")).strip() or transcript[:200]
        return title, summary
    return "会话摘要", transcript[:280].replace("\n", " ")


def _run_semantic_llm(settings: AppSettings, summary: str) -> list[dict[str, Any]]:
    llm = build_chat_model(
        settings,
        model_name=settings.model.AGENT_MODEL,
        temperature=0.0,
    )
    chain = SEMANTIC_EXTRACT_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"summary": summary[:4000]})
    items = _extract_json_array(raw)
    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact", "")).strip()
        if not fact:
            continue
        results.append(
            {
                "fact": fact,
                "category": str(item.get("category", "other")).strip() or "other",
                "confidence": float(item.get("confidence", 0.7)),
            }
        )
    return results[:12]


def _finalize_open_episode(settings: AppSettings, session_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    episode_id = _open_episode_id(meta)
    turns = _read_jsonl(session_dir / "turns.jsonl")
    episode_turns = [row for row in turns if str(row.get("episode_id", "")) == episode_id]
    if not episode_turns:
        meta["turns_in_open_episode"] = 0
        return meta

    lines: list[str] = []
    for row in episode_turns:
        role = row.get("role", "")
        content = str(row.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    transcript = "\n".join(lines)

    try:
        title, summary = _run_episode_llm(settings, transcript)
    except Exception:
        title, summary = "会话摘要", transcript[:280].replace("\n", " ")

    episode_record = {
        "episode_id": episode_id,
        "title": title,
        "summary": summary,
        "turn_ids": [row.get("turn_id") for row in episode_turns if row.get("turn_id")],
        "parent_ids": [],
        "children_ids": [],
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(session_dir / "episodes.jsonl", episode_record)
    new_episodes = 1

    try:
        facts = _run_semantic_llm(settings, summary)
    except Exception:
        facts = []

    next_sem = int(meta["next_semantic_seq"])
    new_semantics = 0
    for fact_row in facts:
        next_sem += 1
        semantic_id = _next_id("sem", next_sem)
        _append_jsonl(
            session_dir / "semantics.jsonl",
            {
                "semantic_id": semantic_id,
                "fact": fact_row["fact"],
                "episode_id": episode_id,
                "category": fact_row["category"],
                "confidence": fact_row["confidence"],
                "source_ids": [],
                "parent_ids": [],
                "children_ids": [],
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )
        new_semantics += 1
    meta["next_semantic_seq"] = next_sem

    meta["episode_counter"] = int(meta["episode_counter"]) + 1
    meta["turns_in_open_episode"] = 0

    _memory_trace(
        settings,
        "memory_build_trace",
        {
            "session_id": session_dir.name,
            "episode_id_closed": episode_id,
            "new_episodes": new_episodes,
            "new_semantics": new_semantics,
        },
    )
    return meta


def persist_agent_turns(
    settings: AppSettings,
    *,
    session_id: str,
    user_text: str,
    assistant_text: str,
    tools_used: list[str],
) -> None:
    if not agent_memory_enabled(settings):
        return

    session_dir = settings.agent_memory_session_dir(session_id)
    meta = _load_meta(session_dir)
    turn_base = int(meta["next_turn_seq"])
    episode_id = _open_episode_id(meta)
    ts = datetime.now(timezone.utc).isoformat()

    for offset, (role, content) in enumerate(
        (
            ("user", user_text),
            ("assistant", assistant_text),
        ),
        start=1,
    ):
        turn_base += 1
        turn_id = _next_id("t", turn_base)
        _append_jsonl(
            session_dir / "turns.jsonl",
            {
                "turn_id": turn_id,
                "episode_id": episode_id,
                "role": role,
                "content": content,
                "tools_used": tools_used if role == "assistant" else [],
                "ts": ts,
            },
        )

    meta["next_turn_seq"] = turn_base
    meta["turns_in_open_episode"] = int(meta["turns_in_open_episode"]) + 2

    max_turns = int(settings.basic.AGENT_MEMORY_EPISODE_MAX_TURNS)
    if meta["turns_in_open_episode"] >= max_turns:
        meta = _finalize_open_episode(settings, session_dir, meta)

    _save_meta(session_dir, meta)

    _memory_trace(
        settings,
        "memory_build_trace",
        {
            "session_id": session_id,
            "turns_appended": 2,
            "open_episode_id": _open_episode_id(meta),
            "turns_in_open_episode": meta["turns_in_open_episode"],
        },
    )

