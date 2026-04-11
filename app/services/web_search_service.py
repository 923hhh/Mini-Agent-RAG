from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import re
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup

from app.schemas.chat import RetrievedReference
from app.services.network import build_requests_session
from app.services.settings import AppSettings


DEFAULT_DUCKDUCKGO_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DEFAULT_WEB_SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class WebSearchSnippet:
    title: str
    url: str
    snippet: str
    source_domain: str
    rank: int


def search_corrective_web_references(
    *,
    settings: AppSettings,
    query: str,
) -> list[RetrievedReference]:
    normalized_query = query.strip()
    if not normalized_query or not settings.kb.ENABLE_CORRECTIVE_WEB_SEARCH:
        return []

    snippets = search_web_snippets(
        settings=settings,
        query=normalized_query,
    )
    return build_web_references(settings, snippets)


def search_web_snippets(
    *,
    settings: AppSettings,
    query: str,
) -> list[WebSearchSnippet]:
    provider = settings.kb.CORRECTIVE_WEB_SEARCH_PROVIDER.strip().lower()
    if provider in {"duckduckgo_html", "duckduckgo", "ddg"}:
        return search_duckduckgo_html(
            query=query,
            limit=settings.kb.CORRECTIVE_WEB_SEARCH_TOP_K,
            endpoint=settings.kb.CORRECTIVE_WEB_SEARCH_ENDPOINT.strip() or DEFAULT_DUCKDUCKGO_HTML_ENDPOINT,
            timeout_seconds=settings.kb.CORRECTIVE_WEB_SEARCH_TIMEOUT_SECONDS,
        )
    raise ValueError(f"不支持的网络补搜 provider: {settings.kb.CORRECTIVE_WEB_SEARCH_PROVIDER}")


def search_duckduckgo_html(
    *,
    query: str,
    limit: int,
    endpoint: str,
    timeout_seconds: int,
) -> list[WebSearchSnippet]:
    session = build_requests_session()
    response = session.post(
        endpoint,
        data={"q": query},
        headers={"User-Agent": DEFAULT_WEB_SEARCH_USER_AGENT},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return parse_duckduckgo_html_results(response.text, limit=limit)


def parse_duckduckgo_html_results(
    html: str,
    *,
    limit: int,
) -> list[WebSearchSnippet]:
    soup = BeautifulSoup(html or "", "html.parser")
    snippets: list[WebSearchSnippet] = []
    seen_urls: set[str] = set()

    result_blocks = soup.select(".result")
    if result_blocks:
        for block in result_blocks:
            candidate = parse_duckduckgo_result_block(block, rank=len(snippets) + 1)
            if candidate is None or candidate.url in seen_urls:
                continue
            seen_urls.add(candidate.url)
            snippets.append(candidate)
            if len(snippets) >= limit:
                return snippets

    for anchor in soup.select("a[href]"):
        candidate = parse_duckduckgo_anchor_fallback(anchor, rank=len(snippets) + 1)
        if candidate is None or candidate.url in seen_urls:
            continue
        seen_urls.add(candidate.url)
        snippets.append(candidate)
        if len(snippets) >= limit:
            break

    return snippets


def parse_duckduckgo_result_block(
    block,
    *,
    rank: int,
) -> WebSearchSnippet | None:
    link = (
        block.select_one("a.result__a")
        or block.select_one(".result__title a")
        or block.select_one("a.result-link")
        or block.select_one("a[href]")
    )
    if link is None:
        return None

    url = normalize_search_result_url(str(link.get("href", "")))
    title = normalize_web_text(link.get_text(" ", strip=True))
    if not url or not title:
        return None

    snippet_node = (
        block.select_one(".result__snippet")
        or block.select_one(".result-snippet")
        or block.select_one(".snippet")
    )
    snippet = normalize_web_text(snippet_node.get_text(" ", strip=True) if snippet_node is not None else "")
    return build_web_search_snippet(
        title=title,
        url=url,
        snippet=snippet,
        rank=rank,
    )


def parse_duckduckgo_anchor_fallback(
    anchor,
    *,
    rank: int,
) -> WebSearchSnippet | None:
    url = normalize_search_result_url(str(anchor.get("href", "")))
    title = normalize_web_text(anchor.get_text(" ", strip=True))
    if not url or not title:
        return None
    if title.lower() in {"next", "previous"}:
        return None

    parent_text = normalize_web_text(anchor.parent.get_text(" ", strip=True) if anchor.parent is not None else "")
    snippet = parent_text.replace(title, "", 1).strip(" -:|")
    return build_web_search_snippet(
        title=title,
        url=url,
        snippet=snippet,
        rank=rank,
    )


def build_web_search_snippet(
    *,
    title: str,
    url: str,
    snippet: str,
    rank: int,
) -> WebSearchSnippet | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return WebSearchSnippet(
        title=title,
        url=url,
        snippet=snippet,
        source_domain=parsed.netloc.lower().removeprefix("www."),
        rank=rank,
    )


def normalize_search_result_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("//"):
        cleaned = f"https:{cleaned}"

    parsed = urlparse(cleaned)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.rstrip("/") == "/l":
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return cleaned


def normalize_web_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def build_web_references(
    settings: AppSettings,
    snippets: list[WebSearchSnippet],
) -> list[RetrievedReference]:
    references: list[RetrievedReference] = []
    max_chars = settings.kb.CORRECTIVE_WEB_SEARCH_SNIPPET_MAX_CHARS
    for snippet in snippets:
        content = build_web_result_content(snippet)
        preview = content[:max_chars].rstrip()
        relevance_score = max(0.3, 0.68 - (snippet.rank - 1) * 0.08)
        references.append(
            RetrievedReference(
                chunk_id=f"web::{sha1(snippet.url.encode('utf-8')).hexdigest()[:16]}",
                source=(snippet.title or snippet.source_domain)[:120],
                source_path=snippet.url,
                extension=".html",
                title=snippet.title[:160],
                section_title=snippet.source_domain,
                section_path=snippet.url,
                content_type="web_search_result",
                source_modality="text",
                evidence_type="web",
                used_for_answer=True,
                original_file_type="html",
                evidence_summary=f"{snippet.source_domain} 网络结果摘要",
                headers={},
                content=content,
                content_preview=preview or content[:240],
                raw_score=max(0.0, 1.0 - relevance_score),
                relevance_score=relevance_score,
            )
        )
    return references


def build_web_result_content(snippet: WebSearchSnippet) -> str:
    lines = [
        f"title: {snippet.title}",
        f"url: {snippet.url}",
    ]
    if snippet.snippet:
        lines.append(f"snippet: {snippet.snippet}")
    return "\n".join(lines).strip()
