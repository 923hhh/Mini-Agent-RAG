"""索引时为每个 chunk 批量生成可检索问题。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.services.core.settings import AppSettings
from app.services.models.llm_service import build_chat_model


CHUNK_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是知识库索引助手。给定一段文本，生成 1-2 个该文本能直接回答的问题。\n"
            "要求：\n"
            "- 问题必须具体，包含文本中的关键实体（人名、机构、数字、时间）\n"
            "- 不要生成过于宽泛的问题\n"
            "- 每行一个问题，不要编号，不要解释",
        ),
        ("human", "文本：\n{chunk_content}"),
    ]
)


def generate_chunk_questions(
    settings: AppSettings,
    chunks: list[Document],
    *,
    progress_callback=None,
) -> list[Document]:
    if not settings.kb.ENABLE_CHUNK_QUESTION_GENERATION:
        return chunks
    if not chunks:
        return chunks

    batch_size = settings.kb.CHUNK_QUESTION_BATCH_SIZE
    max_questions = settings.kb.CHUNK_QUESTION_MAX_PER_CHUNK
    llm = build_chat_model(settings, temperature=0.0)
    chain = CHUNK_QUESTION_PROMPT | llm | StrOutputParser()

    total = len(chunks)
    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        for chunk in batch:
            questions = _generate_questions_for_chunk(chain, chunk, max_questions)
            chunk.metadata["generated_questions"] = questions
        if progress_callback is not None:
            progress_callback(min(batch_start + batch_size, total), total)

    return chunks


def _generate_questions_for_chunk(
    chain,
    chunk: Document,
    max_questions: int,
) -> list[str]:
    content = chunk.page_content.strip()
    if len(content) < 20:
        return []
    try:
        result = chain.invoke({"chunk_content": content[:1500]})
    except Exception:
        return []
    lines = [line.strip() for line in result.strip().splitlines() if line.strip()]
    questions = [line for line in lines if line.endswith("？") or line.endswith("?")]
    if not questions:
        questions = lines
    return questions[:max_questions]
