"""组织 RAG 主链路的检索、生成与审校流程。"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.schemas.chat import ChatMessage, RetrievedReference
from app.services.core.settings import AppSettings
from app.services.models.llm_service import build_chat_model
from app.services.models.streaming_llm import stream_prompt_output
from app.services.retrieval.answer_guard_service import (
    maybe_refine_rag_answer,
)
from app.services.runtime.rag_runtime_service import (
    RetrievalCoverageGrade,
    append_answer_trace,
    append_corrective_trace,
    build_rag_prompt,
    build_rag_variables,
    convert_history,
    generate_corrective_query,
    grade_documents,
    merge_corrective_references,
    resolve_rag_prompt_kind,
)


RAG_SYSTEM_PROMPT = """你是一个基于本地知识库的问答助手。
请严格依据提供的上下文回答用户问题。

【引用规则——必须遵守】
- 回答中每个事实性陈述必须在句末标注来源编号，格式为 [证据N]，N 对应上下文中的证据序号。
- 一句话可以标注多个来源，如 [证据1][证据3]。
- 如果某个事实在上下文中找不到支撑，不要写出来，改为"根据当前检索到的内容，无法确定该点"。
- 宁可少答也不要编造。未标注来源的事实性陈述视为无效。

【回答风格】
- 回答尽量简洁、准确，优先提炼要点。
- 如果问题是单一事实、单个时间点、单个对象属性或单个数值，第一句必须先直接给结论。
- 除非用户明确要求"依据 / 来源 / 为什么 / 从哪看出"，否则不要在正文里加入"根据上下文 / 根据资料"等模板化前缀（引用标注 [证据N] 除外）。
- 如果问题包含多个并列子问题、多个角色、多个原因、多个措施或"哪些/分别/同时/以及"等要求，必须逐项覆盖，不得遗漏。
- 若上下文已经明确列出了清单、步骤、职责、原因或措施，回答时应尽量完整保留这些要点。
- 优先使用分点或小标题作答，让每个子问题都有明确落点。

【多模态证据处理】
- 如果上下文包含"文本证据 / OCR 证据 / 视觉描述证据"，请区分哪些信息是直接证据，哪些只是推断。
- 不要把 OCR 识别文本与视觉描述混为同一类事实。
- 如果上下文同时包含"时间序列证据"和"文本证据"，回答时应分别说明趋势观察和文本结论。

【时间约束】
- 若问题包含"当前 / 最新 / 现任 / 截至某时间 / 哪一年 / 何时"等时间约束，回答时必须保留证据中的具体时间点。"""

IMAGE_RAG_SYSTEM_PROMPT = """你是一个基于本地知识库的多模态问答助手。
当前问题更偏向图片、OCR 或视觉描述理解，请优先依据提供的图片相关证据回答。
请遵守以下规则：
1. OCR 文字属于直接文本证据，但可能有识别错误；引用时不要自动纠正成你主观认为更合理的内容。
2. 视觉描述属于图像理解结果，只能用于"看到的画面特征"判断，不能当作绝对事实扩展。
3. 如果 OCR 证据和视觉描述证据冲突，必须明确指出冲突，不要强行合并。
4. 如果图片证据不足，请明确说明"根据当前检索到的图片相关证据，无法确定"，不要拿普通正文内容替代图片结论。
5. 回答尽量简洁，先说可确认内容，再说可能推断。"""

CORRECTIVE_RAG_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 RAG 检索质量评估器。"
            "你需要判断当前检索证据是否足以回答用户问题。"
            "只允许输出 JSON，不要输出额外解释。"
            'JSON schema: {"grade":"relevant|partial|insufficient","reason":"<=60字","missing_aspects":"<=60字"}。',
        ),
        (
            "human",
            "对话历史：\n{history_text}\n\n用户问题：\n{query}\n\n当前证据摘要：\n{reference_summary}\n\n"
            "请评估当前证据覆盖度。",
        ),
    ]
)

CORRECTIVE_RAG_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 RAG 的二次检索查询生成器。"
            "请基于原问题和当前证据缺口，输出一条更适合第二轮检索的短查询。"
            "必须保留型号、参数、专有名词和关键约束。"
            "不要回答问题，不要解释，只输出一行查询。",
        ),
        (
            "human",
            "对话历史：\n{history_text}\n\n用户问题：\n{query}\n\n"
            "当前分级：{grade}\n原因：{reason}\n缺失点：{missing_aspects}\n\n"
            "当前证据摘要：\n{reference_summary}\n\n"
            "请输出一条更适合二次检索的查询：",
        ),
    ]
)

RAG_UNIFIED_REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 RAG 答案审校器，同时负责完备性和事实性两项检查。"
            "你只能依据给定证据进行审校。\n\n"
            "【完备性检查】\n"
            "- 是否漏答了并列子问题、清单项、角色职责？\n"
            "- 若上下文同时包含时间序列证据和文本证据，答案是否分别覆盖了序列观察与文本背景？\n"
            "- 不要补充证据中不存在的事实。\n\n"
            "【事实性检查】\n"
            "- 每个事实性陈述必须有 [证据N] 标注且对应证据确实支持该陈述；\n"
            "- 标注了 [证据N] 但对应证据不支持的，属于错误引用，必须删除或修正；\n"
            "- 没有标注来源的事实性陈述，如果证据中能找到支撑则补上标注，否则删除；\n"
            "- 扩大解释、主语错配、时间外推、因果外推均视为不支持；\n"
            "- 保留有证据明确支持且正确标注的内容，不要过度删减。\n\n"
            "只允许输出 JSON，不要输出额外解释。\n"
            'JSON schema: {"status":"ok|revise","missing_points":["<=30字"],"unsupported_claims":["<=40字"],"revised_answer":"string"}。\n'
            "若当前答案已足够完整且忠实，status=ok，revised_answer 置空字符串。",
        ),
        (
            "human",
            "参考上下文：\n{context}\n\n{coverage_requirements}{answer_requirements}"
            "用户问题：\n{query}\n\n当前答案草稿：\n{draft_answer}\n\n"
            "请同时检查完备性（是否漏答）和事实性（引用是否正确），"
            "若需要修改，请直接给出修订后的最终答案。",
        ),
    ]
)

RAG_FACTUAL_REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 RAG 答案事实审校器。"
            "你只能依据给定证据逐句检查当前答案。审校规则："
            "1. 每个事实性陈述必须有 [证据N] 标注且对应证据确实支持该陈述；"
            "2. 标注了 [证据N] 但对应证据不支持该陈述的，属于错误引用，必须删除或修正；"
            "3. 没有标注来源的事实性陈述，如果证据中能找到支撑则补上标注，否则删除；"
            "4. 扩大解释、主语错配、时间外推、因果外推均视为不支持；"
            "5. 保留那些有证据明确支持且正确标注的内容，不要过度删减。"
            "只允许输出 JSON，不要输出额外解释。"
            'JSON schema: {"status":"ok|revise","unsupported_claims":["<=40字"],"revised_answer":"string"}。'
            "若当前答案已足够忠实，status=ok，revised_answer 置空字符串。",
        ),
        (
            "human",
            "参考上下文：\n{context}\n\n{coverage_requirements}{answer_requirements}"
            "用户问题：\n{query}\n\n当前答案：\n{draft_answer}\n\n"
            "请逐句验证引用标注是否正确，删除或改写没有被证据明确支持的内容，并输出最终答案。",
        ),
    ]
)


def _prepare_rag_invocation(
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    *,
    agent_memory_context: str,
) -> tuple[str, ChatPromptTemplate, dict[str, object]]:
    prompt_kind = resolve_rag_prompt_kind(query, references)
    prompt = build_rag_prompt(
        query,
        references,
        system_prompt=RAG_SYSTEM_PROMPT,
        image_system_prompt=IMAGE_RAG_SYSTEM_PROMPT,
    )
    variables = build_rag_variables(
        settings,
        query,
        references,
        history,
        agent_memory_context=agent_memory_context,
    )
    return prompt_kind, prompt, variables

def generate_rag_answer(
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    agent_memory_context: str = "",
) -> str:
    prompt_kind, prompt, variables = _prepare_rag_invocation(
        settings,
        query,
        references,
        history,
        agent_memory_context=agent_memory_context,
    )
    llm = build_chat_model(settings, temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(variables)
    answer = maybe_refine_rag_answer(
        settings=settings,
        completeness_review_prompt=RAG_UNIFIED_REVIEW_PROMPT,
        factual_review_prompt=RAG_FACTUAL_REVIEW_PROMPT,
        query=query,
        references=references,
        context=str(variables["context"]),
        coverage_requirements=str(variables["coverage_requirements"]),
        answer_requirements=str(variables["answer_requirements"]),
        draft_answer=answer,
    )
    append_answer_trace(
        settings=settings,
        query=query,
        references=references,
        prompt_kind=prompt_kind,
        answer=answer,
    )
    return answer


def stream_rag_answer(
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    agent_memory_context: str = "",
) -> Iterator[str]:
    prompt_kind, prompt, variables = _prepare_rag_invocation(
        settings,
        query,
        references,
        history,
        agent_memory_context=agent_memory_context,
    )
    answer_parts: list[str] = []
    for delta in stream_prompt_output(settings, prompt, variables, temperature=0.0):
        answer_parts.append(delta)
        yield delta
    append_answer_trace(
        settings=settings,
        query=query,
        references=references,
        prompt_kind=prompt_kind,
        answer="".join(answer_parts),
    )


def maybe_run_corrective_retrieval(
    *,
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    top_k: int,
    score_threshold: float,
    retrieve: Callable[[str, int, float], list[RetrievedReference]],
    search_web: Callable[[str], list[RetrievedReference]] | None = None,
    source_type: str,
    target_name: str,
) -> list[RetrievedReference]:
    if not settings.kb.ENABLE_CORRECTIVE_RAG:
        return references

    initial_grade = grade_documents(
        settings=settings,
        query=query,
        references=references,
        history=history,
        corrective_grade_prompt=CORRECTIVE_RAG_GRADE_PROMPT,
    )
    should_retry = not references or initial_grade.grade in {"partial", "insufficient"}
    if not should_retry:
        append_corrective_trace(
            settings=settings,
            query=query,
            source_type=source_type,
            target_name=target_name,
            initial_reference_count=len(references),
            initial_grade=initial_grade,
            triggered=False,
            follow_up_query="",
            follow_up_reference_count=0,
            final_references=references,
        )
        return references

    follow_up_query = generate_corrective_query(
        settings=settings,
        query=query,
        references=references,
        history=history,
        grade=initial_grade,
        corrective_query_prompt=CORRECTIVE_RAG_QUERY_PROMPT,
    )
    second_pass_top_k = min(
        20,
        max(top_k + 2, settings.kb.CORRECTIVE_RAG_SECOND_PASS_TOP_K),
    )
    second_pass_threshold = min(score_threshold, settings.kb.CORRECTIVE_RAG_SECOND_PASS_SCORE_THRESHOLD)
    final_limit = min(
        24,
        max(
            second_pass_top_k,
            top_k
            + (
                settings.kb.CORRECTIVE_WEB_SEARCH_TOP_K
                if settings.kb.ENABLE_CORRECTIVE_WEB_SEARCH and initial_grade.grade == "partial"
                else 0
            ),
        ),
    )
    follow_up_references: list[RetrievedReference] = []
    second_pass_error = ""
    try:
        follow_up_references = retrieve(
            follow_up_query,
            second_pass_top_k,
            second_pass_threshold,
        )
    except Exception as exc:
        second_pass_error = str(exc)

    final_references = merge_corrective_references(
        references,
        follow_up_references,
        limit=final_limit,
    )
    web_search_triggered = False
    web_search_query = ""
    web_reference_count = 0
    web_sources: list[str] = []
    web_error_message = ""
    if (
        search_web is not None
        and settings.kb.ENABLE_CORRECTIVE_WEB_SEARCH
        and initial_grade.grade == "partial"
    ):
        web_search_triggered = True
        web_search_query = follow_up_query or query.strip()
        try:
            web_references = search_web(web_search_query)
        except Exception as exc:
            web_error_message = str(exc)
        else:
            web_reference_count = len(web_references)
            web_sources = [item.source for item in web_references[:6]]
            final_references = merge_corrective_references(
                final_references,
                web_references,
                limit=final_limit,
            )

    error_messages = [message for message in (second_pass_error, web_error_message) if message]
    append_corrective_trace(
        settings=settings,
        query=query,
        source_type=source_type,
        target_name=target_name,
        initial_reference_count=len(references),
        initial_grade=initial_grade,
        triggered=True,
        follow_up_query=follow_up_query,
        follow_up_reference_count=len(follow_up_references),
        final_references=final_references,
        error_message="; ".join(error_messages),
        web_search_triggered=web_search_triggered,
        web_search_query=web_search_query,
        web_reference_count=web_reference_count,
        web_sources=web_sources,
        web_error_message=web_error_message,
    )
    return final_references


