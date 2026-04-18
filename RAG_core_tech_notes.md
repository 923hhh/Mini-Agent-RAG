# 当前 RAG 核心技术与代码说明

这份文档围绕当前项目里真正启用的 RAG 主链路整理，重点展示三件事：

- 当前系统到底用了哪些 RAG 技术
- 这些技术在代码里分别在哪里
- 每一部分最关键的代码片段是什么

## 说明

- 本文优先写当前主链路里真正启用、真正参与评测的部分。
- 对于“代码里存在，但当前默认未启用”的路径，也会明确标注，避免把“有这段代码”误解成“当前就在运行”。
- 每个技术点都分成三部分：
  - 通俗解释
  - 代码位置
  - 核心代码片段
- 代码片段不追求“整段全贴”，而是只保留最能体现逻辑的核心部分。
- 如果老师时间有限，优先看“先说结论”“配置层面”“当前代码层面的真实结论”这三节即可。

## 一、先说结论

当前系统不是“单次向量检索 + 直接拼 prompt”的基础版 RAG，而是下面这套增强链路：

1. 查询改写与多查询检索
2. 向量检索 + BM25 混合检索
3. 启发式重排 + 模型重排
4. `small-to-big` 上下文扩展
5. 文本 / OCR / 视觉证据分组组织
6. 生成后完备性补查

可以把当前主链路概括成：

`用户问题`
→ `查询改写 / 多查询`
→ `dense + BM25 混合检索`
→ `启发式重排`
→ `模型重排`
→ `small-to-big 扩展上下文`
→ `按文本 / OCR / 视觉组织证据`
→ `生成答案`
→ `完备性补查`
→ `最终输出`

## 二、配置层面：当前哪些开了

配置文件位置：`configs/kb_settings.yaml`

当前最关键的几项配置如下：

```yaml
ENABLE_QUERY_REWRITE: true
ENABLE_MULTI_QUERY_RETRIEVAL: true
ENABLE_HYDE: false
ENABLE_HYBRID_RETRIEVAL: true
ENABLE_CORRECTIVE_RAG: false
ENABLE_CORRECTIVE_WEB_SEARCH: false
ENABLE_HEURISTIC_RERANK: true
ENABLE_MODEL_RERANK: true
ENABLE_SMALL_TO_BIG_CONTEXT: true

MULTI_QUERY_MAX_QUERIES: 3
HYBRID_DENSE_TOP_K: 30
HYBRID_LEXICAL_TOP_K: 30
HYBRID_RERANK_TOP_K: 5
SMALL_TO_BIG_EXPAND_CHUNKS: 1
```

这段配置说明当前系统的真实状态是：

- 查询改写开着
- 多查询检索开着
- 混合检索开着
- 双层重排开着
- `small-to-big` 开着
- `HyDE` 和 `Corrective RAG` 代码里虽然有，但当前默认没开

因此，这里要特别区分两件事：

- “代码里有这条路径”
- “这条路径当前默认真的在运行”

比如 `HyDE`、`Corrective RAG`、`Corrective Web Search` 都属于“代码里有，但本轮默认没开”的能力。

## 三、核心技术与代码片段

### 1. 查询改写与多查询检索

通俗解释：

用户问题有时太口语、太短、太模糊，直接检索不一定好。系统会先判断是否值得改写；如果值得，就优先生成多条检索 query；如果多查询失败，再退回单查询改写。

代码位置：

- `app/services/query_rewrite_service.py:86-118`

核心代码片段：

```python
def generate_multi_queries(
    settings: AppSettings,
    query: str,
    history: list[ChatMessage] | None = None,
) -> list[str]:
    normalized_query = query.strip()
    if not normalized_query:
        return []
    if not settings.kb.ENABLE_QUERY_REWRITE or _should_skip_rewrite(normalized_query):
        return [normalized_query]

    max_queries = settings.kb.MULTI_QUERY_MAX_QUERIES
    if max_queries <= 1:
        return [normalized_query]

    if settings.kb.ENABLE_MULTI_QUERY_RETRIEVAL:
        generated_queries = _invoke_multi_query_rewrite(
            settings=settings,
            query=normalized_query,
            history=history,
            max_queries=max_queries,
        )
        if generated_queries:
            return deduplicate_query_candidates(
                normalized_query,
                generated_queries,
                limit=max_queries,
            )

    rewritten = _invoke_single_query_rewrite(settings, normalized_query, history)
    if not rewritten:
        return [normalized_query]
    return deduplicate_query_candidates(normalized_query, [rewritten], limit=max_queries)
```

这段代码最核心的意思是：

- 太短、太简单的问题直接跳过改写
- 默认最多生成 `3` 条 query
- 优先走多查询
- 多查询失败才退回单查询

也就是说，当前不是“先问什么就搜什么”，而是“先把问题改成更适合检索的形式再搜”。

### 2. 检索主入口：从 query 到 reference 的总流程

通俗解释：

这一层是整个 RAG 检索主链路的总调度器。它把“改写后的 query”“混合检索”“重排”“上下文扩展”全部串起来。

代码位置：

- `app/retrievers/local_kb.py:174-219`
- `app/retrievers/local_kb.py:233-306`

核心代码片段：

```python
query_candidates = generate_multi_queries(settings, query, history)
rewritten_query = query_candidates[1] if len(query_candidates) > 1 else query.strip()
query_bundle = build_query_bundle(query_candidates)
dense_query_bundle = build_dense_query_bundle(
    query_bundle,
    generate_hypothetical_doc(settings, query, history),
)

candidates = retrieve_candidates(
    settings=settings,
    vector_store=vector_store,
    all_documents=filtered_documents,
    query_bundle=query_bundle,
    dense_query_bundle=dense_query_bundle,
    bm25_index=bm25_index,
    top_k=top_k,
    metadata_filters=metadata_filters,
    query_profile=query_profile,
    diagnostics=retrieval_diagnostics,
)

reranked = rerank_candidates(
    settings=settings,
    query=query,
    candidates=candidates,
    query_bundle=query_bundle,
    query_profile=query_profile,
    top_k=top_k,
)
filtered = [item for item in reranked if item.relevance_score >= score_threshold]

grouped_documents = group_documents_by_doc_id(filtered_documents)
final_candidates = diversify_candidates(
    filtered,
    target_count=top_k,
    query_profile=query_profile,
)
references = [
    candidate_to_reference(
        settings=settings,
        candidate=item,
        grouped_documents=grouped_documents,
    )
    for item in final_candidates
]
```

这段代码可以直接读成一句话：

“先生成 query bundle，再做候选召回，再重排，再过滤，再扩展上下文，最后转成给生成模型使用的 `references`。”

这里有一个容易误解的点需要说明：

- 代码里保留了 `generate_hypothetical_doc(...)` 这条 `HyDE` 路径
- 但由于当前配置中 `ENABLE_HYDE = false`
- 所以本轮默认运行时，这一项不会真正向检索链路额外加入假设文档

也就是说，当前默认运行的是“查询改写 + 多查询 + 混合检索”，不是“查询改写 + HyDE + 混合检索”。

### 3. 混合检索：向量检索 + BM25 + 融合打分

通俗解释：

- 向量检索负责找“意思接近”的内容
- BM25 负责找“关键词明确命中”的内容
- 最后再把两路结果融合成统一分数

代码位置：

- `app/retrievers/local_kb.py:310-418`

核心代码片段 1：两路召回

```python
candidate_map: dict[str, RetrievalCandidate] = {}
dense_limit = max(top_k, settings.kb.HYBRID_DENSE_TOP_K)
dense_queries = dense_query_bundle or query_bundle

if modality_grouped_dense_used:
    for source_modality in select_modalities_for_query(modality_groups, query_profile):
        collect_dense_candidates(
            vector_store=vector_store,
            query_bundle=dense_queries,
            dense_limit=per_modality_dense_limit,
            candidate_map=candidate_map,
            metadata_filters=merge_metadata_filters_with_source_modality(
                metadata_filters,
                source_modality,
            ),
            dense_fetch_multiplier=settings.kb.METADATA_FILTER_DENSE_FETCH_MULTIPLIER,
        )
else:
    collect_dense_candidates(
        vector_store=vector_store,
        query_bundle=dense_queries,
        dense_limit=dense_limit,
        candidate_map=candidate_map,
        metadata_filters=metadata_filters,
        dense_fetch_multiplier=settings.kb.METADATA_FILTER_DENSE_FETCH_MULTIPLIER,
    )

if settings.kb.ENABLE_HYBRID_RETRIEVAL:
    collect_lexical_candidates(
        settings=settings,
        all_documents=all_documents,
        query_bundle=query_bundle,
        bm25_index=bm25_index,
        candidate_map=candidate_map,
        lexical_limit=settings.kb.HYBRID_LEXICAL_TOP_K,
    )
```

核心代码片段 2：融合分数

```python
max_dense = max((item.dense_relevance for item in fused), default=1.0) or 1.0
max_lexical = max((item.lexical_score for item in fused), default=1.0) or 1.0
for item in fused:
    score = 0.0
    if item.dense_rank is not None:
        score += 1.0 / (settings.kb.HYBRID_RRF_K + item.dense_rank)
    if item.lexical_rank is not None:
        score += 1.0 / (settings.kb.HYBRID_RRF_K + item.lexical_rank)
    score += settings.kb.HYBRID_DENSE_SCORE_WEIGHT * (item.dense_relevance / max_dense)
    score += settings.kb.HYBRID_LEXICAL_SCORE_WEIGHT * (item.lexical_score / max_lexical)
    score += modality_bonus_for_candidate(item.document, query_profile)
    item.fused_score = score
```

这部分说明当前系统不是“dense 和 BM25 二选一”，而是：

- 两边都召回
- 再通过 `RRF + 分数归一化 + 模态 bonus` 做融合

所以当前混合检索本质上是一个“多路打分合成”的策略。

### 4. 双层重排：启发式重排 + 模型重排

通俗解释：

混合检索拿回来的只是候选集合，顺序不一定准，所以还要继续精排。当前项目的精排分成两层：

- 第一层：启发式重排
- 第二层：模型重排

#### 4.1 启发式重排

代码位置：

- `app/retrievers/local_kb.py:598-673`

核心代码片段：

```python
overlap_ratio = (
    len(doc_terms & query_term_set) / len(query_term_set)
    if query_term_set
    else 0.0
)
phrase_bonus = 1.0 if any(query_text and query_text in search_text.lower() for query_text in plain_queries) else 0.0
normalized_bonus = (
    1.0 if any(query_text and query_text in normalized_text for query_text in normalized_queries) else 0.0
)
source_bonus = 0.4 if any(term in source_text for term in query_terms if len(term) >= 2) else 0.0

candidate.rerank_score = (
    0.35 * fused_component
    + 0.25 * dense_component
    + 0.20 * lexical_component
    + 0.10 * overlap_ratio
    + 0.06 * phrase_bonus
    + 0.10 * normalized_bonus
    + 0.04 * source_bonus
    + modality_bonus
)
```

这段代码说明启发式重排看的不只是一个分数，而是综合考虑：

- 融合得分
- dense 得分
- lexical 得分
- 词项重合
- 短语完全命中
- 标题/来源字段命中
- 模态匹配

也就是说，它不是“简单按向量分数排序”，而是一个多特征打分。

#### 4.2 模型重排

代码位置：

- `app/retrievers/local_kb.py:531-595`
- `app/services/rerank_service.py:24-78`

核心代码片段 1：把候选送入 rerank 模型

```python
top_n = max(settings.kb.RERANK_CANDIDATES_TOP_N, top_k)
rerank_inputs = [
    RerankTextInput(
        candidate_id=get_chunk_id(candidate.document),
        text=build_search_text(candidate.document),
    )
    for candidate in heuristic_ranked[:top_n]
]
rerank_outcome = rerank_texts(
    settings=settings,
    query=query.strip(),
    items=rerank_inputs,
    top_n=top_n,
)
```

核心代码片段 2：模型分数与启发式分数融合

```python
candidate.model_rerank_score = model_score
heuristic_component = candidate.rerank_score / max_heuristic
candidate.rerank_score = model_score + 0.03 * heuristic_component + 0.02 * candidate.fused_score
if model_score < settings.kb.RERANK_SCORE_THRESHOLD:
    candidate.relevance_score = min(candidate.relevance_score, model_score)
else:
    candidate.relevance_score = min(
        1.0,
        0.85 * model_score + 0.15 * candidate.relevance_score,
    )
```

核心代码片段 3：CrossEncoder 输出

```python
limit = min(len(items), max(top_n or settings.kb.RERANK_CANDIDATES_TOP_N, 1))
selected_items = items[:limit]
pairs = [(query, item.text) for item in selected_items]
raw_scores = model.predict(pairs, show_progress_bar=False)

scores = {
    item.candidate_id: normalize_rerank_score(raw_score)
    for item, raw_score in zip(selected_items, raw_scores, strict=False)
}
```

这几段放在一起看，当前重排逻辑非常明确：

- 先用规则把顺序大致排好
- 再把前 `top_n` 个候选送给 CrossEncoder 精排
- 最后再做一次轻量融合，而不是完全丢掉启发式结果

#### 4.3 同组样本优先

代码位置：

- `app/retrievers/local_kb.py:1173-1213`

核心代码片段：

```python
dominant_group = group_stats[0]
dominance_ratio = max(0.0, (dominant_score - second_score) / dominant_score)

if sample_id == dominant_group.sample_id:
    boost = 0.06 + 0.08 * group_ratio + count_bonus + 0.10 * dominance_ratio
    candidate.rerank_score += boost
    candidate.relevance_score = min(
        1.0,
        candidate.relevance_score + 0.04 + 0.06 * dominance_ratio,
    )
else:
    penalty = 0.04 + 0.08 * (1.0 - group_ratio) + 0.12 * dominance_ratio
    candidate.rerank_score -= penalty
    candidate.relevance_score = max(-0.25, candidate.relevance_score - penalty)
```

这段就是之前评测里我们提到的“同组样本优先”。在 `CRUD` 这种一个事件对应多篇新闻的任务里，这个机制会更容易把同一事件下的相关块一起顶上来。

这也意味着：

- 它非常适合当前 `CRUD` 这类项目内闭环数据
- 但不应直接把它理解成开放场景下的通用最优策略

### 5. `small-to-big`：检索时小块命中，回答时扩成大块

通俗解释：

小块更容易检索到，但只给模型一个小块，往往上下文不够。所以当前项目会在最终转 `reference` 时，把命中的 chunk 向前后扩一圈。

代码位置：

- `app/retrievers/local_kb.py:725-800`

核心代码片段：

```python
expanded_content = build_expanded_content(
    settings=settings,
    document=candidate.document,
    grouped_documents=grouped_documents,
)

def build_expanded_content(
    *,
    settings: AppSettings,
    document: Document,
    grouped_documents: dict[str, dict[int, Document]],
) -> str:
    if not settings.kb.ENABLE_SMALL_TO_BIG_CONTEXT:
        return document.page_content

    expand_chunks = settings.kb.SMALL_TO_BIG_EXPAND_CHUNKS
    for index in range(chunk_index - expand_chunks, chunk_index + expand_chunks + 1):
        item = available.get(index)
        if item is None:
            continue
        text = item.page_content.strip()
        if not text or text in seen_texts:
            continue
        pieces.append(text)

    return "\n".join(pieces) if pieces else document.page_content
```

当前配置里 `SMALL_TO_BIG_EXPAND_CHUNKS = 1`，也就是：

- 命中 1 个 chunk
- 回答时把前后各 1 个相邻块一起拼回去

所以当前项目实际上走的是“小块检索，大块回答”。

### 6. 多模态证据组织：文本 / OCR / 视觉描述分开拼

通俗解释：

当前系统不是把所有证据混成一坨文本，而是会先判断当前问题是不是偏图片理解，然后再把证据按类型分组组织。

代码位置：

- `app/chains/rag.py:318-342`
- `app/chains/rag.py:370-399`

核心代码片段 1：判断是否走图片型 prompt

```python
def should_use_image_rag_prompt(
    query: str,
    references: list[RetrievedReference],
) -> bool:
    lowered_query = query.strip().lower()
    query_hint_hit = any(hint in lowered_query for hint in IMAGE_QUERY_HINTS)

    image_evidence_count = 0
    text_evidence_count = 0
    for ref in references:
        group = resolve_reference_context_group(ref)
        if group in {"ocr", "vision"}:
            image_evidence_count += 1
        else:
            text_evidence_count += 1

    if query_hint_hit and image_evidence_count > 0:
        return True
    if image_evidence_count > 0 and text_evidence_count == 0:
        return True
    if image_evidence_count >= 2 and image_evidence_count >= text_evidence_count:
        return True
    if query_hint_hit and not references:
        return True
    return False
```

核心代码片段 2：按证据类型组织上下文

```python
grouped_blocks = {
    "text": [],
    "ocr": [],
    "vision": [],
}

for index, ref in enumerate(prompt_references, start=1):
    grouped_blocks[resolve_reference_context_group(ref)].append(
        format_reference_block(index, ref)
    )

section_order = (
    ("text", "文本证据"),
    ("ocr", "OCR 证据"),
    ("vision", "视觉描述证据"),
)
for key, title in section_order:
    blocks = grouped_blocks[key]
    if not blocks:
        continue
    sections.append(f"## {title}\n" + "\n\n".join(blocks))
```

这说明当前生成阶段已经不是“普通文本 prompt”那么简单，而是：

- 先判断当前问题更像文本问答还是图片问答
- 再把文本证据、OCR 证据、视觉描述证据分开写进 prompt

这样做的价值是，能减少“OCR 内容”和“视觉描述”被混成一个事实来源。

需要说明的是：

- 当前代码已经支持多模态证据组织
- 但本轮评测主体仍然以文本问答场景为主
- 因此这部分能力更多体现为“系统已经具备”，而不是“本轮评测的主要矛盾”

### 7. 生成与完备性补查

通俗解释：

当前项目不是检索完直接回答就结束，而是：

1. 先基于证据生成初稿
2. 再检查这个初稿有没有漏答
3. 如果漏答，就让模型基于原证据再修订一次

代码位置：

- `app/chains/rag.py:103-131`
- `app/chains/rag.py:508-548`

核心代码片段 1：先生成，再补查

```python
def generate_rag_answer(
    settings: AppSettings,
    query: str,
    references: list[RetrievedReference],
    history: list[ChatMessage],
    agent_memory_context: str = "",
) -> str:
    prompt_kind = resolve_rag_prompt_kind(query, references)
    prompt = build_rag_prompt(query, references)
    variables = build_rag_variables(query, references, history, agent_memory_context=agent_memory_context)
    llm = build_chat_model(settings, temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(variables)
    answer = maybe_refine_rag_answer(
        settings=settings,
        query=query,
        references=references,
        context=str(variables["context"]),
        coverage_requirements=str(variables["coverage_requirements"]),
        draft_answer=answer,
    )
    return answer
```

核心代码片段 2：审校后决定是否修订

```python
if not should_run_answer_completeness_review(
    query=query,
    references=references,
    coverage_requirements=coverage_requirements,
    draft_answer=draft_answer,
):
    return draft_answer

raw_output = chain.invoke(
    {
        "context": context,
        "coverage_requirements": coverage_requirements,
        "query": query.strip(),
        "draft_answer": draft_answer.strip(),
    }
)

payload = extract_json_payload(raw_output)
status = str(payload.get("status", "")).strip().lower()
revised_answer = str(payload.get("revised_answer", "")).strip()
if status != "revise" or not revised_answer:
    return draft_answer
return revised_answer
```

这两段说明当前系统已经有一个很实用的后处理逻辑：

- 如果问题本身像“哪些、分别、同时、原因、措施、职责”这类容易漏项的问题
- 就会触发一次完备性审校
- 若发现漏答，就直接返回修订版答案

所以当前项目不是“生成完就算了”，而是会再做一次“答全了没有”的检查。

同时也要说明：

- 这一步是轻量的答案审校
- 不是重新联网检索，也不是新一轮外部证据补充
- 因此它更适合解决“漏项”问题，不足以单独解决“证据本身没找对”的问题

## 四、评测脚本和主链路的对应关系

### 检索评测

脚本位置：

- `scripts/eval_retrieval.py`

对应指标：

- `Recall@k`
- `MRR`
- `NDCG@k`

主要评什么：

- 能不能找到正确文档
- 找到之后排得靠不靠前

### 生成评测

脚本位置：

- `scripts/eval_ragas.py`

对应指标：

- `llm_context_recall`
- `faithfulness`
- `factual_correctness`
- `response_relevancy`

主要评什么：

- 回答有没有覆盖证据
- 有没有脱离证据乱答
- 事实是否正确
- 是否真正答到了问题

## 五、当前代码层面的真实结论

如果只用一句更准确的话概括当前项目，可以这样说：

“当前系统已经不是基础版 RAG，而是一个带查询改写、混合检索、双层重排、small-to-big 和多模态证据组织的增强型 RAG；现阶段最主要的问题不是完全检索不到，而是 Domain 场景下排序精度和生成事实正确性还需要继续优化。”

如果换成更适合给老师汇报的说法，可以简化成：

“当前系统已经完成了从查询改写、混合检索、重排到答案补查的一整套增强型 RAG 链路；下一步重点不是继续堆功能，而是把 Domain 场景下的排序精度和事实正确性做上去。”

## 六、后续最值得继续改的地方

结合当前代码结构和现有评测结果，下一步最值得优先做的是：

1. 继续提升 `Domain` 场景下的排序质量
2. 专门优化时间敏感问题的检索
3. 强化生成阶段对证据的严格约束
4. 继续改善多文档综合问答能力

如果后面还要继续完善这份文档，一个很自然的下一步就是再补一版“每个代码片段对应的真实运行案例”，把代码、评测、案例三份材料完全对齐。 
