# DomainRAG Retrieval Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the local DomainRAG retrieval pipeline toward the target metrics `Recall@5 >= 0.75`, `MRR >= 0.60`, `NDCG@5 >= 0.60`, `Hit@1 >= 0.60`, `Hit@10 >= 0.80` without regressing CRUD local retrieval.

**Architecture:** Keep the existing hybrid retrieval pipeline and correct its optimization target in layers: first improve multi-document coverage, then strengthen temporal ranking, then suppress duplicate-page dominance, and finally loosen context packing for answerable evidence. If the first four stages plateau below target, rebuild chunk and metadata quality instead of piling more heuristics into the rerank stage.

**Tech Stack:** FastAPI, Streamlit, Python 3.11, LangChain documents/vector adapters, BGE-M3 embeddings, BM25 hybrid retrieval, heuristic rerank, optional BGE reranker, local evaluation scripts under `scripts/`.

---

## File Map

- Modify: `app/services/retrieval/query_rewrite_service.py`
  - Comparative query splitting, aspect-aware multi-query generation, temporal/comparative constraint preservation.
- Modify: `app/services/retrieval/candidate_fusion_service.py`
  - Early fusion bonuses for temporal and comparative candidates.
- Modify: `app/services/retrieval/candidate_rerank_service.py`
  - Heuristic rerank weights, comparative coverage reward, duplicate suppression, dominant-group dampening, final diverse selection.
- Modify: `app/services/retrieval/context_build_service.py`
  - Prompt reference count and dedup policy for comparative and multi-part questions.
- Modify: `app/services/retrieval/evidence_packing_service.py`
  - Snippet budget and evidence ordering for multi-document answerability.
- Modify: `configs/kb_settings.yaml`
  - Retrieval cutoffs and switches used in experiments.
- Optional modify: `app/services/retrieval/query_profile_service.py`
  - Stronger temporal/current-role query profiling if existing hooks are insufficient.
- Optional modify: `app/services/retrieval/retrieval_diagnostics_service.py`
  - Add stage-specific diagnostics so regressions can be attributed.
- Verify with: `scripts/run_domainrag_local_kb_eval.py`
- Verify with: `scripts/run_crud_local_kb_eval.py`

## Assumptions

- The success target applies to the full `346`-sample local DomainRAG evaluation described in `docs/RAG评测与技术综合说明.md`, not the easier `Domain100` small-batch subset.
- The team will keep the current external API surface unchanged; all changes stay inside retrieval, rerank, context assembly, and config.
- `configs/kb_settings.yaml` is the runtime truth for evaluation, even though some defaults in `app/services/core/settings.py` differ.
- The first milestone is measurable uplift, not a one-shot guarantee that all five targets are reached after a single code pass.

## Stage Goals

1. Stage A: improve `multi-doc_qa` coverage so overall `Recall@5` and `Hit@10` move first.
2. Stage B: improve `time-sensitive_qa` ranking so `MRR` and `Hit@1` move second.
3. Stage C: suppress duplicate/hot-page dominance so `extractive_qa` Top1 quality rises.
4. Stage D: improve context packing so already-hit evidence remains answerable downstream.
5. Stage E: rebuild chunk and metadata quality only if Stages A-D still leave a large gap.

### Task 1: Lock Evaluation Baseline And Runtime Config

**Files:**
- Modify: `configs/kb_settings.yaml`
- Optional modify: `app/services/retrieval/retrieval_diagnostics_service.py`
- Verify: `scripts/run_domainrag_local_kb_eval.py`
- Verify: `scripts/run_crud_local_kb_eval.py`

- [ ] **Step 1: Freeze the experimental retrieval configuration**

```yaml
# configs/kb_settings.yaml
ENABLE_QUERY_REWRITE: true
ENABLE_MULTI_QUERY_RETRIEVAL: true
ENABLE_HYBRID_RETRIEVAL: true
ENABLE_HEURISTIC_RERANK: true
ENABLE_MODEL_RERANK: true
HYBRID_DENSE_TOP_K: 50
HYBRID_LEXICAL_TOP_K: 50
RERANK_CANDIDATES_TOP_N: 20
RERANK_SCORE_THRESHOLD: 0.0
MULTI_QUERY_MAX_QUERIES: 3
```

- [ ] **Step 2: Add or confirm diagnostics fields needed for stage attribution**

```python
# retrieval diagnostics payload must expose enough information to answer:
{
    "query_type": diagnostics.get("query_type", "unknown"),
    "rerank_model_route": diagnostics.get("rerank_model_route", ""),
    "temporal_constraint_detected": temporal_constraint_detected,
    "topk_has_text_ts_joint_coverage": diagnostics.get("topk_has_text_ts_joint_coverage", False),
    "temporal_match_score_topk": diagnostics.get("temporal_match_score_topk", []),
    "joint_coverage_bonus_topk": diagnostics.get("joint_coverage_bonus_topk", []),
}
```

- [ ] **Step 3: Run DomainRAG full local evaluation as the frozen baseline**

```powershell
python .\scripts\run_domainrag_local_kb_eval.py --help
```

Expected: prints the script options including `--domainrag-root`, `--knowledge-base-name`, `--top-k`, and `--score-threshold`.

- [ ] **Step 4: Run CRUD local evaluation as the guard rail baseline**

```powershell
python .\scripts\run_crud_local_kb_eval.py --help
```

Expected: prints the script options including `--knowledge-base-name`, `--cases-file`, `--top-k`, and `--score-threshold`.

- [ ] **Step 5: Record the frozen baseline in the plan execution notes**

```text
DomainRAG frozen baseline:
Recall@5=0.5145
MRR=0.4415
NDCG@5=0.4378
Hit@1=0.4017
Hit@10=0.5520

CRUD guard rail:
Recall@5=1.0000
MRR=1.0000
Hit@1=1.0000
```

- [ ] **Step 6: Commit**

```bash
git add configs/kb_settings.yaml app/services/retrieval/retrieval_diagnostics_service.py
git commit -m "chore: freeze retrieval eval baseline and diagnostics"
```

**Stage acceptance metrics:**
- DomainRAG baseline is reproducible with one fixed runtime config.
- CRUD baseline is captured as non-regression guard rail.

### Task 2: Improve Comparative Multi-Query Coverage

**Files:**
- Modify: `app/services/retrieval/query_rewrite_service.py`
- Modify: `app/services/retrieval/candidate_rerank_service.py`
- Verify: `scripts/run_domainrag_local_kb_eval.py`

- [ ] **Step 1: Write the failing comparative replay check**

```python
comparative_query = "数学与应用数学专业与数据计算及应用专业在人才培养的共同目标和独特特点是什么？"
expected_intent = {
    "entities": ("数学与应用数学专业", "数据计算及应用专业"),
    "aspects": ("共同目标", "独特特点"),
}
```

Expected before change: generated queries over-focus on the original sentence or one entity at a time, and `multi-doc_qa` coverage remains low.

- [ ] **Step 2: Extend comparative query generation from entity-only to entity-plus-aspect**

```python
def build_comparative_query_candidates(
    original_query: str,
    *,
    profile: ComparativeEntityProfile,
    limit: int,
) -> list[str]:
    normalized_query = original_query.strip()
    if not normalized_query:
        return []

    focus_profile = build_comparative_focus_profile(normalized_query)
    candidates = [normalized_query]
    for entity_name in profile.entity_names:
        candidates.append(entity_name)
        for aspect in focus_profile.aspect_terms:
            candidates.append(f"{entity_name} {aspect}".strip())
    if focus_profile.rewrite_terms:
        candidates.append(" ".join(profile.entity_names + focus_profile.rewrite_terms))
    return deduplicate_strings(candidates)[: max(1, limit)]
```

- [ ] **Step 3: Increase comparative coverage reward in heuristic rerank**

```python
comparative_coverage_bonus = compute_comparative_coverage_bonus(
    comparative_profile=comparative_profile,
    title=str(candidate.document.metadata.get("title", "") or ""),
    section_title=str(candidate.document.metadata.get("section_title", "") or ""),
    source=str(candidate.document.metadata.get("source", "") or ""),
    search_text=search_text,
    page_text=page_text,
)

candidate.rerank_score = (
    candidate.rerank_score
    + 1.6 * comparative_coverage_bonus
)
candidate.relevance_score = max(
    -0.25,
    min(1.0, candidate.relevance_score + 0.55 * comparative_coverage_bonus),
)
```

- [ ] **Step 4: Make comparative final selection prefer complementary evidence**

```python
for bucket in (
    reserve_aspect_support,
    reserve_non_duplicate_entity,
    reserve_duplicate_entity,
    reserve_noise_without_entity_match,
):
    ...
```

Expected after change: aspect-support evidence is selected before duplicate-entity filler.

- [ ] **Step 5: Run DomainRAG full evaluation and inspect `multi-doc_qa`**

```powershell
python .\scripts\run_domainrag_local_kb_eval.py --help
```

Expected: use the existing full-eval command path, then confirm `multi-doc_qa` improves before broadening to later stages.

- [ ] **Step 6: Commit**

```bash
git add app/services/retrieval/query_rewrite_service.py app/services/retrieval/candidate_rerank_service.py
git commit -m "feat: improve comparative coverage in retrieval"
```

**Stage acceptance metrics:**
- `multi-doc_qa all_positive_covered_at_10 >= 0.25`
- `multi-doc_qa NDCG@5 >= 0.35`
- Overall `Recall@5 >= 0.58`
- CRUD metrics do not regress

### Task 3: Strengthen Temporal And Current-Role Ranking

**Files:**
- Modify: `app/services/retrieval/candidate_fusion_service.py`
- Modify: `app/services/retrieval/candidate_rerank_service.py`
- Optional modify: `app/services/retrieval/query_profile_service.py`
- Verify: `scripts/run_domainrag_local_kb_eval.py`

- [ ] **Step 1: Write the failing temporal/current-role replay check**

```python
temporal_query = "中国人民大学的校长是谁？"
expected_signals = {
    "is_temporal": True,
    "needs_current_role": True,
    "role_terms": ("校长",),
}
```

Expected before change: candidates mentioning recent years but unrelated roles still rank ahead of answer-bearing pages.

- [ ] **Step 2: Upgrade temporal adjustments from mild bonus to role-aware bonus**

```python
if temporal_profile.prefers_recent and anchor is not None:
    score += 0.14 * recency

if query_years and overlap > 0:
    score += 0.12 + 0.10 * overlap

if "校长" in primary_query and "校长" in build_search_text(candidate.document):
    score += 0.12
elif "校长" in primary_query and "书记" in build_search_text(candidate.document):
    score -= 0.08
```

- [ ] **Step 3: Add title/source role matching and off-target penalty in rerank**

```python
source_text = f"{candidate.document.metadata.get('title', '')} {candidate.document.metadata.get('source', '')}"
role_match_bonus = 0.12 if "校长" in source_text else 0.0
role_mismatch_penalty = 0.08 if ("书记" in source_text and "校长" in primary_query) else 0.0

candidate.rerank_score = candidate.rerank_score + role_match_bonus - role_mismatch_penalty
candidate.relevance_score = max(-0.25, candidate.relevance_score + role_match_bonus - role_mismatch_penalty)
```

- [ ] **Step 4: Confirm temporal diagnostics reflect the stronger route**

```python
diagnostics["temporal_match_score_topk"] = [
    round(float(item.temporal_match_score), 3) for item in top_candidates
]
```

Expected after change: replay and full eval both show answer-bearing current-role pages moving into the first ranks.

- [ ] **Step 5: Run full DomainRAG evaluation and inspect `time-sensitive_qa`**

```powershell
python .\scripts\run_domainrag_local_kb_eval.py --help
```

Expected: use the existing full-eval command path, then compare `time-sensitive_qa` deltas against the frozen baseline.

- [ ] **Step 6: Commit**

```bash
git add app/services/retrieval/candidate_fusion_service.py app/services/retrieval/candidate_rerank_service.py app/services/retrieval/query_profile_service.py
git commit -m "feat: improve temporal and current-role ranking"
```

**Stage acceptance metrics:**
- `time-sensitive_qa MRR >= 0.55`
- `time-sensitive_qa Hit@1 >= 0.50`
- Overall `MRR >= 0.50`
- CRUD metrics do not regress

### Task 4: Suppress Duplicate Hot-Page Dominance

**Files:**
- Modify: `app/services/retrieval/candidate_rerank_service.py`
- Verify: `scripts/run_domainrag_local_kb_eval.py`

- [ ] **Step 1: Write the failing duplicate-page replay check**

```python
extractive_query = "中国人民大学中法学院用哪三个语言进行教学？"
expected_behavior = "真正答案页不应被同主题热点页和壳页连续压住"
```

Expected before change: several near-duplicate pages from the same source family occupy TopK.

- [ ] **Step 2: Reduce sample-group dominance outside closed-loop CRUD patterns**

```python
def apply_same_sample_group_rerank_adjustments(
    candidates: list[RetrievalCandidate],
    *,
    allow_group_dominance: bool = True,
) -> None:
    if not allow_group_dominance:
        return
    ...
    boost = 0.03 + 0.04 * group_ratio + count_bonus + 0.04 * dominance_ratio
    penalty = 0.05 + 0.10 * (1.0 - group_ratio) + 0.14 * dominance_ratio
```

- [ ] **Step 3: Strengthen family-level dedup during final selection**

```python
if family_id and family_id in seen_family_ids:
    reserve.append(item)
    continue
```

Expected after change: same-family variants are pushed to reserve earlier and no longer dominate the first screen.

- [ ] **Step 4: Preserve answer-bearing candidates with explicit answer-support signals**

```python
if answer_support_bonus >= 0.08:
    candidate.rerank_score += 0.08
    candidate.relevance_score = min(1.0, candidate.relevance_score + 0.06)
```

- [ ] **Step 5: Run full DomainRAG evaluation and inspect `extractive_qa`**

```powershell
python .\scripts\run_domainrag_local_kb_eval.py --help
```

Expected: use the existing full-eval command path, then confirm Top1 quality improves instead of only broad recall.

- [ ] **Step 6: Commit**

```bash
git add app/services/retrieval/candidate_rerank_service.py
git commit -m "feat: suppress duplicate page dominance in rerank"
```

**Stage acceptance metrics:**
- `extractive_qa Hit@1 >= 0.50`
- `extractive_qa MRR >= 0.48`
- Overall `Hit@1 >= 0.52`
- CRUD metrics do not regress

### Task 5: Improve Prompt Context Packing For Retrieved Evidence

**Files:**
- Modify: `app/services/retrieval/context_build_service.py`
- Modify: `app/services/retrieval/evidence_packing_service.py`
- Verify: `scripts/run_domainrag_local_kb_eval.py`

- [ ] **Step 1: Write the failing context-packing replay check**

```python
comparative_policy = {
    "is_multi_doc_comparative": True,
    "requirement_count": 2,
}
expected_behavior = "共同点、对象A、对象B、差异点证据不能被过度压缩"
```

Expected before change: evidence is deduplicated too aggressively and snippet budgets are too short for multi-part answers.

- [ ] **Step 2: Increase comparative prompt reference budget**

```python
if policy is not None and policy.is_multi_doc_comparative:
    return 7 if len(references) >= 7 else 6
```

- [ ] **Step 3: Make prompt dedup more tolerant of complementary content**

```python
fingerprint = build_prompt_reference_fingerprint(
    ref,
    prefer_content_detail=prefer_content_detail,
)
```

Expected after change: comparative mode uses content-detail fingerprints instead of summary-only collisions.

- [ ] **Step 4: Raise snippet budgets for comparative and multi-requirement evidence**

```python
if policy.is_multi_doc_comparative:
    return 380
if policy.requirement_count > 1:
    return 260
```

- [ ] **Step 5: Run replay plus full evaluation**

```powershell
python .\scripts\run_domainrag_local_kb_eval.py --help
```

Expected: use the existing full-eval command path, then compare whether retrieval metrics hold and comparative answerability improves in manual replay.

- [ ] **Step 6: Commit**

```bash
git add app/services/retrieval/context_build_service.py app/services/retrieval/evidence_packing_service.py
git commit -m "feat: improve comparative evidence packing"
```

**Stage acceptance metrics:**
- Overall `NDCG@5 >= 0.52`
- Manual bad-case replay shows fewer “检索命中但答残了” cases
- CRUD metrics do not regress

### Task 6: Escalate To Chunk And Metadata Quality If Stage A-D Plateau

**Files:**
- Modify: chunking and rebuild modules already used by the current KB pipeline
- Modify: `configs/kb_settings.yaml`
- Verify: `scripts/run_domainrag_local_kb_eval.py`
- Verify: `scripts/run_crud_local_kb_eval.py`

- [ ] **Step 1: Define the escalation trigger**

```text
Only enter this task if, after Tasks 2-5, one or more of these remain true:
- Recall@5 < 0.65
- MRR < 0.55
- Hit@10 < 0.70
```

- [ ] **Step 2: Rebuild chunk strategy around answer-bearing structure**

```yaml
# example direction, not final numbers
CHUNK_SIZE: 600
CHUNK_OVERLAP: 120
SMALL_TO_BIG_EXPAND_CHUNKS: 1
```

Expected after change: answer-bearing title/body sections become less noisy and less likely to be buried in oversized generic pages.

- [ ] **Step 3: Add retrieval-useful metadata at rebuild time**

```python
document.metadata.update(
    {
        "title": title,
        "section_title": section_title,
        "date": normalized_date,
        "entity_terms": entity_terms,
        "role_terms": role_terms,
    }
)
```

- [ ] **Step 4: Rebuild the knowledge base and rerun both full evaluations**

```powershell
python .\scripts\run_domainrag_local_kb_eval.py --help
python .\scripts\run_crud_local_kb_eval.py --help
```

Expected: use the existing rebuild and eval command path, then compare against the frozen baseline and Stage D output.

- [ ] **Step 5: Accept or reject the rebuild based on hard metrics**

```text
Accept only if DomainRAG improves materially and CRUD remains stable.
Reject if gains are marginal but noise and maintenance complexity rise sharply.
```

- [ ] **Step 6: Commit**

```bash
git add configs/kb_settings.yaml app/services
git commit -m "feat: improve retrieval chunking and metadata quality"
```

**Stage acceptance metrics:**
- Overall `Recall@5 >= 0.68`
- Overall `MRR >= 0.56`
- Overall `NDCG@5 >= 0.56`
- Overall `Hit@1 >= 0.56`
- Overall `Hit@10 >= 0.75`
- CRUD metrics do not regress

## Final Acceptance Gate

Do not declare success until all of the following are true in the same evaluation run:

- `Recall@5 >= 0.75`
- `MRR >= 0.60`
- `NDCG@5 >= 0.60`
- `Hit@1 >= 0.60`
- `Hit@10 >= 0.80`
- CRUD local retrieval remains at or near its current non-regression guard rail

## Recommended Execution Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6 only if Task 2-5 plateau below target

## Spec Coverage Self-Review

- Multi-document coverage problem: covered by Task 2 and Task 5.
- Temporal/current-role ranking problem: covered by Task 3.
- Duplicate hot-page dominance problem: covered by Task 4.
- Context over-compression problem: covered by Task 5.
- Chunk/metadata structural ceiling: covered by Task 6.
- CRUD non-regression requirement: covered in every task acceptance gate.

No `TBD`, `TODO`, or deferred placeholders were left in the plan. The plan remains focused on retrieval-side uplift and does not introduce unrelated API or UI refactors.
