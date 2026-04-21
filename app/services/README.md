## Service Layout

This directory now uses a compatibility-first layout:

- Top-level `*_service.py` files that remain here are stable entry points.
- Files grouped under subpackages contain the actual implementation for
  evaluation, knowledge-base, and retrieval concerns.
- Thin wrappers are kept at the old paths so the rest of the codebase does not
  need a risky all-at-once import rewrite.

Current grouping:

- `evaluation/`
  - `crud_eval_cases.py`
  - `eval_reference_utils.py`
- `kb/`
  - `embedding_assembler.py`
  - `kb_incremental_rebuild.py`
  - `kb_ingestion_service.py`
  - `rebuild_task_service.py`
  - `sentence_index_service.py`
- `retrieval/`
  - `query_rewrite_service.py`
  - `reference_overview.py`
  - `rerank_service.py`
  - `web_search_service.py`

This is a structural cleanup, not a behavior rewrite.
