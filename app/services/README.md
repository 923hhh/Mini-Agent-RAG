## Service Layout

`app/services` now keeps only category directories so the structure is readable
at a glance.

- `core/`
  - shared configuration and infrastructure helpers
  - `settings.py`
  - `network.py`
  - `observability.py`
- `models/`
  - model client construction and model-adjacent helpers
  - `llm_service.py`
  - `streaming_llm.py`
  - `embedding_service.py`
  - `image_caption_service.py`
- `runtime/`
  - app lifecycle and transient runtime state
  - `init_service.py`
  - `memory_service.py`
  - `temp_kb_service.py`
- `evaluation/`
  - evaluation case and reference helpers
- `kb/`
  - knowledge-base build and rebuild services
- `retrieval/`
  - query rewrite, rerank, web correction, reference formatting

The goal of this layout is clarity, not behavior change.
