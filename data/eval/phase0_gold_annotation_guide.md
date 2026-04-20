# Phase 0 人工集标注说明

## 目标

- 这套人工集不是追求大规模，而是追求“问题、gold 文档、gold 段落、失败原因”都足够准。
- 后续所有排序优化，优先在这套人工集上做判断。

## 推荐规模

- 第一版先做 `100~200` 条。
- 建议任务分布：
  - `extractive_qa`
  - `time-sensitive_qa`
  - 少量 `multi-doc_qa`
  - 少量 `OCR / image evidence` 样本

## 字段说明

- `case_id`
  - 人工集唯一编号，建议固定前缀，如 `phase0-001`
- `source_dataset`
  - 样本来源，如 `crud_manual`、`domain_manual`
- `review_status`
  - 当前标注状态，推荐值：`seed_for_review / verified / rejected`
- `query`
  - 用户真实问题，不要改写成更标准的问句
- `history_qa`
  - 多轮问题的上文问答；单轮样本可留空
- `question_type`
  - 推荐值：`definition / parameter / procedure / diagnosis`
- `needs_image_evidence`
  - 是否必须依赖图像、OCR 或图文关系
- `answerable_from_single_chunk`
  - 是否能由单个 chunk 直接回答
- `gold_documents`
  - 能支撑答案的文档级来源
- `gold_passages`
  - 真正承载答案的段落或 chunk
- `acceptable_negatives`
  - 主题接近但不该算正例的干扰项
- `needs_cross_passage_aggregation`
  - 是否需要跨段聚合
- `failure_bucket`
  - 初始可留空，复盘 bad case 时再回填
- `notes`
  - 记录边界条件或特殊说明

## failure_bucket 统一取值

- `missed_recall`
- `low_rank`
- `chunk_noise`
- `cross_passage`
- `ocr_noise`
- `image_text_misaligned`
- `query_rewrite_drift`

## 标注原则

- `gold_documents` 要偏保守：只标真正支撑答案的文档。
- `gold_passages` 要比 `gold_documents` 更严格：必须是直接承载答案的句子、段落或 chunk。
- 如果一个问题依赖两段以上信息，`answerable_from_single_chunk=false`，并把关键段都写进 `gold_passages`。
- 如果是多轮问题，必须补 `history_qa`，不要只保留孤立的当前轮 query。
- 如果一个样本只有在“当前年份/当前版本”前提下才能成立，必须在 `notes` 里把时间锚点写清楚。

## 候选数策略

- 粗召回默认看更深：
  - dense `top-50`
  - lexical `top-50`
- 强 rerank：
  - 至少保留 `top-10`
- 生成前压缩：
  - 简单题 `top-3`
  - 并列题 / 多条件题 / 多文档题 `top-5`

## 评测协议

- 检索侧：
  - `Recall@20`
  - `Recall@50`
  - `MRR@10`
  - `NDCG@10`
  - `Top1 accuracy`
- 生成侧：
  - `faithfulness`
  - `factual_correctness`
