# Mini-Agent-RAG：RAG 系统评测与技术说明

> 初版时间：2026-04-17 | 最近更新：2026-04-23

---

## 摘要

Mini-Agent-RAG 是一个面向高校文档问答的增强型 RAG 系统，采用 FastAPI + Streamlit 架构，集成查询改写、Dense+BM25 混合检索、双层重排及生成后审校等技术。

**核心结论**：
- 检索侧：DomainRAG 数据集 MRR 从 0.626 提升至 0.734，Hybrid 检索优于单一路径
- 生成侧：Factual Correctness 在 Domain 场景仅为 0.057，是当前最突出的短板
- 后续重点：时间敏感检索优化、生成事实正确性提升

---

## 一、系统概述

### 1.1 技术架构

| 层级 | 技术栈 | 核心职责 |
|------|--------|----------|
| 后端 | FastAPI (app/api/) | 知识库管理、RAG 问答、Agent 对话 REST 接口 |
| 前端 | Streamlit (app/ui/) | 知识库管理、RAG 对话、Agent 对话交互界面 |
| 模型 | DeepSeek-Chat + BGE-M3 | LLM 生成、Embedding 编码、Rerank 精排 |

### 1.2 RAG 主链路

```
用户问题  查询改写/多查询  Dense+BM25 混合检索  启发式重排  模型重排
          句级索引/Small-to-Big  多模态证据组织  LLM 生成  完备性/事实审校  输出
```

### 1.3 核心配置

| 配置项 | 状态 | 说明 |
|--------|------|------|
| ENABLE_QUERY_REWRITE | True | 查询改写 + 多查询（最多3条） |
| ENABLE_HYBRID_RETRIEVAL | True | Dense + BM25 混合检索 |
| ENABLE_HEURISTIC_RERANK | True | 启发式重排（多特征加权） |
| ENABLE_MODEL_RERANK | True | 模型重排（bge-reranker） |
| ENABLE_SMALL_TO_BIG_CONTEXT | True | 小块检索、大块回答 |
| ENABLE_SENTENCE_INDEX | True | 句级索引提示 |

### 1.4 核心技术点

| 技术点 | 实现概要 | 代码位置 |
|--------|----------|----------|
| 查询改写 | LLM 改写 + 多查询生成（最多3条） | app/services/retrieval/query_rewrite_service.py |
| 混合检索 | Dense(BGE-M3) + BM25，RRF 融合 | app/retrievers/local_kb.py |
| 双层重排 | 启发式特征打分 + 模型重排 | app/retrievers/local_kb.py |
| 上下文增强 | 答案句窗口 + 句级索引 + Small-to-Big | app/services/kb/sentence_index_service.py |
| 生成审校 | 完备性检查 + 事实审校 | app/chains/rag.py |

---

## 二、实验设计与方法

### 2.1 数据集

| 数据集 | 规模 | 用途 | 特点 |
|--------|------|------|------|
| CRUD-RAG | 100条 | 闭环稳定性验证 | 项目内构造，知识库与问题强对齐 |
| DomainRAG | 100条 | 开放场景评测 | 真实高校问答，暴露系统短板 |

DomainRAG 包含 5 类任务（各20条）：extractive_qa、conversation_qa、multi-doc_qa、time-sensitive_qa、structured_qa。

### 2.2 评测指标

**检索侧**：Recall@5、MRR、NDCG@5

**生成侧**（RAGAS）：Context Recall、Faithfulness、Factual Correctness、Answer Relevance

### 2.3 对比实验设置

| 实验类型 | 对比内容 | 目的 |
|----------|----------|------|
| 版本对比 | 初始基线 vs 当前稳定版 | 验证整体链路优化效果 |
| 检索范式对比 | Dense only vs BM25 only vs Hybrid | 验证混合检索必要性 |
| 组件对比 | bge-reranker-base vs v2-m3 | 验证重排模型差异 |

---

## 三、实验结果与分析

### 3.1 检索性能总体对比

| 数据集 | 版本 | Recall@5 | MRR | NDCG@5 |
|--------|------|----------|-----|--------|
| CRUD 100 | 闭环基线 | 1.0000 | 1.0000 | 1.0000 |
| Domain 100 | 初始基线 | 0.8000 | 0.6262 | 0.6236 |
| Domain 100 | 当前稳定版 | **0.8900** | **0.7340** | **0.7379** |

**提升幅度**：MRR +0.1078，NDCG +0.1143，说明增强型 Hybrid 链路有效。

### 3.2 检索范式对比（Domain 100）

| 方法 | Recall@5 | MRR | NDCG@5 |
|------|----------|-----|--------|
| Dense only | 0.8500 | 0.6583 | 0.6604 |
| BM25 only | 0.8500 | 0.6905 | 0.6930 |
| **Hybrid** | **0.8900** | **0.7318** | **0.7265** |

Hybrid 在 MRR 上比 Dense 提升 +0.0735，比 BM25 提升 +0.0413，验证多路召回有效性。

### 3.3 Reranker 组件对比

| 指标 | bge-reranker-base | bge-reranker-v2-m3 | 增量 |
|------|-------------------|--------------------|------|
| Recall@5 | 0.8700 | 0.8900 | +0.0200 |
| MRR | 0.7298 | 0.7418 | +0.0120 |
| NDCG@5 | 0.7317 | 0.7482 | +0.0165 |
| Top1 Hit | 0.6600 | 0.6700 | +0.0100 |

v2-m3 在 extractive_qa 和 structured_qa 上收益更明显，系统最终采用"按 query type 路由 reranker"方案。

### 3.4 Domain 分任务结果（当前稳定版）

| 任务类型 | Recall@5 | MRR | NDCG@5 | 状态 |
|----------|----------|-----|--------|------|
| structured_qa | 1.0000 | 0.9750 | 0.9815 | 优秀 |
| multi-doc_qa | 1.0000 | 0.9750 | 0.8086 | 良好 |
| conversation_qa | 0.9500 | 0.8208 | 0.8531 | 良好 |
| extractive_qa | 0.9000 | 0.5808 | 0.6589 | 需优化 |
| **time-sensitive_qa** | **0.6000** | **0.3183** | **0.3874** | **最弱** |

time-sensitive_qa 已从初始 0.4000/0.2050/0.2511 提升至 0.6000/0.3183/0.3874，但仍是后续优先优化方向。

### 3.5 生成质量评测

| 数据集 | 样本数 | Context Recall | Faithfulness | Factual Correctness | Answer Relevance |
|--------|--------|----------------|--------------|---------------------|------------------|
| CRUD | 10 | 1.0000 | 0.7288 | 0.4540 | 0.4841 |
| Domain | 10 | 0.8333 | 0.5284 | **0.0570** | 0.2402 |

**核心发现**：Factual Correctness 在 Domain 场景仅为 0.057，是当前最突出问题。系统已能召回证据，但生成阶段证据利用不充分。

### 3.6 Phase 0 人工评测（50条）

**Bad Case 分桶**：

| 类型 | 数量 | 占比 | 含义 |
|------|------|------|------|
| cross_passage | 18 | 36% | 答案依赖跨段信息整合 |
| chunk_noise | 12 | 24% | 命中但块未直接承载答案 |
| low_rank | 12 | 24% | 命中但排不到 Top1 |
| missed_recall | 6 | 12% | 完全未召回 |
| passed | 2 | 4% | 完全正确 |

**关键指标**：Recall@20 = 0.88，MRR@10 = 0.67，Top1 accuracy = 0.58

**结论**：问题已从"召回失败"转向"块不够准、排序不够前、跨段聚合不足"。

### 3.7 典型案例分析

**案例1：检索成功但生成未答全**

| 项目 | 内容 |
|------|------|
| 问题 | 国家卫健委在"启明行动"中推广了哪些核心知识？医疗机构和家长应承担哪些角色？ |
| 检索 | Recall@5=1, MRR=1.0 |
| 生成 | Factual Correctness=0.50 |
| 分析 | 检索完整命中，但生成阶段未将"医疗机构和家长角色"完整答出，典型"找到了但没答全" |

**案例2：时间敏感问答失败**

| 项目 | 内容 |
|------|------|
| 问题 | 中国人民大学的校长是谁？ |
| 检索 | Recall@5=0 |
| 分析 | Top5 未命中标准证据，时间敏感类排序仍是结构性短板 |

**案例3：多文档整合不足**

| 项目 | 内容 |
|------|------|
| 问题 | 数学与应用数学与数据计算及应用专业的共同目标和独特特点？ |
| 检索 | Recall@5=1, MRR=1.0 |
| 生成 | Factual Correctness=0.11 |
| 分析 | 证据已覆盖两个专业，但生成阶段未能有效整合，短板从"找不到"转为"证据利用不充分" |

---

## 四、结论与优化方向

### 4.1 总体判断

系统已从"基础 RAG 能否跑通"进入"精细化优化"阶段：
- **检索侧**：CRUD 闭环稳定，Domain 开放场景排序精度有提升空间
- **生成侧**：核心矛盾从"找证据"转向"用好证据"，Factual Correctness 是最突出短板

### 4.2 优化优先级

| 优先级 | 方向 | 目标 | 关联指标 |
|--------|------|------|----------|
| 1 | 时间敏感检索优化 | 提升时间约束理解与排序 | time-sensitive_qa MRR |
| 2 | 生成事实正确性 | 强化证据约束，减少外推 | Factual Correctness |
| 3 | 多文档证据整合 | 提升跨文档信息聚合 | multi-doc_qa 完整性 |
| 4 | Chunk 与元数据优化 | 语义切分、标题/时间元数据 | chunk_noise 占比 |
| 5 | 评测闭环扩大 | Phase 0 人工集上形成稳定循环 | 整体指标稳定性 |

### 4.3 推荐推进顺序

| 阶段 | 内容 | 目标 |
|------|------|------|
| 第一阶段 | 时间敏感检索优化 | 解决当前最弱任务 |
| 第二阶段 | 生成事实正确性优化 | 强化证据约束 |
| 第三阶段 | 多文档证据整合优化 | 提升复杂问题处理能力 |
| 第四阶段 | Chunk 与 Metadata 优化 | 为更大规模评测打基础 |
| 第五阶段 | 评测样本与回归流程 | 形成稳定优化闭环 |

---

## 附录：核心代码位置速查

| 模块 | 文件路径 | 核心功能 |
|------|----------|----------|
| RAG 主链路 | app/chains/rag.py | 检索-生成-审校全流程 |
| 混合检索 | app/retrievers/local_kb.py | Dense+BM25 召回、RRF 融合、双层重排 |
| 查询改写 | app/services/retrieval/query_rewrite_service.py | 多查询生成、HyDE |
| 句级索引 | app/services/kb/sentence_index_service.py | 句级向量索引、Small-to-Big |
| 重排服务 | app/services/retrieval/rerank_service.py | 模型重排 |
| 评测构造 | app/services/evaluation/ | CRUD-RAG 评测样本构造 |