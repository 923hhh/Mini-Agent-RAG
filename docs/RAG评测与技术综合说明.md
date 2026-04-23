# Mini-Agent-RAG：RAG 系统评测与技术说明

> 初版时间：2026-04-17 | 最近更新：2026-04-20

---

## 一、项目概述

Mini-Agent-RAG 是一个基于 FastAPI 的本地知识库检索增强生成（RAG）系统，面向高校领域文档问答场景。系统采用前后端分离架构：

- **后端**：FastAPI（app/api/），提供知识库管理、RAG 问答、Agent 对话等 REST 接口
- **前端**：Streamlit（app/ui/），提供知识库管理、RAG 对话、Agent 对话三个功能页面
- **默认模型**：LLM 使用 DeepSeek-Chat（OpenAI Compatible 接口），Embedding 使用 BGE-M3（Ollama 本地部署）

系统核心链路并非"单次向量检索 + 直接拼 Prompt"的基础 RAG，而是集成了查询改写、混合检索、双层重排、上下文扩展、多模态证据组织及生成后审校的增强型 RAG。

---

## 二、系统架构

### 2.1 整体流程

`
用户问题
   查询改写 / 多查询生成
   Dense + BM25 混合检索
   启发式重排
   模型重排
   答案句窗口 / 句级索引提示
   Small-to-Big 上下文扩展
   按文本 / OCR / 视觉分组组织证据
   LLM 生成答案
   完备性补查
   事实审校
   最终输出
`

### 2.2 模块结构

| 模块　　　 | 路径　　　　　　　　　　　　　　　　　　　　　　| 职责　　　　　　　　　　　　　　　　　　　　 |
| ------------| -------------------------------------------------| ----------------------------------------------|
| API 层　　 | app/api/　　　　　　　　　　　　　　　　　　　　| FastAPI 路由，含 chat、knowledge_base、tools |
| RAG 链路　 | app/chains/rag.py　　　　　　　　　　　　　　　 | 检索-生成-审校主链路　　　　　　　　　　　　 |
| 检索器　　 | app/retrievers/local_kb.py　　　　　　　　　　　| 混合检索、重排、候选扩展　　　　　　　　　　 |
| 查询改写　 | app/services/retrieval/query_rewrite_service.py | 多查询生成、HyDE　　　　　　　　　　　　　　 |
| 重排服务　 | app/services/retrieval/rerank_service.py　　　　| 模型重排　　　　　　　　　　　　　　　　　　 |
| 知识库构建 | app/services/kb/　　　　　　　　　　　　　　　　| 增量重建、Embedding 组装、句级索引　　　　　 |
| 文档加载　 | app/loaders/　　　　　　　　　　　　　　　　　　| PDF、Office、图片、VLM 多格式解析　　　　　　|
| 向量存储　 | app/storage/　　　　　　　　　　　　　　　　　　| FAISS 适配、BM25 索引、元数据过滤　　　　　　|
| Agent　　　| app/agents/multistep.py　　　　　　　　　　　　 | 多步工具调用 Agent　　　　　　　　　　　　　 |
| 评测　　　 | app/services/evaluation/　　　　　　　　　　　　| CRUD-RAG 评测样本构造　　　　　　　　　　　　|

### 2.3 关键配置项

配置文件 configs/kb_settings.yaml 中当前启用的核心能力：

| 配置项　　　　　　　　　　　 | 当前值 | 说明　　　　　　　　　　　　　　　　 |
| ------------------------------| --------| --------------------------------------|
| ENABLE_QUERY_REWRITE　　　　 | True　| 查询改写　　　　　　　　　　　　　　 |
| ENABLE_MULTI_QUERY_RETRIEVAL | True　| 多查询检索（最多 3 条）　　　　　　　|
| ENABLE_HYDE　　　　　　　　　| False　| 假设文档生成（代码已实现，默认关闭） |
| ENABLE_HYBRID_RETRIEVAL　　　| True　| Dense + BM25 混合检索　　　　　　　　|
| ENABLE_HEURISTIC_RERANK　　　| True　| 启发式重排　　　　　　　　　　　　　 |
| ENABLE_MODEL_RERANK　　　　　| True　| 模型重排　　　　　　　　　　　　　　 |
| ENABLE_SMALL_TO_BIG_CONTEXT　| True　| 小块检索、大块回答　　　　　　　　　 |
| ENABLE_SENTENCE_INDEX　　　　| True　| 句级索引提示　　　　　　　　　　　　 |
| ENABLE_CORRECTIVE_RAG　　　　| False　| 纠正性检索（代码已实现，默认关闭）　 |

---

## 三、核心技术说明

### 3.1 查询改写与多查询检索

用户原始问题可能过于口语化或模糊，系统首先通过 LLM 将其改写为更适合检索的形式，并生成最多 3 条不同角度的查询，以扩大召回覆盖面。改写策略为：优先多查询，多查询失败时退回单查询改写；过于简短的问题直接跳过改写。

- **代码位置**：app/services/retrieval/query_rewrite_service.py

### 3.2 混合检索与融合打分

系统同时执行两路检索：

- **Dense 检索**（向量检索）：基于 BGE-M3 Embedding，捕获语义相似性
- **Lexical 检索**（BM25）：基于关键词匹配，捕获精确词项命中

两路结果通过 RRF（Reciprocal Rank Fusion）+ 分数归一化 + 模态加权进行融合，最终形成统一排序。

- **代码位置**：app/retrievers/local_kb.py
- **关键参数**：HYBRID_DENSE_TOP_K=50，HYBRID_LEXICAL_TOP_K=50，HYBRID_RRF_K=60

### 3.3 双层重排

混合检索的候选集排序精度有限，系统采用两层精排：

1. **启发式重排**：综合融合得分、Dense 得分、Lexical 得分、词项重合率、短语完全命中、标题/来源命中、模态匹配等多特征加权打分
2. **模型重排**：将启发式排序的 Top-N 候选送入 Rerank 模型，模型分数与启发式分数再次融合

- **代码位置**：app/retrievers/local_kb.py，app/services/retrieval/rerank_service.py
- **关键参数**：RERANK_CANDIDATES_TOP_N=20

### 3.4 上下文增强

- **答案句窗口增强**：对命中的 chunk，检查其内部是否存在与查询高度匹配的答案句，优先提取答案句附近窗口作为证据
- **句级索引提示**：为每个 chunk 建立句级向量索引，检索时用句子粒度做补充召回
- **Small-to-Big**：以小块粒度检索（精确匹配），回答时将前后相邻块拼回（完整上下文），当前扩展窗口为前后各 1 块

- **代码位置**：app/retrievers/local_kb.py，app/services/kb/sentence_index_service.py

### 3.5 多模态证据组织

系统支持文本、图片（OCR + VLM 视觉描述）等多模态文档。生成阶段会先判断当前问题是否偏向图片理解，然后将证据按"文本证据 / OCR 证据 / 视觉描述证据"三组分别组织进 Prompt，避免不同来源证据被混淆。

- **代码位置**：app/chains/rag.py

### 3.6 生成后审校

系统在 LLM 生成初稿后，执行两轮自动审校：

1. **完备性审校**：检查答案是否存在漏答、并列要点缺失、子问题未覆盖等问题，对"哪些/分别/同时/原因/措施/职责"类问题尤为有效
2. **事实审校**：检查答案是否包含证据未明确支持的外推内容，如有则删除或收缩表述

审校仍基于原证据执行，不引入外部信息，属于轻量后处理。

- **代码位置**：app/chains/rag.py

---

## 四、评测方法与结果

### 4.1 评测设计

评测分为**检索侧**和**生成侧**两个独立维度，使用两套数据集：

| 数据集 | 规模 | 特点 |
|--------|------|------|
| CRUD-RAG | 100 条 | 项目内闭环构造，验证系统基础稳定性 |
| DomainRAG | 100 条 | 更接近真实开放场景，暴露系统短板 |

DomainRAG 包含 5 种任务类型：extractive_qa、conversation_qa、multi-doc_qa、	ime-sensitive_qa、structured_qa。

### 4.2 检索侧评测指标

| 指标 | 含义 |
|------|------|
| Recall@5 | 前 5 条结果中是否包含正确证据 |
| MRR | 正确证据首次出现的排名倒数均值 |
| NDCG@5 | 前 5 条结果的整体排序质量 |

### 4.3 检索侧总体结果

| 数据集 | 版本 | Recall@5 | MRR | NDCG@5 |
|--------|------|----------|-----|--------|
| CRUD 100 | 闭环基线 | 1.0000 | 1.0000 | 1.0000 |
| Domain 100 | 初始基线（04-17） | 0.8000 | 0.6262 | 0.6236 |
| Domain 100 | 当前稳定版（04-20） | 0.8900 | 0.7340 | 0.7379 |

**解读**：CRUD 满分说明基础链路稳定；Domain 三项指标较初始基线均有明显提升（MRR +0.11，NDCG +0.11），但距目标 MRR  0.75 仍差一点。

### 4.4 Domain 分任务结果（当前稳定版）

| 任务类型　　　　　| Recall@5 | MRR    | NDCG@5 |
| -------------------| ----------| --------| --------|
| extractive_qa　　 | 0.9000   | 0.5808 | 0.6589 |
| conversation_qa　 | 0.9500   | 0.8208 | 0.8531 |
| multi-doc_qa　　　| 1.0000   | 0.9750 | 0.8086 |
| time-sensitive_qa | 0.6000   | 0.3183 | 0.3874 |
| structured_qa　　 | 1.0000   | 0.9750 | 0.9815 |

**解读**：

- structured_qa 和 multi-doc_qa 表现最好，说明系统在结构化问题和多文档聚合上已较成熟
- 	ime-sensitive_qa 仍然最弱，但较初始基线（0.4000/0.2050/0.2511）已有显著提升，说明时间约束保留和日期元数据优化有效
- extractive_qa 的 MRR 偏低，是后续重点优化方向

### 4.5 生成侧评测（RAGAS）

采用缩减样本（CRUD 10 条 + Domain 10 条均衡抽样）进行趋势判断。

| 指标 | 含义 | CRUD | Domain |
|------|------|------|--------|
| Context Recall | 回答是否覆盖检索到的关键信息 | 1.0000 | 0.8333 |
| Faithfulness | 回答是否遵循给定证据 | 0.7288 | 0.5284 |
| Factual Correctness | 回答事实层面是否正确 | 0.4540 | 0.0570 |
| Answer Relevance | 回答是否对准问题 | 0.4841 | 0.2402 |

**解读**：

- Context Recall 较高说明系统能把检索到的信息带进回答
- **Factual Correctness 是当前最突出的短板**：Domain 仅为 0.0570，说明核心问题已从"能不能找到证据"转向"找到证据后答案是否真正正确"
- Faithfulness 和 Answer Relevance 偏低，说明回答中仍存在脱离证据自由发挥和偏题现象

### 4.6 Phase 0 人工评测基线

为建立更可靠的评测口径，项目构建了 Phase 0 人工标注集（50 条），并进行了 Bad Case 分桶分析：

| 分桶类型 | 数量 | 含义 |
|----------|------|------|
| passed | 2 | 检索和生成均正确 |
| chunk_noise | 12 | 命中但返回块未直接承载答案 |
| low_rank | 12 | 命中但排不到 Top1 |
| cross_passage | 18 | 答案依赖跨段信息整合 |
| missed_recall | 6 | 完全未召回到（主要为 structured_qa） |

**Phase 0 检索指标**：Recall@20 = 0.88，Recall@50 = 0.88，MRR@10 = 0.67，NDCG@10 = 0.80，Top1 accuracy = 0.58

---

## 五、典型案例分析

### 案例 1：检索成功但生成未答全（CRUD）

| 字段 | 内容 |
|------|------|
| 问题 | 国家卫健委在"启明行动"中推广了哪些核心知识？医疗机构和家长应承担哪些角色？ |
| 检索指标 | Recall@5 = 1，MRR = 1.0000 |
| 生成指标 | Context Recall = 1.0，Faithfulness = 0.76，Factual Correctness = 0.50 |

**分析**：检索完整命中，但生成阶段未将"医疗机构和家长角色"完整答出，属于典型的"找到了但没答全"。

### 案例 2：时间敏感问答失败（Domain）

| 字段 | 内容 |
|------|------|
| 问题 | 中国人民大学的校长是谁？ |
| 检索指标 | Recall@5 = 0，MRR = 0 |
| 生成指标 | 全部为 0 |

**分析**：Top5 内完全没有命中标准证据，问题根源在检索阶段时间敏感类问题的排序仍是结构性短板。

### 案例 3：多文档整合不足（Domain）

| 字段 | 内容 |
|------|------|
| 问题 | 数学与应用数学专业与数据计算及应用专业在人才培养的共同目标和独特特点是什么？ |
| 检索指标 | Recall@5 = 1，MRR = 1.0 |
| 生成指标 | Factual Correctness = 0.11 |

**分析**：检索已覆盖两个专业的证据，但生成阶段未能将两类信息有效整合，当前短板已从"找不到"转为"证据利用不充分"。

---

## 六、当前结论与优化方向

### 6.1 总体判断

当前系统已从"基础 RAG 能否跑通"的阶段，进入"针对不同任务类型做精细化优化"的阶段。系统在 CRUD 闭环场景下表现稳定，在 Domain 开放场景下检索排序和生成事实正确性仍有明显优化空间。

### 6.2 优化优先级

| 优先级 | 方向 | 目标 |
|--------|------|------|
| 1 | 时间敏感检索优化 | 提升 time-sensitive_qa 的 MRR 和 NDCG |
| 2 | 多文档证据覆盖优化 | 提升 multi-doc_qa 排序与答案完整性 |
| 3 | 生成侧事实正确性 | 提升 Factual Correctness 与 Answer Relevance |
| 4 | Chunk 与元数据设计 | 保留标题/章节/时间到元数据，过渡到语义切分 |
| 5 | 扩大评测闭环 | 在 Phase 0 人工集上形成稳定的优化-验证循环 |

### 6.3 推荐推进顺序

| 阶段　　 | 内容　　　　　　　　　 | 主要目标　　　　　　　 |
| ----------| ------------------------| ------------------------|
| 第一阶段 | 时间敏感检索优化　　　 | 先解决当前最弱任务　　 |
| 第二阶段 | 多文档证据覆盖优化　　 | 提升排序和答案完整性　 |
| 第三阶段 | 生成侧事实正确性优化　 | 强化证据约束，减少外推 |
| 第四阶段 | Chunk 与 Metadata 优化 | 为更大规模评测打基础　 |
| 第五阶段 | 扩大评测样本与回归流程 | 形成稳定优化闭环　　　 |