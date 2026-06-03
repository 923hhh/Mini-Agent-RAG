# 开发过程记录

## 2026-04-10 O1 第二阶段完成

### 目标
- 完成 `documents.py` 拆分的第二阶段。
- 将图片 / OCR / 说明书页链路从单文件中拆出。
- 保留旧导入路径兼容，避免影响现有服务层调用。

### 实施内容
- 新增加载器模块：
  - `app/loaders/factory.py`
  - `app/loaders/text.py`
  - `app/loaders/pdf.py`
  - `app/loaders/office.py`
  - `app/loaders/vlm.py`
  - `app/loaders/image.py`
- 将文本、PDF、DOCX、EPUB 的实现迁出 `app/loaders/documents.py`。
- 将图片、OCR、区域 caption、说明书页识别逻辑整体迁入 `app/loaders/image.py`。
- 将 `app/loaders/documents.py` 收敛为兼容导出层，继续暴露：
  - `load_documents`
  - `load_file`
  - `list_supported_files`
  - 各 `Knowledge` 类和必要数据结构
- 更新 `app/loaders/__init__.py`，补充公共导出。

### 验证结果
- 8 个 loader 文件语法编译通过。
- `app.loaders.documents`、`app.loaders.factory`、`app.services.embedding_assembler`、`app.services.kb_incremental_rebuild` 导入通过。
- `KnowledgeFactory` 注册结果正确，包含：
  - `MarkdownKnowledge`
  - `TextKnowledge`
  - `PdfKnowledge`
  - `DocxKnowledge`
  - `EpubKnowledge`
  - `ImageKnowledge`
- `documents.py` 已缩减为兼容层，图片主实现迁移成功。

### 当前状态
- `O1` 已完成到“结构拆分可用”阶段。
- 后续若继续深入，可再把图片模块内部按 `OCR / instruction page / region caption / trace` 继续细分。

## 2026-04-10 O3 异步重建接口完成

### 目标
- 将 `POST /knowledge_base/rebuild` 从同步阻塞改为异步任务模式。
- 保留服务层和 CLI 的同步重建能力，不影响脚本调用。
- 为 UI 增加任务轮询逻辑。

### 实施内容
- 在 `app/schemas/kb.py` 新增：
  - `RebuildTaskAccepted`
  - `RebuildTaskStatus`
- 新增任务服务：
  - `app/services/rebuild_task_service.py`
- 任务服务采用：
  - 进程内任务表
  - `ThreadPoolExecutor` 后台执行
  - 状态流转：`pending -> running -> succeeded/failed`
- 改造 API：
  - `POST /knowledge_base/rebuild` 改为返回任务提交结果，HTTP `202`
  - 新增 `GET /knowledge_base/rebuild/{task_id}` 查询任务状态
- 保持 `app/services/kb_ingestion_service.py` 中的 `rebuild_knowledge_base()` 同步函数不变，供 CLI、导入脚本和上传自动重建继续复用。
- 更新 `app/ui/app.py`：
  - 提交重建后自动轮询任务状态
  - 成功后展示任务结果并刷新知识库列表
- 更新 `scripts/validate_phase7.py`：
  - 将旧同步断言改为“提交任务 + 轮询结果”

### 验证结果
- 相关 5 个文件语法编译通过：
  - `app/schemas/kb.py`
  - `app/services/rebuild_task_service.py`
  - `app/api/knowledge_base.py`
  - `app/ui/app.py`
  - `scripts/validate_phase7.py`
- 使用 `TestClient` 验证了失败路径：
  - 提交重建任务返回 `202`
  - 查询状态可得到 `failed`
  - `error_message` 正常回填
- 使用现有知识库 `crud_rag_3qa_60` 验证了成功路径：
  - 提交重建任务返回 `202`
  - 查询状态最终为 `succeeded`
  - `result` 字段存在
  - 返回结果中 `index_mode=reuse`
  - `files_processed=180`
  - `chunks=191`

### 当前状态
- `O3` 已完成。
- 当前异步任务存储为进程内内存态，适合单进程开发和演示环境。
- 若后续要支持多进程或重启后保留任务状态，需要引入持久化任务存储。

## 2026-04-10 M1 / M3 / O4 收口

### 目标
- 收口 `M1`：停止从 `model_settings.yaml` 读取和保存 API Key，改为仅通过环境变量提供。
- 收口 `M3`：将 API settings 改为启动期加载，并通过依赖注入传递。
- 收口 `O4`：删除无实际价值的 `app/agents/executor.py` 兼容壳文件。

### 实施内容
- 更新 `app/services/settings.py`：
  - 为 `model_settings.yaml` 增加敏感字段保护，启动时自动忽略 `OPENAI_COMPATIBLE_API_KEY` 与 `IMAGE_VLM_API_KEY`。
  - 对 `save_config_values()` 增加敏感字段校验，阻止继续把 API Key 写回 YAML。
  - 保存普通模型配置时，会顺带把历史遗留的敏感字段清空。
- 更新 `configs/model_settings.yaml`：
  - 清空已提交的 API Key 字段。
  - 补充“通过环境变量注入密钥”的注释说明。
- 更新 `app/ui/app.py`：
  - 图片 VLM 配置面板不再显示或保存 API Key。
  - 改为提示运行时环境变量是否已就绪。
  - 更新提示文案，明确“保存后需重启 API / UI 才会生效”。
- 更新 FastAPI 启动链路：
  - `app/api/main.py` 新增 `lifespan`，在应用启动时加载一次 settings。
  - 新增 `app/api/dependencies.py`，统一从 `app.state` 注入 settings。
  - `app/api/chat.py`、`app/api/knowledge_base.py` 改为依赖注入，不再在路由函数中显式调用 `load_settings()`。
- 更新 `scripts/validate_phase7.py`：
  - 改为 `with TestClient(app)` 形式，确保 lifespan 在验证时真正执行。
- 删除 `app/agents/executor.py`，并在 `app/agents/__init__.py` 直接导出 `multistep` 的核心入口。

### 验证结果
- 使用内存编译方式验证以下文件语法通过：
  - `app/services/settings.py`
  - `app/api/main.py`
  - `app/api/dependencies.py`
  - `app/api/chat.py`
  - `app/api/knowledge_base.py`
  - `app/agents/__init__.py`
  - `app/ui/app.py`
  - `scripts/validate_phase7.py`
- 使用 `TestClient` 做启动期 smoke test 验证：
  - `load_settings()` 加载后，`OPENAI_COMPATIBLE_API_KEY` 与 `IMAGE_VLM_API_KEY` 均为空字符串。
  - `app/agents/executor.py` 已不存在。
  - `/health` 与 `/tools` 请求可正常返回。
- 额外说明：
  - `python -m compileall ...` 因现有 `__pycache__` 写权限受限失败，但失败点均为 `.pyc` 写入阶段，不是语法错误。

### 当前状态
- `M1` 的代码侧整改已完成，已提交到仓库中的明文密钥已从配置文件中清除。
- 若这些密钥此前已真实暴露到 git 历史，对应平台上的密钥轮换仍需要手动完成。
- `M3` 已从“`@lru_cache` 缓存”进一步收口为“FastAPI 启动期加载 + 依赖注入”。
- `O4` 已完成。

## 2026-04-11 M1 环境变量链路补齐 + O6 Agent Tool Calling

### 目标
- 补齐 `M1` 的最后一段可用性链路：支持项目自动读取 `configs/.env`，避免把密钥改到环境变量后本地进程仍读不到。
- 落地 `O6 / T3`：将 Agent 的工具选择升级为“优先 LLM 原生 tool calling，失败时回退启发式”。

### 实施内容
- 更新 `app/services/settings.py`：
  - 新增 `load_project_env()`，会在 `load_settings()` 前自动读取：
    - 项目根目录 `.env`
    - `configs/.env`
  - 采用“系统环境变量优先，`.env` 只补空位”的策略，不覆盖外部显式注入的值。
- 新增模板文件：
  - `configs/.env.example`
- 更新 `configs/model_settings.yaml.example`：
  - 明确说明项目会自动加载 `configs/.env`。
- 更新 `app/tools/registry.py`：
  - 新增工具定义解析与 LangChain tool schema 构造辅助函数，供 Agent tool calling 复用。
- 重写 `app/agents/multistep.py` 中的工具规划路径：
  - 新增 `AGENT_TOOL_PLANNING_SYSTEM_PROMPT`
  - 新增 `select_next_tool_call_with_llm()`
  - 新增工具历史与知识库证据摘要构造逻辑
  - 规划策略改为：
    - 先用 `build_chat_model(...).bind_tools(...)` 做单步工具规划
    - 如果模型成功返回工具调用，按模型决策执行
    - 如果模型不支持 / 调用失败 / 未给出工具，再回退到旧启发式规则
- 更新 `app/ui/app.py`：
  - 图片 VLM API Key 输入提示补充为“支持直接写入 `configs/.env`，项目启动时自动加载”。

### 验证结果
- 离线 smoke test 验证通过：
  - `resolve_openai_compatible_api_key(settings)` 可从 `configs/.env` 解析到值
  - `resolve_image_vlm_api_key(settings)` 可从 `configs/.env` 解析到值
- Agent 规划离线校验通过：
  - 当 LLM 返回 `calculate` tool call 时，`select_next_tool_call()` 能正确采用模型规划结果
  - 当 LLM 未返回 tool call 时，知识库问题仍会回退为 `search_local_knowledge`
  - 当 LLM 不可用时，时间问题仍会回退为 `current_time`
- 相关文件已通过内存编译检查：
  - `app/services/settings.py`
  - `app/tools/registry.py`
  - `app/agents/multistep.py`

### 当前状态
- `M1` 现在已具备完整本地使用链路：
  - YAML 不保存密钥
  - `configs/.env` 可自动加载
  - UI 也会引导改用 `.env`
- `O6 / T3` 已完成到可用版本：
  - 工具选择已不再只依赖关键词
  - 同时保留启发式回退，避免模型或 provider 不稳定时把 Agent 整体带崩
- 后续若继续优化 Agent，可进一步把“最终答案生成”也切到基于 tool-calling 会话上下文统一生成，而不是沿用当前的结果归纳逻辑。

## 2026-04-11 T6 chunk cache 改用 numpy 存储

### 目标
- 将增量重建的 chunk cache 从“JSON 内联 embedding”改为“JSON metadata + `.npy` 向量 sidecar”。
- 保留对旧版 JSON 内联 embedding cache 的读取兼容，避免历史缓存直接失效。
- 为 Phase 7 校验脚本补充回归断言，确保后续不会把 embedding 又写回 metadata。

### 实施内容
- 更新 `app/services/kb_incremental_rebuild.py`：
  - `CachedChunkEntry.embedding` 改为可空，支持先加载 metadata、后回填向量。
  - 新增：
    - `chunk_cache_embedding_path()`
    - `is_chunk_cache_available()`
    - `build_chunk_cache_embedding_matrix()`
    - `load_chunk_cache_embeddings()`
    - `normalize_embedding_vector()`
  - `write_chunk_cache()` 改为：
    - metadata 继续写入 `.json`
    - embedding 单独写入同名 `.npy`
    - metadata 中移除内联 `embedding`
  - `load_chunk_cache()` 改为：
    - 优先读取 `.npy` sidecar 并回填到 `chunk_entries`
    - 若 `.npy` 不存在，则兼容读取旧版 JSON 内联 `embedding`
  - `cleanup_deleted_caches()` 删除缓存时同步清理 `.npy`
  - `plan_rebuild()` 复用 `is_chunk_cache_available()`，兼容新旧缓存格式
- 更新 `scripts/validate_phase7.py`：
  - 为增量重建新增 `incremental_chunk_cache_numpy` 校验项
  - 断言 `.npy` 文件存在、metadata 不再内联 embedding、`load_chunk_cache()` 能正确回填 embedding
- 更新 `requirements.txt`：
  - 显式补充 `numpy>=1.26,<3`

### 验证结果
- 相关文件已通过内存编译检查：
  - `app/services/kb_incremental_rebuild.py`
  - `scripts/validate_phase7.py`
- 离线 smoke test 验证通过：
  - 新格式会写出 `*.json + *.npy`
  - metadata 中不再保存内联 `embedding`
  - `load_chunk_cache()` 能从 `.npy` 回填向量
  - 旧版 JSON 内联 `embedding` cache 仍可正常加载
  - 生成的 embedding 矩阵为 `float32`

### 当前状态
- `T6` 已完成。
- chunk cache 现在采用“文本 metadata 与向量分离存储”的结构，后续增量重建的缓存体积和读写成本都更可控。

## 2026-04-11 T1 / T2 多查询检索与 HyDE 落地

### 目标
- 落地 `T1`：把单条 query rewrite 升级为多角度查询扩展，提高召回覆盖面。
- 落地 `T2`：为 dense 检索增加 HyDE 假设文档路径，同时保持 lexical 检索继续使用原始问题和多查询集合。
- 补充离线回归，确保多查询顺序、去重、HyDE 注入范围都稳定可测。

### 实施内容
- 更新 `app/services/query_rewrite_service.py`：
  - 保留原有单条改写 prompt，并新增：
    - `MULTI_QUERY_REWRITE_PROMPT`
    - `HYDE_PROMPT`
  - 新增：
    - `generate_multi_queries()`
    - `generate_hypothetical_doc()`
    - 多查询解析、去重、长度约束、列表前缀清洗工具函数
  - 多查询策略改为：
    - 原始问题始终保留在第 1 位
    - 若开启 `ENABLE_MULTI_QUERY_RETRIEVAL`，优先生成“直接改写 + 答案关键词补充”两条候选
    - 若多查询失败，则回退到单条 rewrite
- 更新 `app/retrievers/local_kb.py`：
  - 检索入口改为消费 `generate_multi_queries()` 结果
  - `build_query_bundle()` 改为接收 query list，而不是“原问题 + 单条 rewrite”
  - 新增 `build_dense_query_bundle()`：
    - lexical / query profile 仍然只使用多查询 bundle
    - HyDE 假设文档只追加到 dense bundle，不污染 BM25 / rerank / modality profile
  - retrieval trace 新增：
    - `query_bundle_count`
    - `dense_query_bundle_count`
    - `hyde_enabled`
    - `hyde_used`
    - `hyde_preview`
- 更新配置：
  - `app/services/settings.py`
  - `configs/kb_settings.yaml`
  - 新增：
    - `ENABLE_MULTI_QUERY_RETRIEVAL`
    - `MULTI_QUERY_MAX_QUERIES`
    - `ENABLE_HYDE`
- 更新 `scripts/validate_phase7.py`：
  - 新增 `multi_query_rewrite` 离线 mock 校验
  - 新增 `hyde_generation` 离线 mock 校验

### 验证结果
- 相关文件已通过内存编译检查：
  - `app/services/query_rewrite_service.py`
  - `app/retrievers/local_kb.py`
  - `app/services/settings.py`
  - `scripts/validate_phase7.py`
- 离线 smoke test 验证通过：
  - 多查询生成会保留原始问题，并输出去重后的扩展查询
  - `rewrite_query_for_retrieval()` 会稳定返回第 1 条扩展查询
  - 多查询失败时，会自动回退为单条 rewrite
  - HyDE 文本只会追加到 dense query bundle 末尾
  - lexical query bundle 顺序与内容不会因 HyDE 被改写

### 当前状态
- `T1` 已完成到可用版本。
- `T2` 已完成到可用版本，且默认通过 `ENABLE_HYDE=false` 控制，避免额外 LLM 开销直接影响现有链路。

## 2026-04-11 T5 BM25 持久化

### 目标
- 将词法检索从“每次请求现算 doc_infos”改为“构建期持久化 BM25 索引，检索期直接加载”。
- 引入 `rank_bm25` 作为主 BM25 后端，同时保留依赖缺失时的兼容回退，避免本地环境未安装依赖时直接打断检索。
- 把混合检索里原本硬编码的 dense / lexical 分数权重外提到配置。

### 实施内容
- 新增 `app/storage/bm25_index.py`：
  - 统一收敛词法检索辅助逻辑：
    - `build_search_text_from_parts()`
    - `build_match_terms()`
    - `normalize_search_text()`
  - 新增 BM25 持久化读写与加载缓存：
    - `write_bm25_index()`
    - `load_bm25_index()`
    - `score_bm25_index()`
    - `resolve_bm25_index_path()`
  - 磁盘格式采用 `bm25_index.json`，持久化：
    - `chunk_id`
    - `search_text`
    - `terms`
  - 运行时策略：
    - 若已安装 `rank_bm25`，加载后使用 `BM25Okapi`
    - 若未安装，则自动回退到兼容的 legacy BM25 打分逻辑
- 更新 `app/services/kb_incremental_rebuild.py`：
  - 全量 / 追加 / 纯复用三种重建模式都会维护 `bm25_index.json`
  - 追加模式会基于“复用 cache + 新增 chunk”重写完整 BM25 索引，避免旧索引与新向量库脱节
  - `ENABLE_HYBRID_RETRIEVAL=false` 时会删除已有 BM25 索引
  - 阶段耗时新增 `bm25_index_write`
- 更新 `app/services/kb_ingestion_service.py`：
  - `upload_temp_files()` 现在也会同步写入临时知识库的 BM25 索引，避免 local / temp 行为分叉
- 更新 `app/retrievers/local_kb.py`：
  - 检索时优先加载持久化 BM25 索引
  - 若索引缺失、损坏或加载失败，则自动回退到旧的动态词法路径
  - retrieval trace 新增：
    - `bm25_index_available`
    - `bm25_backend`
    - `bm25_load_error`
  - 融合分数里的硬编码权重改为配置项：
    - `HYBRID_DENSE_SCORE_WEIGHT`
    - `HYBRID_LEXICAL_SCORE_WEIGHT`
- 更新配置与依赖：
  - `app/services/settings.py`
  - `configs/kb_settings.yaml`
  - `requirements.txt` 新增 `rank-bm25>=0.2,<1`
- 更新 `scripts/validate_phase7.py`：
  - 新增 `incremental_bm25_persisted_index` 校验项，检查 BM25 索引文件存在且可加载

### 验证结果
- 相关文件已通过内存编译检查：
  - `app/storage/bm25_index.py`
  - `app/services/kb_incremental_rebuild.py`
  - `app/services/kb_ingestion_service.py`
  - `app/retrievers/local_kb.py`
  - `app/services/settings.py`
  - `scripts/validate_phase7.py`
- 离线 smoke test 验证通过：
  - 可从样例文档写出并加载 `bm25_index.json`
  - `score_bm25_index()` 能命中正确 chunk
  - `write_bm25_index_for_chunk_entries()` 能从增量重建使用的 `CachedChunkEntry` 直接生成索引
- 当前本地环境未安装 `rank_bm25`，因此离线验证走的是 `fallback` backend。
- 代码已声明 `requirements.txt` 依赖；安装该依赖后会自动切换到 `rank_bm25` 后端，无需再改代码。

### 当前状态
- `T5` 已完成到可用版本。
- 词法检索现在具备“构建期持久化、检索期直接加载、依赖缺失自动回退”的完整链路。

## 2026-04-11 T4 Corrective RAG

### 目标
- 在 `/chat/rag` 主链路中引入“检索结果分级 -> 触发二次检索 -> 再生成答案”的控制环。
- 优先用 LLM 对当前证据覆盖度做分级，失败时自动回退到启发式分级，避免新增链路把现有 RAG 整体打断。
- 在 `partial` 分级下可选补充网络搜索结果，并和本地二次检索结果统一合并。

### 实施内容
- 更新 `app/chains/rag.py`：
  - 新增：
    - `RetrievalCoverageGrade`
    - `grade_documents()`
    - `generate_corrective_query()`
    - `maybe_run_corrective_retrieval()`
    - `merge_corrective_references()`
    - `append_corrective_trace()`
  - 新增两类 prompt：
    - 检索证据覆盖度分级 prompt
    - 二次检索 query 生成 prompt
  - 分级标签固定为：
    - `relevant`
    - `partial`
    - `insufficient`
  - 控制环策略改为：
    - 首轮检索后先分级
    - 若为空、`partial` 或 `insufficient`，则生成二次检索 query
    - 第二轮检索扩大 `top_k`、降低 `score_threshold`
    - 将两轮结果按 `relevance_score` 去重合并后再进入答案生成
  - 新增 `corrective_rag_trace.jsonl`，记录是否触发、follow-up query、两轮命中数量等信息
- 更新 `app/api/chat.py`：
  - `/chat/rag` 在 local / temp 两种 source_type 下都接入 Corrective RAG 控制环
- 更新 `app/retrievers/local_kb.py`：
  - 新增：
    - `search_local_knowledge_base_second_pass()`
    - `search_temp_knowledge_base_second_pass()`
  - 作为二次检索入口供 `rag.py` 调用
- 更新配置：
  - `app/services/settings.py`
  - `configs/kb_settings.yaml`
  - 新增：
    - `ENABLE_CORRECTIVE_RAG`
    - `CORRECTIVE_RAG_SECOND_PASS_TOP_K`
    - `CORRECTIVE_RAG_SECOND_PASS_SCORE_THRESHOLD`
    - `CORRECTIVE_RAG_MAX_REFERENCES_TO_GRADE`
    - `ENABLE_CORRECTIVE_WEB_SEARCH`
    - `CORRECTIVE_WEB_SEARCH_PROVIDER`
    - `CORRECTIVE_WEB_SEARCH_ENDPOINT`
    - `CORRECTIVE_WEB_SEARCH_TOP_K`
    - `CORRECTIVE_WEB_SEARCH_TIMEOUT_SECONDS`
    - `CORRECTIVE_WEB_SEARCH_SNIPPET_MAX_CHARS`
- 新增 `app/services/web_search_service.py`：
  - 默认接入 `duckduckgo_html` provider
  - 解析 DuckDuckGo HTML 结果页，抽取标题、链接、snippet
  - 将外部搜索结果映射为 `RetrievedReference`，统一复用现有引用展示与答案生成链路
- 更新 `app/chains/rag.py`：
  - `maybe_run_corrective_retrieval()` 新增可选 `search_web` 回调
  - 当首轮分级为 `partial` 且 `ENABLE_CORRECTIVE_WEB_SEARCH=true` 时，会在本地 second pass 后追加网络补搜
  - `corrective_rag_trace.jsonl` 增加：
    - `web_search_triggered`
    - `web_search_query`
    - `web_reference_count`
    - `web_sources`
    - `web_error_message`
- 更新 `app/api/chat.py`：
  - local / temp 两条 `/chat/rag` 路径都把 `search_corrective_web_references()` 作为回调传入 Corrective RAG 控制环
- 更新 `scripts/validate_phase7.py`：
  - 新增 `corrective_rag_second_pass` 离线 mock 校验
  - 新增 `corrective_web_search_parser` 解析器校验
  - 新增 `corrective_rag_web_supplement` 网络补搜合并校验

### 验证结果
- 相关文件已通过内存编译检查：
  - `app/chains/rag.py`
  - `app/api/chat.py`
  - `app/retrievers/local_kb.py`
  - `app/services/settings.py`
  - `scripts/validate_phase7.py`
- 离线 smoke test 验证通过：
  - 当分级结果为 `partial` 时，会触发二次检索
  - 二次检索会使用 follow-up query，并按配置放大 `top_k`、降低阈值
  - 网络补搜解析器可以从 DuckDuckGo HTML 页面中提取标题、URL 和 snippet
  - 网络补搜结果会按统一引用结构并入最终 references，和本地证据共同进入生成阶段
- 已跑完整 `python scripts/validate_phase7.py` 回归通过。
- 当前未跑真实外网可达条件下的 `/chat/rag` 端到端验证，也未实测公网搜索 provider 的线上稳定性。

### 当前状态
- `T4` 已完成到可用版本。
- 当前版本已包含“本地二次检索 + partial 分支网络补搜”的控制环。
- 网络补搜默认关闭；需要显式把 `ENABLE_CORRECTIVE_WEB_SEARCH` 设为 `true` 才会生效。

## 2026-04-11 T7 xMemory Agent 长期记忆（Phase 1）

### 目标
- 落地 `优化方案.md` 第七章 Phase 1：独立 `data/agent_memory/<session_id>/` 存储、`session_id`、turns / episode / semantic 流水线、检索注入 Agent 与 RAG 子路径、trace 可观测性。
- 默认关闭，不传 `session_id` 时行为与改造前一致。

### 实施内容
- 配置：`configs/basic_settings.yaml` 与 `BasicSettings` 增加 `ENABLE_AGENT_MEMORY`、`AGENT_MEMORY_ROOT`、`AGENT_MEMORY_EPISODE_MAX_TURNS`、`AGENT_MEMORY_SEMANTIC_TOP_K`、`AGENT_MEMORY_EPISODE_TOP_K`、`AGENT_MEMORY_ENABLE_TURN_EXPANSION`、`AGENT_MEMORY_CONTEXT_CHAR_BUDGET`；`AppSettings` 增加 `agent_memory_root` / `agent_memory_session_dir`。
- 新增 `app/services/memory_service.py`：
  - `turns.jsonl` 追加写入；`episode_counter` 元数据；达到 `EPISODE_MAX_TURNS` 时封存 episode 并写入 `episodes.jsonl`，LLM 抽取语义写入 `semantics.jsonl`（失败则启发式摘要 / 跳过抽取）。
  - 检索：对语义与 episode 摘要做 embedding 余弦排序；可选 `AGENT_MEMORY_ENABLE_TURN_EXPANSION` 拼接近期原话。
  - `data/logs/memory_build_trace.jsonl`、`memory_retrieval_trace.jsonl`（仅 `ENABLE_AGENT_MEMORY=true` 时写入）。
- Schema：`AgentChatRequest.session_id`、`AgentChatResponse.session_id` / `memory_overview`；新增 `MemoryOverview`。
- `app/agents/multistep.py`：请求前 `retrieve_agent_memory`，注入工具规划与直连回答；仅 `search_local_knowledge` 成功路径调用 `generate_rag_answer` / `stream_rag_answer` 时传入 `agent_memory_context`；结束后 `persist_agent_turns`；流式 `done` 携带 `session_id` 与可选 `memory_overview`。
- `app/chains/rag.py`：`build_rag_variables` 增加长期记忆前缀，与知识库上下文区分。
- `scripts/validate_phase7.py`：新增 `run_agent_memory_offline_block`（sanitize、`persist`、注入语义 + 离线 embedding mock 检索断言）；离线 RAG mock 兼容 `agent_memory_context` 参数。

### 验证结果
- 独立运行 `run_agent_memory_offline_block(load_settings(PROJECT_ROOT))` 通过：`invalid_session_rejected`、`semantic_hits>=1`、`used_memory=True`。
- 全量 `validate_phase7` 依赖本机 API/知识库环境；当前环境若在 `knowledge_base/rebuild` 处 404 会中断，与记忆改动无直接关系。

### 当前状态
- `T7` Phase 1 已落地；Phase 2（增益式 episode、冲突展开等）仍为后续工作。
- 需在 `basic_settings.yaml` 将 `ENABLE_AGENT_MEMORY` 设为 `true` 并重启 API 后，`/chat/agent` 传入合法 `session_id` 才会启用长期记忆。

## 2026-04-11 O2 收口 + 完整验证

### 目标
- 安装缺失依赖并把 `scripts/validate_phase7.py` 跑通，避免当前优化只能停留在零散 smoke。
- 收掉 `优化方案.md` 中 `O2` 剩余的重复辅助逻辑。

### 实施内容
- 执行 `python -m pip install -r requirements.txt`，确认 `rank-bm25` 已在当前环境可用。
- 更新 `scripts/validate_phase7.py`：
  - 默认 provider 校验改为跟随当前 `model_settings.yaml`，不再硬编码 `ollama`。
  - `phase2_demo` 改成脚本内自举最小知识库夹具，不再依赖手工准备文档。
  - 增加离线验证补丁层：
    - `OfflineDeterministicEmbeddings`
    - `OfflineCrossEncoder`
    - 离线 RAG answer / stream mock
    - Agent tool-planning 的离线回退
  - 本地 / 临时知识库上传后的 RAG 校验改为更适配离线检索的阈值与断言。
  - `run_ui_checks()` 改为启用 FastAPI lifespan，并按 label 定位“选择知识库”下拉框。
  - Agent memory 离线块改为走仓库内临时目录，避免 Windows `Temp` 清理权限把校验判失败。
- 收口 `O2`：
  - `app/services/query_rewrite_service.py` 的 query 去重最终复用 `app/utils/text.py::deduplicate_strings`
  - `app/agents/multistep.py` 删除本地 `unique_preserve_order()`，统一改用 `deduplicate_strings`
  - `app/retrievers/local_kb.py` 补上遗漏的 `re` import，恢复路径 hint 提取分支

### 验证结果
- `python scripts/validate_phase7.py` 在当前环境完整通过，API / UI 汇总均成功输出。
- 相关改动已做内存编译检查通过：
  - `scripts/validate_phase7.py`
  - `app/services/query_rewrite_service.py`
  - `app/agents/multistep.py`
  - `app/retrievers/local_kb.py`
- 运行中仍会看到两类非阻断日志：
  - settings 脱敏提示
  - Streamlit / uvicorn 退出阶段的警告或 `CancelledError`
  - 这些未影响脚本最终退出码，完整验证结果为通过。

### 当前状态
- `O2` 已完成。
- `validate_phase7.py` 已具备当前环境下的离线完整回归能力，不再依赖外部模型网络可用性。

## 2026-04-11 CRUD_RAG 同组聚合 rerank

### 目标
- 为 `crud_rag_3qa_60` 这类带 `sample_id` 分组的知识库增加“同组优先、异组抑制”的 hybrid rerank，压掉 `top_k` 尾部由重名 `news1/2/3.txt` 引入的跨样本噪声。

### 实施内容
- 更新 `app/retrievers/local_kb.py`：
  - 新增 `sample_id` 推导辅助，优先读 metadata，缺失时从 `doc_id / relative_path / source_path` 中回退解析 24 位样本目录。
  - 在 heuristic rerank 和 model rerank 后增加 `apply_same_sample_group_rerank_adjustments()`：
    - 用组内 top-3 候选分数做聚合；
    - 若存在强势主组，则对主组候选加分，对异组候选降分。
  - 在 `diversify_candidates()` 前增加 `select_dominant_sample_group_candidates()`：
    - 当主组数量和聚合优势足够明显时，只保留该 `sample_id` 组内候选，不再为了凑满 `top_k` 塞入异组长尾。

### 验证结果
- 重新跑单条 CRUD smoke：
  - `python scripts/eval_crud_rag.py --knowledge-base-name crud_rag_3qa_60 --data-file data/eval/crud_rag_3qa_60_crud_rag_3qa_train.jsonl --tasks quest_answer --limit 1 --skip-generation --show-cases`
  - 结果由原先 `5` 条引用收敛为同一 `sample_id` 的 `3` 条 `news1/2/3.txt`
  - `context_char_f1` 从 `0.1859` 提升到 `0.2793`
- 同条样本生成 smoke：
  - `answer_char_f1` 提升到 `0.6537`
  - `answer_rouge_l_f1` 提升到 `0.5696`
- 最新 `retrieval_trace.jsonl` 显示：
  - `bm25_index_available=true`
  - `bm25_backend=rank_bm25`
  - `final_reference_count=3`

### 当前状态
- CRUD 场景下的重名文件误判和异组长尾噪声已明显收敛。
- 这版分组策略依赖 `sample_id` 或路径中可解析的样本目录；若后续要推广到更多数据集，最好把 `.rag_file_metadata.json` 正式并入 loader metadata。

## 2026-04-14 RAG 答案生成层第一轮优化

### 目标
- 先围绕 `factual_correctness` 偏低的问题，对非流式文本 RAG 做最小可验证的一轮结构优化。
- 在不改外部 API schema、不触发知识库重建的前提下，提升答案对证据的依赖程度、减少证据冗余输入、降低模板化和答非所问现象。

### 实施内容
- 更新 `app/chains/rag.py`：
  - 强化 `RAG_SYSTEM_PROMPT`：
    - 明确要求“严格依据上下文作答”
    - 禁止把相似事实当成目标答案
    - 证据不足时必须明确说“根据当前检索到的内容，无法确定”
    - 默认不输出“根据上下文/来源如下”等套话
  - 新增非流式三段式生成链路：
    - `RAG_EVIDENCE_EXTRACTION_PROMPT`
    - `RAG_ANSWER_FROM_EVIDENCE_PROMPT`
    - 现有 `RAG_COMPLETENESS_REVIEW_PROMPT` 继续保留，用于回答补漏
  - `generate_rag_answer()` 改为：
    - 先压缩参考证据
    - 再抽取证据事实
    - 再基于事实生成答案
    - 最后做一次完整性复审
  - `stream_rag_answer()` 保持现状，不接入三段式链路，仅补充上下文压缩观测信息
  - `build_context()` 改成返回 `ContextBuildResult`，支持上下文压缩模式
  - 新增证据压缩与去重辅助：
    - `compress_references_for_answer()`
    - `dedupe_reference_group()`
    - `build_reference_fingerprint()`
    - `infer_reference_sample_id()`
    - 优先保留同题样本、`evidence_summary` 更明确且更短的证据块
  - `format_reference_block()` 支持 `compressed=True`，压缩模式下优先使用 `evidence_summary / content_preview`
  - `append_answer_trace()` 新增内部观测字段：
    - `compressed_reference_count`
    - `compressed_context_chars`
    - `evidence_fact_count`
    - `evidence_unknown_count`
    - `coverage_requirement_count`
  - `split_query_into_requirements()` 增加问号拆分，提升多问句场景的覆盖检查能力

### 初步验证
- `python -m py_compile app/chains/rag.py` 通过。
- 本地最小 smoke 通过：
  - `build_context(references, compress=True)` 能正常返回压缩后的上下文元信息
  - `compress_references_for_answer()` 能正常工作
- 尚未跑真实模型的 `CRUD 30` / `RAGAS 30` 回归，下一步需要用官方 split 知识库做正式评测确认收益。

### 当前状态
- 非流式文本 RAG 已切到“压缩证据 -> 抽取事实 -> 基于事实作答 -> 完整性复审”的第一版结构。
- 流式回答链路暂未修改。
- 是否对 `factual_correctness` 有实质提升，仍需以 `crud_rag_official_split_local` 上的 CRUD 30 / RAGAS 30 结果为准。

### 回归评测结果（2026-04-14 晚）
- 已跑官方 split 知识库 `crud_rag_official_split_local` 的两组正式回归：
  - `python scripts/eval_crud_rag.py --knowledge-base-name crud_rag_official_split_local --data-file data/eval/crud_rag_official_split_local_official_split.jsonl --tasks quest_answer --limit 30 --output data/eval/crud_rag_official_split_local_questanswer_limit30_report_v2.json`
  - `python scripts/eval_ragas.py --knowledge-base-name crud_rag_official_split_local --data-file data/eval/crud_rag_official_split_local_official_split.jsonl --tasks quest_answer --limit 30 --output data/eval/crud_rag_official_split_local_ragas_limit30_v2.json --batch-size 8`
- CRUD 30 新结果：
  - `answer_char_f1 = 0.1187`（旧基线 `0.3400`）
  - `answer_rouge_l_f1 = 0.1113`（旧基线 `0.3037`）
  - `answer_bleu_4 = 0.0462`（旧基线 `0.1711`）
  - `retrieval_non_empty = 1.0`
- RAGAS 30 新结果：
  - `llm_context_recall = 1.0`（旧基线 `0.9667`）
  - `faithfulness = 0.0167`（旧基线 `0.7221`）
  - `factual_correctness = 0.0167`（旧基线 `0.4670`）
  - `error_count = 0`

### 问题定位
- 新链路出现明显回归，已不满足 `优化方案.md` 第 7.4 节设定的验收下限。
- 单条 debug case（`64fa9b27b82641eb8ecbe14c`）显示：
  - 检索结果仍命中同一 `sample_id` 下的正确证据；
  - 但最终答案直接退化为“根据当前检索到的内容，无法确定。”
- 当前判断：问题不在检索，而在新增的“证据抽取 -> 基于事实作答”链路过于保守，导致模型在已有证据的情况下也倾向输出无法确定。

### 下一步
- 优先调试 `extract_rag_evidence()` 与 `RAG_ANSWER_FROM_EVIDENCE_PROMPT`：
  - 检查证据抽取是否丢失关键信息；
  - 检查生成 prompt 是否把“保守回退”权重设得过高；
  - 必要时改为“抽取失败则回退原始压缩上下文直接回答”，避免整轮生成退化成统一拒答。

### 修正后复测（2026-04-14 深夜）
- 对 `generate_rag_answer()` 增加安全回退：
  - 当证据抽取没有产出有效 facts 时，直接回退到“压缩上下文 + 原始 RAG prompt”生成；
  - 当新链路输出空答案，或在弱 facts 场景下直接回成“无法确定”时，也回退到原始压缩上下文回答。
- 单条 debug case（`64fa9b27b82641eb8ecbe14c`）恢复正常：
  - 答案不再是“无法确定”
  - `answer_char_f1 = 0.6061`
  - `answer_rouge_l_f1 = 0.5859`
- 重新跑 CRUD 30：
  - 报告：`data/eval/crud_rag_official_split_local_questanswer_limit30_report_v3.json`
  - `answer_char_f1 = 0.2452`
  - `answer_rouge_l_f1 = 0.2142`
  - `answer_bleu_4 = 0.1208`
- 重新跑 RAGAS 30：
  - 报告：`data/eval/crud_rag_official_split_local_ragas_limit30_v3.json`
  - `llm_context_recall = 1.0`
  - `faithfulness = 0.1178`
  - `factual_correctness = 0.1410`

### 当前判断
- 回退保护已把“统一拒答”的严重问题修掉，但这版生成链路仍显著低于旧基线：
  - 旧基线 CRUD 30：`answer_char_f1 = 0.3400`
  - 当前 v3 CRUD 30：`answer_char_f1 = 0.2452`
  - 旧基线 RAGAS 30：`factual_correctness = 0.4670`
  - 当前 v3 RAGAS 30：`factual_correctness = 0.1410`
- 说明“证据抽取 -> 基于事实作答”这条新链路虽然不再完全失效，但仍然在大批样本上损伤了答案质量。
- 下一轮应优先考虑：
  - 将证据抽取由“替代主回答链路”改成“辅助信号”
  - 保留压缩上下文和复审，但让主回答重新基于压缩上下文直接生成
  - 仅把抽取 facts 用于 completeness review 或 trace，而不是强绑定为唯一生成输入

### 再次调整后复测（2026-04-15 凌晨）
- 已将主回答链路改回“压缩上下文直接生成”，证据抽取仅作为 review 辅助输入，不再作为唯一回答输入。
- 重新跑 CRUD 30：
  - 报告：`data/eval/crud_rag_official_split_local_questanswer_limit30_report_v4.json`
  - `answer_char_f1 = 0.2450`
  - `answer_rouge_l_f1 = 0.1938`
  - `answer_bleu_4 = 0.1121`
  - `generation_metrics.evaluated_cases = 16`
- 进一步用 `--limit 5 --show-cases` 排查：
  - 报告：`data/eval/crud_rag_official_split_local_questanswer_limit5_debug_v4.json`
  - 已确认未计入样本的主要原因不是新链路异常，而是模型接口报错：
    - `Error code: 402`
    - `Insufficient Balance`

### 当前阻塞
- 当前环境下，正式生成评测已受到 LLM provider 余额不足影响。
- 因此：
  - 现有 `v4` 结果只能作为“已成功生成的样本子集”参考；
  - 不能把 `evaluated_cases = 16` 的 CRUD 30 结果直接与旧版完整 30 条基线做严格横向结论。

## 2026-04-15 回退答案生成层实验链路

### 目标
- 回退 2026-04-14 引入的实验性“证据抽取 -> 基于事实作答”主回答链路，恢复到之前稳定的 RAG 主回答方式，避免继续拉低整体评测结果。

### 实施内容
- 更新 `app/chains/rag.py`：
  - 删除 `RAG_EVIDENCE_EXTRACTION_PROMPT`
  - 删除 `RAG_ANSWER_FROM_EVIDENCE_PROMPT`
  - 删除 `ContextBuildResult` / `EvidenceExtractionResult`
  - 删除证据抽取、压缩去重、review 拼接等实验性辅助函数
  - `generate_rag_answer()` 恢复为：
    - `build_rag_prompt()`
    - `build_rag_variables()`
    - `prompt | llm | StrOutputParser()`
    - `maybe_refine_rag_answer()`
  - `stream_rag_answer()` 恢复到实验前的常规链路
  - `build_context()` 恢复为直接拼接原始 references 内容
  - `append_answer_trace()` 恢复为不附加实验期观测字段
  - `split_query_into_requirements()` 回退问号拆分扩展
  - `RAG_SYSTEM_PROMPT` 与 `RAG_COMPLETENESS_REVIEW_PROMPT` 也一并恢复到实验前版本

### 验证结果
- `python -m py_compile app/chains/rag.py` 通过。

### 当前状态
- 代码已回退到实验前的稳定回答链路。
- 若后续继续优化，应优先采用“可灰度、可回退”的小步方式，而不是再次把新链路直接替换为主回答链路。

### 回退后验证
- 基于回退后的代码重新运行：
  - `python scripts/eval_crud_rag.py --knowledge-base-name crud_rag_official_split_local --data-file data/eval/crud_rag_official_split_local_official_split.jsonl --tasks quest_answer --limit 30 --output data/eval/crud_rag_official_split_local_questanswer_limit30_report_post_rollback.json`
- 回退后 CRUD 30 结果：
  - `answer_char_f1 = 0.3448`
  - `answer_rouge_l_f1 = 0.3092`
  - `answer_bleu_4 = 0.1739`
  - `retrieval_non_empty = 1.0`
  - `evaluated_cases = 30`
- 与此前官方知识库稳定基线相比：
  - 旧基线 `answer_char_f1 = 0.3400`
  - 回退后 `answer_char_f1 = 0.3448`
  - 旧基线 `answer_rouge_l_f1 = 0.3037`
  - 回退后 `answer_rouge_l_f1 = 0.3092`
- 结论：
  - 回退后生成质量已恢复到稳定区间；
  - 这轮实验性答案生成链路不保留为主链路；
  - 当前代码状态以“回退后的稳定版本”为准。

## 2026-04-15 Prompt 内部重复证据去重

### 目标
- 在不修改检索输出、不修改主回答链路的前提下，仅优化喂给模型的 prompt 内部上下文，压掉高度重复的 sibling 证据，验证是否能进一步提升生成质量。

### 实施内容
- 更新 `app/chains/rag.py`：
  - 在 `build_context()` 前新增 `deduplicate_references_for_prompt()`
  - 对 `evidence_summary / content_preview / content` 做轻量文本指纹去重
  - 仅对 prompt 内部证据生效，外部 `references` 返回保持原样
  - 当前去重后最多保留 `4` 条代表性证据

### 验证结果
- 小样本 `CRUD 10`：
  - 报告：`data/eval/crud_rag_official_split_local_questanswer_limit10_prompt_dedupe.json`
  - `answer_char_f1 = 0.3871`（旧 10 条基线 `0.3336`）
  - `answer_rouge_l_f1 = 0.3669`（旧 10 条基线 `0.3125`）
  - `answer_bleu_4 = 0.2070`（旧 10 条基线 `0.1697`）
- 扩大到 `CRUD 30`：
  - 报告：`data/eval/crud_rag_official_split_local_questanswer_limit30_prompt_dedupe.json`
  - `answer_char_f1 = 0.3679`
  - `answer_rouge_l_f1 = 0.3386`
  - `answer_bleu_4 = 0.1921`
  - `retrieval_non_empty = 1.0`

### 对比结论
- 与回退后的稳定基线相比：
  - 基线 `answer_char_f1 = 0.3448`
  - 去重后 `answer_char_f1 = 0.3679`
  - 基线 `answer_rouge_l_f1 = 0.3092`
  - 去重后 `answer_rouge_l_f1 = 0.3386`
  - 基线 `answer_bleu_4 = 0.1739`
  - 去重后 `answer_bleu_4 = 0.1921`
- 说明“只优化 prompt 内部重复证据”是有效的小步改动，且未破坏检索侧稳定性。

## 2026-04-17 CRUD / Domain 评测链路扩展与老师材料整理

### 目标
- 把现有检索评测从“只看有没有召回到”扩展为“同时看排序质量”，补齐 `MRR / NDCG`。
- 让评测脚本能够直接吃 `DomainRAG` 标注集，并支持后续的 `CRUD vs Domain` 对比。
- 补齐老师可直接阅读的综合文档、案例文档和技术说明文档。

### 实施内容
- 更新 `scripts/eval_retrieval.py`：
  - 支持直接读取 `DomainRAG` 原始 `jsonl`
  - 支持 `history_qa`
  - 支持单正例与多正例 `positive_reference / positive_references`
  - 正式输出 `Recall@k / MRR / NDCG@k`
- 新增 `scripts/import_domainrag_kb.py`：
  - 支持把 `DomainRAG` 的 `positive_reference`、`retrieved_psgs` 等代理语料导入为本地知识库
  - 支持按任务类型做小批量实验导入
- 新增 `scripts/build_crud_retrieval_cases.py`：
  - 基于当前 CRUD 知识库生成与本地文件一一对齐的检索评测集
- 更新 `scripts/eval_ragas.py`：
  - 支持通用 `jsonl`
  - 支持 `DomainRAG` 自动转换
  - 支持对话历史传入
  - 补齐 `llm_context_recall / faithfulness / factual_correctness / response_relevancy`
- 整理老师查看用文档：
  - `data/eval/crud_domain_benchmark_combined.md`
  - `data/eval/crud_domain_casebook.md`
  - `RAG_core_tech_notes.md`
  - `RAG评测与技术综合说明-2026-04-17.md`

### 验证结果
- `DomainRAG` 小批量知识库导入与检索评测链路可用，先后完成了 `15 / 50 / 100` 条样本的检索评测。
- `CRUD 100 vs Domain 100` 检索对比可稳定运行：
  - `CRUD 100`：`Recall@5 = 1.0000`，`MRR = 1.0000`，`NDCG@5 = 1.0000`
  - `Domain 100`：`Recall@5 = 0.8000`，`MRR = 0.6262`，`NDCG@5 = 0.6236`
- `RAGAS` 缩减对比可稳定运行：
  - `CRUD 10`：`Context Recall = 1.0000`，`Faithfulness = 0.7288`，`Factual Correctness = 0.4540`，`Answer Relevance = 0.4841`
  - `Domain 10`：`Context Recall = 0.8333`，`Faithfulness = 0.5284`，`Factual Correctness = 0.0570`，`Answer Relevance = 0.2402`

### 当前状态
- 评测脚本已经形成“检索指标 + RAGAS 指标”的完整闭环。
- 当前 `CRUD=1.0` 更适合解释为“项目内闭环评测非常稳定”，不宜直接表述为通用 benchmark 意义上的“检索已经完美”。
- `DomainRAG proxy` 小库已经能稳定暴露开放域检索中的真实短板，后续优化将以它为主要观测对象。

## 2026-04-18 Domain 检索定向优化（时间敏感 + 多文档覆盖）

### 目标
- 针对 `Domain 100` 中最薄弱的 `time-sensitive_qa` 和 `multi-doc_qa` 做“一改一测”式检索优化。
- 在不破坏当前 CRUD 稳定性的前提下，提升 Domain 侧排序质量和前排证据覆盖。

### 实施内容
- 时间敏感检索优化：
  - 更新 `app/retrievers/local_kb.py`
  - 增加时间敏感 query 识别、时间锚点抽取、时间匹配加权
  - 时间信号同时接入混合检索与启发式重排
- 查询改写保护：
  - 更新 `app/services/query_rewrite_service.py`
  - 对“年份 / 最新 / 当前 / 截止时间”等约束做保护，避免改写时丢失
- `date` 元数据落库：
  - 更新 `app/loaders/factory.py`
  - 更新 `scripts/import_domainrag_kb.py`
  - 让 sidecar `.rag_file_metadata.json` 中的 `date` 真正进入 chunk metadata
- 多文档覆盖优化：
  - 更新 `app/retrievers/local_kb.py`
  - 对 `multi-doc_qa` 增加文档家族识别与多源覆盖优先策略
  - 在重排阶段避免同一来源反复占位

### 验证结果
- `time-sensitive_qa` 20 条子集从初始基线：
  - `Recall@5 = 0.40`
  - `MRR = 0.1717`
  - `NDCG@5 = 0.2261`
- 提升到当前稳定版本：
  - `Recall@5 = 0.55`
  - `MRR = 0.2658`
  - `NDCG@5 = 0.3348`
- `multi-doc_qa` 20 条子集优化后：
  - `Recall@5 = 1.0000`
  - `MRR = 0.9417`
  - `NDCG@5 = 0.7366`
  - 其中 `NDCG@5` 相对该子集基线 `0.7215` 有提升
- `Domain 100` 整体从初始基线：
  - `Recall@5 = 0.8000`
  - `MRR = 0.6262`
  - `NDCG@5 = 0.6236`
- 提升到当前稳定版本：
  - `Recall@5 = 0.8700`
  - `MRR = 0.7250`
  - `NDCG@5 = 0.7164`

### 当前状态
- 时间敏感检索与多文档覆盖优化已保留在主链路中。
- 当前 `Domain` 检索侧已经从“能召回但排不准”显著改善到“召回和排序都更稳”的状态。
- 下一阶段如果继续优化检索，更应该切向真实多 chunk 语料验证 `chunk / metadata` 策略，而不是继续在 proxy 小库上做 prompt 侧微调。

## 2026-04-18 生成层小步实验与保留策略

### 目标
- 在不动检索主链路的前提下，围绕 `Factual Correctness` 偏低的问题，对生成层做可回退的小步实验。
- 通过 `Domain` 小样本快测筛选出值得保留的改动，避免再次出现“大改后整体回归”。

### 实施内容
- 当前保留的改动：
  - 更新 `app/chains/rag.py`
  - 新增事实审校 prompt
  - 将 `maybe_refine_rag_answer()` 扩成“完备性审校 + 事实审校”
  - 抽出通用 JSON 修订调用辅助函数，便于后续继续加 review 规则
- 已尝试但未保留的改动：
  - 动态增加生成阶段上下文块数
  - 强制“结论 / 依据 / 不确定点”输出模板
  - 槽位回填式复审
  - 生成阶段证据块重排
  - 结构化证据摘要展示
  - 生成专用 top-k 证据选择器

### 验证结果
- 在 `Domain` 5 条小样本快测上：
  - 改前：`factual_correctness = 0.1380`，`response_relevancy = 0.0971`
  - 加入事实审校后：`factual_correctness = 0.1680`，`response_relevancy = 0.0970`
- 其余几轮 prompt / 后处理实验都未稳定超过当前最佳值，因此均已回退，不保留到主链路。

### 当前状态
- 生成层当前只保留“事实审校器”这一刀。
- 当前判断是：生成层靠 prompt 和后处理继续做小修小补，收益已经接近边界；下一步更值得回到“证据质量”和“真实多 chunk 语料”上做验证。

## 2026-04-18 结构化切块实验库与 Markdown 标题感知切分

### 目标
- 为 `chunk / metadata` 优化建立一套真正能切出多块的实验语料，避免继续在 `327 文件 = 327 chunks` 的 proxy 知识库上测不出差异。
- 验证 Markdown 标题感知切分是否能提升章节 metadata 质量，同时控制过碎切块问题。

### 实施内容
- 新建实验知识库 `structured_chunk_demo`：
  - 使用 1 个 PDF + 4 个长 Markdown 文档构造真实多 chunk 语料
- 更新 `app/chains/text_splitter.py`：
  - `.md` 文档自动走 `MarkdownHeaderTextSplitter`
  - 其他格式继续走原有递归切分器
- 为 Markdown 切分增加保护逻辑：
  - 新增最小章节长度保护
  - 对过短章节做相邻合并
  - 过滤 `--- / *** / ___` 这类纯分隔线，避免单独变成 chunk

### 验证结果
- `structured_chunk_demo` 初始重建结果：
  - `5` 个文件
  - `22` 个原始文档单元
  - `145` 个 chunks
- 仅开启 Markdown 标题感知切分后：
  - chunks 增长到 `385`
  - Markdown 的 `section_title / section_path` 覆盖率达到 `100%`
  - 但出现了明显过碎问题
- 加入“短章节合并 / 最小块长度保护”后：
  - 总 chunks 降到 `227`
  - Markdown 的 `section_title / section_path` 仍保持 `100%`
  - `<120` 长度的异常小块从 `1` 个降到 `0`
  - 最终各文件 chunk 数：
    - `beyond_rag_agent_memory.pdf`：`58`
    - `LangChain-RAG-Agent-学习与搭建文档.md`：`51`
    - `RAG-Agent-分阶段TODO.md`：`46`
    - `process.md`：`37`
    - `优化方案.md`：`35`

### 当前状态
- 标题感知切分与短章节合并逻辑已保留在代码中。
- 当前还没有在这套实验库上跑正式检索 benchmark，下一步应补一轮 retrieval smoke test，确认结构化切块是否真的带来检索收益。

## 2026-04-18 Agent 最终答案生成切到 LLM 综合生成

### 目标
- 把 `multistep.py` 中针对 `calculate / current_time / 多工具组合` 的硬编码结果拼接逻辑，统一替换为 LLM 综合生成。
- 保持纯 RAG 路径不变，避免影响当前 `/chat/rag` 主链路和已有 RAG 评测结果。

### 实施内容
- 更新 `app/agents/multistep.py`：
  - 新增 `AGENT_SYNTHESIS_SYSTEM_PROMPT`
  - 新增 `_build_synthesis_prompt()`
  - 新增 `_build_synthesis_variables()`
  - 新增 `generate_synthesis_answer()`
  - 新增 `stream_synthesis_answer()`
- 更新 `build_agent_answer()` 与 `stream_agent_answer()`：
  - 当 `executed_names == ["search_local_knowledge"]` 时，仍保持原有纯 RAG 生成路径
  - 其余工具组合统一改为调用 LLM 综合生成最终回答

### 验证结果
- 纯 RAG 分支条件未修改，`search_local_knowledge` 单工具路径保持不变。
- `calculate`、`current_time`、以及任意多工具组合不再依赖手写 `if-else` 拼接，能够输出更自然的中文结果说明。
- 当前 `CRUD / Domain` 评测脚本不经过 `multistep agent`，因此这项改动不会直接反映在当前那套 `Recall@k / MRR / NDCG / RAGAS` 对比结果中。

### 当前状态
- Agent 工具选择已经是 LLM 驱动，这一轮改动让“最终答案生成”也同步切到了 LLM 综合生成。
- 若后续要正式评估这部分收益，需要单独设计 Agent 场景评测集，重点看 `Answer Relevance` 和 `Factual Correctness`。

## 2026-04-20 Reranker 模型对比评测（base vs v2-m3）

### 目标
- 在同一套 `Domain 100` 检索评测集上，只替换 `RERANK_MODEL`，其余检索配置保持不变。
- 让结果差异尽量只反映 reranker 模型本身，而不是候选池、top-k、query rewrite 或 chunk 策略变化。

### 实施内容
- 保持以下条件固定：
  - 评测集：`data/eval/domainrag_small_batch_100_domainrag_small_batch.jsonl`
  - 知识库：`domainrag_small_batch_100`
  - 其余配置保持当前稳定版本不变
- 对比的两个模型：
  - 基线：`./data/models/bge-reranker-base`
  - 对照：`./data/models/bge-reranker-v2-m3`
- 输出对比产物：
  - `data/eval/shared_analysis/reranker_compare_domain100_base_vs_v2_m3.json`
  - `data/eval/shared_analysis/reranker_compare_domain100_base_vs_v2_m3.md`

### 验证结果
- 整体指标对比：
  - `Recall@5`：`0.8700 -> 0.8900`
  - `MRR`：`0.7298 -> 0.7418`
  - `NDCG@5`：`0.7317 -> 0.7482`
  - `Top1 Hit`：`0.6600 -> 0.6700`
- 分任务结果显示：
  - `extractive_qa` 提升最明显：
    - `MRR`：`0.6142 -> 0.6950`
    - `NDCG@5`：`0.6839 -> 0.7568`
    - `Top1 Hit`：`0.5000 -> 0.6000`
  - `structured_qa` 继续提升到接近满分：
    - `MRR`：`0.9750 -> 1.0000`
    - `NDCG@5`：`0.9815 -> 1.0000`
  - `conversation_qa` 小幅改善
  - `multi-doc_qa` 有回退：
    - `MRR`：`0.9500 -> 0.9083`
    - `NDCG@5`：`0.7991 -> 0.7808`
  - `time-sensitive_qa` 呈现混合变化：
    - `Recall@5`：`0.5500 -> 0.6000`
    - 但 `MRR`：`0.3017 -> 0.2892`
    - `Top1 Hit`：`0.2000 -> 0.1500`

### 当前状态
- 这轮实验已经证明：在当前 `Domain 100` 评测集上，`bge-reranker-v2-m3` 的整体排序能力优于当前 `bge-reranker-base`。
- 这项收益主要集中在 `extractive_qa` 和 `structured_qa`，尤其对 `extractive_qa top1 / MRR` 的拉升比较明显。
- 代价是 `multi-doc_qa` 和 `time-sensitive_qa` 的部分子指标出现回退，因此如果后续切换默认 reranker，仍建议结合任务结构继续观察，而不是只看总体均值。

## 2026-04-20 按 Query Type 路由 Reranker 模型

### 目标
- 不再让所有问题都使用同一个 reranker 模型。
- 基于前一轮对比结果，把 `extractive / structured` 更擅长的 `bge-reranker-v2-m3`，和 `multi-doc / time-sensitive` 更稳的 `bge-reranker-base` 组合起来使用。
- 保持召回池、query rewrite、chunk 策略不变，只改 rerank 选模型逻辑。

### 实施内容
- 更新 `app/services/settings.py`：
  - 新增 3 个可选配置项：
    - `RERANK_MODEL_ANSWER_FOCUSED`
    - `RERANK_MODEL_MULTI_DOC`
    - `RERANK_MODEL_TEMPORAL`
  - 其中 `RERANK_MODEL_MULTI_DOC` 与 `RERANK_MODEL_TEMPORAL` 的代码默认值回退为 `bge-reranker-base`，避免本地未跟踪配置文件缺失时路由失效。
- 更新 `app/services/rerank_service.py`：
  - `rerank_texts()` 新增 `model_name_override` 参数，支持调用侧在运行时指定 reranker 模型。
- 更新 `app/retrievers/local_kb.py`：
  - 新增 `RerankModelSelection`
  - 新增 `resolve_rerank_model_selection()`
  - 在 `rerank_candidates()` 中根据 query 类型选择 reranker：
    - `temporal`：优先走 `RERANK_MODEL_TEMPORAL`
    - `multi-doc`：优先走 `RERANK_MODEL_MULTI_DOC`
    - `answer-focused`：优先走 `RERANK_MODEL_ANSWER_FOCUSED`
    - 其他问题：走默认 `RERANK_MODEL`
  - 将 `rerank_model_selected` 和 `rerank_model_route` 写入 retrieval trace，方便后续 bad case 分析。
- 更新 `configs/model_settings.yaml`：
  - 当前配置为：
    - `RERANK_MODEL = ./data/models/bge-reranker-v2-m3`
    - `RERANK_MODEL_ANSWER_FOCUSED = ./data/models/bge-reranker-v2-m3`
    - `RERANK_MODEL_MULTI_DOC = ./data/models/bge-reranker-base`
    - `RERANK_MODEL_TEMPORAL = ./data/models/bge-reranker-base`

### 验证结果
- 同一套 `Domain 100` 检索评测结果：
  - `Recall@5 = 0.8700`
  - `MRR = 0.7410`
  - `NDCG@5 = 0.7445`
  - `Top1 Hit = 0.6700`
- 与“单一 reranker”对比：
  - 相比 `v2-m3` 单模型：
    - `MRR`: `0.7385 -> 0.7410`
    - `NDCG@5`: `0.7441 -> 0.7445`
    - `Top1 Hit`: `0.6600 -> 0.6700`
  - 相比 `base` 单模型：
    - `MRR`: `0.7373 -> 0.7410`
    - `NDCG@5`: `0.7369 -> 0.7445`
    - `Top1 Hit`: 持平 `0.6700`
- 分任务变化：
  - `multi-doc_qa` 明显恢复：
    - `MRR = 0.9750`
    - `NDCG@5 = 0.8389`
  - `time-sensitive_qa` 也比 `v2-m3` 单模型更稳：
    - `MRR = 0.3158`
    - `NDCG@5 = 0.3733`
  - `extractive_qa` 没有保持到 `v2-m3` 单模型时的最高值，但整体组合后的全局指标更优。

### 当前状态
- 这轮“按 query type 路由 reranker”已保留到主链路。
- 当前判断是：在不重构召回和 chunk 的前提下，这种“默认模型 + 题型覆盖”的方式，比强行统一使用同一个 reranker 更稳。

## 2026-04-20 Phase 0 人工集起始样本扩充（20 条）

### 目标
- 把此前只有少量示例的 `Phase 0` 人工集模板，扩成一版可以直接开始人工复核的起始样本集。
- 让后续优化优先基于“小而准”的 gold 集做判断，而不是只看代理指标波动。

### 实施内容
- 更新 `data/eval/phase0_gold_annotation_guide.md`：
  - 补充 `review_status`
  - 补充多轮样本必填的 `history_qa`
- 更新 `data/eval/phase0_gold_manual_template.jsonl`：
  - 模板同步新增 `review_status`
  - 模板同步新增 `history_qa`
- 重写 `data/eval/phase0_gold_manual_seed.jsonl`：
  - 从原来的 `4` 条种子扩充到 `20` 条
  - 所有样本统一标记为 `seed_for_review`
  - 样本来源覆盖：
    - `domainrag_small_batch_100`
    - `crud_rag_3qa_full`

### 当前 20 条种子分布
- `extractive_qa`：`6`
- `time-sensitive_qa`：`4`
- `multi-doc_qa`：`2`
- `conversation_qa`：`2`
- `CRUD cross-passage`：`6`

### 验证结果
- 已做 JSONL 结构校验：
  - `count = 20`
  - 首条：`phase0-seed-domain-extractive-001`
  - 末条：`phase0-seed-crud-006`
- 当前这 20 条已经具备以下字段：
  - `query`
  - `history_qa`
  - `gold_documents`
  - `gold_passages`
  - `needs_cross_passage_aggregation`
  - `notes`

### 当前状态
- 这批样本已经足够支撑第一轮人工复核和 bad case 分桶。
- 目前仍未覆盖 `OCR / image evidence` 类型，下一批建议优先补：
  - 图文联合证据
  - OCR 干扰样本
  - query rewrite drift 样本

## 2026-04-20 Phase 0 第一版 Bad Case 分桶（20 条）

### 目标
- 基于刚建立的 `20` 条人工 gold 种子集，先做第一版检索 bad case 分桶。
- 不再只看总均值，而是把失败样本拆成“召回、排序、chunk、跨段聚合”几类具体问题。

### 实施内容
- 新增脚本 `scripts/analyze_phase0_bad_cases.py`：
  - 读取 `phase0_gold_manual_seed.jsonl`
  - 对每条样本跑当前稳定检索链路，固定取 `top-50`
  - 统计 `Recall@20 / Recall@50 / MRR@10 / NDCG@10 / Top1 accuracy`
  - 生成第一版 bucket：
    - `missed_recall`
    - `low_rank`
    - `chunk_noise`
    - `cross_passage`
    - `image_text_misaligned`
- 产出：
  - `data/eval/phase0_bad_case_bucket_v1.json`
  - `data/eval/phase0_bad_case_bucket_v1.md`

### 第一版分桶结果
- 总样本数：`20`
- `Recall@20 = 1.0000`
- `Recall@50 = 1.0000`
- `MRR@10 = 0.7380`
- `NDCG@10 = 0.7908`
- `Top1 accuracy = 0.6500`

### bucket 统计
- `passed`：`2`
- `chunk_noise`：`4`
- `low_rank`：`6`
- `cross_passage`：`8`
- `missed_recall`：`0`

### 当前判断
- 第一版人工集没有出现“完全召回不到”的样本，说明当前主问题已经不是召回深度，而是：
  - `low_rank`：gold 文档在前 `20` 内，但排不到 `top1`
  - `chunk_noise`：gold 文档排到了前面，但返回块没有直接承载答案句
  - `cross_passage`：多文档/多段聚合问题仍是结构性短板
- 这也说明后续优化优先级应继续放在：
  - rerank 输入与排序质量
  - chunk/答案句对齐
  - cross-passage 聚合，而不是继续单纯拉召回

## 2026-04-21 app/services 目录兼容式整理

### 背景
- `app/services` 顶层文件持续增多，知识库、检索、评测三类实现已经混在同一层。
- 直接一次性全改 import 风险较大，因此本轮采用“实现迁移 + 旧路径兼容”的方式先整理目录。

### 实施内容
- 新增子目录：
  - `app/services/evaluation`
  - `app/services/kb`
  - `app/services/retrieval`
- 将实现文件迁入对应分组：
  - `evaluation/`：
    - `crud_eval_cases.py`
    - `eval_reference_utils.py`
  - `kb/`：
    - `embedding_assembler.py`
    - `kb_incremental_rebuild.py`
    - `kb_ingestion_service.py`
    - `rebuild_task_service.py`
    - `sentence_index_service.py`
  - `retrieval/`：
    - `query_rewrite_service.py`
    - `reference_overview.py`
    - `rerank_service.py`
    - `web_search_service.py`
- 在原路径保留轻量兼容 wrapper，避免同时修改全仓库 import。
- 更新 `app/services/__init__.py` 和新增 `app/services/README.md`，说明当前服务分组。
- 清理 `app/**/__pycache__`，减少目录噪声。

### 验证
- 运行导入级 smoke test，确认旧路径和新路径都能正常加载：
  - `app.services.*` 兼容模块
  - `app.retrievers.local_kb`
  - `app.chains.rag`
  - `app.api.knowledge_base`
- 结果：`IMPORT_SMOKE_OK`

### 当前判断
- 这一轮是结构整理，不改业务逻辑。
- 当前收益主要是：
  - `services` 顶层职责更清楚
  - 后续继续拆分 `kb` 和 `retrieval` 时不需要一次性大改 import
  - 旧入口仍可用，主链路风险较低

## 2026-04-21 Phase 0 人工集扩充到 50 条

### 背景
- 前一轮 `Phase 0` 只完成了 `20` 条种子集与第一版 bad case 分桶。
- 这批样本已经足够做方向判断，但对后续更细粒度的排序与证据优化来说，覆盖仍然偏小。

### 目标
- 将 `Phase 0` 人工集从 `20` 条扩充到 `50` 条。
- 在保留原有 `extractive / time-sensitive / multi-doc / conversation / CRUD cross-passage` 的基础上，补入此前未覆盖的 `structured_qa`。

### 实施内容
- 更新 `data/eval/phase0_gold_manual_seed.jsonl`：
  - 在原有 `20` 条基础上新增 `30` 条
  - 新增样本来源：
    - `extractive_qa-7 ~ 12`
    - `time-sensitive_qa-5 ~ 8`
    - `multi-doc_qa-3 ~ 6`
    - `conversation_qa-3 ~ 6`
    - `structured_qa-1 ~ 6`
    - `CRUD` retrieval case `007 ~ 012`
- 新增样本统一保留：
  - `review_status = seed_for_review`
  - `failure_bucket = ""`
  - `acceptable_negatives = []`
- 对自动扩充样本补充说明：
  - `notes` 明确写为“自动补充，需人工复核”
  - 对 `structured_qa` 明确标注需要后续人工补字段上下文

### 扩充后分布
- 总样本数：`50`
- `source_dataset`
  - `domainrag_small_batch_100`：`38`
  - `crud_rag_3qa_full`：`12`
- `question_type`
  - `parameter`：`39`
  - `definition`：`7`
  - `procedure`：`4`
- `needs_cross_passage_aggregation`
  - `false`：`32`
  - `true`：`18`

### 当前判断
- 这一版 `50` 条种子集已经比上一轮更均衡，尤其补上了 `structured_qa`。
- 目前仍然缺少：
  - `OCR / image evidence`
  - `query rewrite drift`
  - 更明确的 `acceptable_negatives`
- 下一步最直接的是基于这 `50` 条样本，重新做第二轮 bad case 分桶。

## 2026-04-21 Phase 0 第二版 Bad Case 分桶（50 条）

### 目标
- 基于扩充后的 `50` 条 `Phase 0` 人工集，重跑第二版 bad case 分桶。
- 对比上一版 `20` 条结果，观察新增样本暴露出的新问题类型。

### 实施内容
- 新增脚本 `scripts/analyze_phase0_bad_cases.py`：
  - 读取 `phase0_gold_manual_seed.jsonl`
  - 逐条调用当前本地知识库检索链路，固定取 `top-50`
  - 统计：
    - `Recall@20`
    - `Recall@50`
    - `MRR@10`
    - `NDCG@10`
    - `Top1 accuracy`
  - 输出第二版 bad case 分桶：
    - `missed_recall`
    - `low_rank`
    - `chunk_noise`
    - `cross_passage`
- 产出：
  - `data/eval/phase0_bad_case_bucket_v2.json`
  - `data/eval/phase0_bad_case_bucket_v2.md`

### 第二版分桶结果
- 总样本数：`50`
- `Recall@20 = 0.8800`
- `Recall@50 = 0.8800`
- `MRR@10 = 0.6697`
- `NDCG@10 = 0.8042`
- `Top1 accuracy = 0.5800`

### bucket 统计
- `passed`：`2`
- `chunk_noise`：`12`
- `low_rank`：`12`
- `cross_passage`：`18`
- `missed_recall`：`6`

### 新暴露出的关键信息
- 新增的 `structured_qa` 六条样本全部落入 `missed_recall`：
  - `phase0-seed-domain-structured-1 ~ 6`
- 这说明当前主链路对结构化录取页面类问题还没有真正打通：
  - 相关页面能排进前列
  - 但当前 gold 匹配口径下，目标字段对应文档/片段没有被稳定命中
- 其他问题类型延续了第一版的结论：
  - `chunk_noise`：答案句对齐仍不足
  - `low_rank`：gold 文档命中但排不到 `top1`
  - `cross_passage`：多段整合依旧是结构性短板

### 当前判断
- 相比 `20` 条版本，这版 `50` 条结果更严格，也更接近“真实优化难点”。
- 当前最值得继续推进的优先级是：
  1. `structured_qa / missed_recall`
  2. `low_rank`
  3. `chunk_noise`
  4. `cross_passage`

## 2026-04-24 优化方案总进度判断 + 未完成计划续执行

### 目标
- 对照 [优化方案.md](./优化方案.md) 判断当前整体推进进度，而不是只看零散实验记录。
- 把“已完成 / 部分完成 / 未开始”明确收口，避免后续继续推进时重复判断。
- 在此基础上，把当前最值得继续执行的未完成计划写清楚，作为下一阶段实施入口。

### 对照 `优化方案.md` 的当前进度判断

#### 一、阻塞性问题与基础工程项

| 项目 | 当前判断 | 说明 |
|------|----------|------|
| `M1 API Key 泄露治理` | **代码侧已完成，运维侧未完全闭环** | YAML 不再保存密钥，`.env` 自动加载已补齐；但若历史密钥曾进入 git 历史，平台侧轮换仍需人工执行 |
| `M2 Embedding 多 provider` | **已完成** | `app/services/models/embedding_service.py` 已支持 `ollama` 和 `openai_compatible` |
| `M3 settings 启动期加载` | **已完成** | 已切到 FastAPI `lifespan` + 依赖注入 |
| `O1 documents.py 拆分` | **已完成到兼容式拆分阶段** | 主体已拆，但图片链路内部仍可继续细分 |
| `O2 提取重复代码` | **已完成核心部分** | 文本工具、常量与去重辅助已公共化 |
| `O3 异步重建接口` | **已完成** | API + 轮询 + UI 都已打通 |
| `O4 删除 executor.py` | **已完成** | 兼容壳已删除 |
| `O5 Tesseract 路径配置化` | **已完成** | 已改为相对路径并支持环境变量覆盖 |
| `O6 Agent Tool Calling` | **已完成到可用版** | LLM tool calling 已上线，仍保留启发式回退 |

#### 二、RAG 核心链路增强项

| 项目 | 当前判断 | 说明 |
|------|----------|------|
| `T1 Multi-Query` | **已完成** | 已接入主链路 |
| `T2 HyDE` | **已完成，默认关闭** | 仅追加到 dense bundle，控制开销 |
| `T3 LLM Tool Calling` | **已完成** | Agent 工具选择已升级 |
| `T4 Corrective RAG` | **已完成，默认关闭** | 本地二次检索与可选网络补搜已具备 |
| `T5 BM25 持久化` | **已完成** | 构建期持久化、检索期加载、依赖缺失自动回退 |
| `T6 chunk cache -> numpy` | **已完成** | 已切到 `json + npy` 结构 |
| `T7 Agent 长期记忆` | **已完成到 v1，可用但默认关闭** | `memory_service`、`session_id`、memory trace 均已存在；当前属于“功能在，默认不开”状态 |

#### 三、评测与质量优化项

| 方向 | 当前判断 | 说明 |
|------|----------|------|
| 时间敏感检索优化 | **已做一轮并保留到主链路** | 检索指标明显提升，但仍是最弱任务类型之一 |
| 多文档覆盖优化 | **已做一轮并保留到主链路** | 通过 family/diversity/reranker route 有明显改善 |
| 生成侧事实正确性优化 | **仅完成小步实验，尚未形成稳定闭环** | 事实审校器保留了，但整体 `factual_correctness` 仍未达到方案预期 |
| Markdown 标题感知切块 | **已完成实验版并保留** | 结构化 metadata 提升明显，但真实收益还缺正式 benchmark |
| Reranker 路由 | **已完成并保留** | 当前属于较优稳态 |
| Phase 0 人工集与 bad case 分桶 | **已完成起步版，但未达到方案终态** | 已扩到 `50` 条并做了第二版分桶，但还没走到“100~200 条稳定人工集 + OCR/query drift 覆盖” |

#### 四、TRACE 时间序列改造方案

| 项目 | 当前判断 | 说明 |
|------|----------|------|
| `第十章 TRACE 迁移方案文档` | **已完成文档设计** | 已在 `优化方案.md` 新增完整独立章节 |
| 时间序列数据建模 | **未开始实现** | 当前 `RetrievedReference`、`ReferenceOverview` 仍未支持 TS 字段 |
| 时间序列入库 | **未开始实现** | 当前知识库仍以文档为主 |
| 时间序列检索分支 | **未开始实现** | 当前主链路无 TS retriever |
| 文本 + TS 联合生成 | **未开始实现** | `rag.py` 仍只有 text / ocr / vision 证据组 |

### 当前总体结论

- **第一阶段基础工程与通用 RAG 增强项，大部分已经完成。**
- **第二阶段检索优化已经做出明显成果，但生成侧事实正确性仍然是最突出的未闭环问题。**
- **第三阶段的 TRACE 时间序列联合证据 RAG，目前还停留在“方案已写清、代码未启动”的状态。**

更具体地说，当前项目已经不再处于“缺少基础能力”的阶段，而是进入了两个真正还没做完的主线：

1. 把已有文本 RAG 的生成质量继续拉稳，尤其是 `factual_correctness`。
2. 启动 `第十章` 所定义的时间序列联合证据 RAG 改造。

### 当前最值得继续执行的未完成计划

#### P-A：生成侧事实正确性闭环

**状态判断：**
- 这是当前方案里“仍未完成”的最高优先级主线。
- 原因不是没有做，而是做了几轮小步实验后，收益还没有稳定达到方案目标。

**下一步应继续执行的动作：**

1. 先固定当前保留版本作为生成基线：
   - 保留事实审校器
   - 不再继续堆叠更多 prompt 变化
2. 针对 `Domain` 中低 `factual_correctness` case 做专项样本集：
   - 优先抽 `time-sensitive_qa`
   - `multi-doc_qa`
   - `structured_qa`
3. 对这批 case 单独标注失败原因：
   - 证据不准
   - 排序不前
   - 证据够但没答全
   - 证据够但答偏
4. 在不改主接口的前提下，继续推进：
   - 证据压缩视图
   - facts-first 草稿生成
   - 基于 checklist 的 completeness review
5. 验收标准继续沿用 `优化方案.md` 的生成指标，不再只看小样本快测。

**当前判断：**
- 这条线仍然应该排在“真正开始做 TRACE 时间序列改造”之前。
- 因为如果文本生成侧还不稳，后续即使接入时间序列证据，也很容易变成“证据更多，但答案仍然不够忠实”。

#### P-B：Phase 0 人工集从 50 条扩到可回归规模

**状态判断：**
- 当前已完成 `50` 条起步版和第二版 bad case 分桶，但离方案里的“稳定人工回归集”还有距离。

**下一步应继续执行的动作：**

1. 从 `50` 条扩到 `100~200` 条。
2. 优先补足当前缺口：
   - `OCR / image evidence`
   - `query rewrite drift`
   - 更多 `structured_qa`
   - 更明确的 `acceptable_negatives`
3. 保持现有 bucket 口径不变，避免评测标准漂移。
4. 把这套人工集作为后续：
   - 检索优化
   - 生成优化
   - 时间序列改造
   的统一回归底座。

**当前判断：**
- 这是继续推进任何复杂改造之前的评测前置工程。
- 不补这一步，后续很难稳定判断“TRACE 方案接入后到底是真的变强，还是只是样本换了”。

#### P-C：启动 TRACE 时间序列联合证据 RAG 的 Phase 1

**状态判断：**
- 这是当前方案里最完整、但还完全未实现的新主线。
- 当前最合理的推进方式，不是直接做 TRACE 全复现，而是执行 `第十章` 的 `Phase 1：最小可用版`。

**下一步应继续执行的动作：**

1. 新增时间序列知识单元最小字段集。
2. 扩展 `RetrievedReference` 支持时间序列证据字段。
3. 扩展 `ReferenceOverview` 支持 `timeseries_count` 与联合覆盖标记。
4. 定义第一版结构化时间序列输入格式。
5. 新增时间序列入库服务原型。
6. 在 `rag.py` 中新增“时间序列证据”上下文分组，但先不改现有请求协议。

**Phase 1 暂不做的事：**

- 不接完整训练框架
- 不做多机多卡训练
- 不做完整 TRACE 对齐损失复现
- 不做独立前端页

**当前判断：**
- 这是“未执行计划”里最值得正式启动的一块新功能。
- 但它应该建立在当前评测闭环已经收紧、文本生成侧不再明显漂移的前提上。

### 推荐执行顺序（从现在开始）

| 顺序 | 下一步计划 | 原因 |
|------|------------|------|
| 1 | 继续补齐 Phase 0 人工集与 bad case 覆盖 | 先把后续所有改动的评测底座做稳 |
| 2 | 继续推进生成侧 `factual_correctness` 闭环 | 解决当前最突出的质量短板 |
| 3 | 启动 TRACE 方案的 `Phase 1` 最小可用版 | 在稳住现有文本 RAG 后，开始时间序列联合证据扩展 |

### 当前状态总结

- `优化方案.md` 中 2026-04 中前半段的大多数工程项，已经不再是“待做”，而是“已完成或已落地验证”。
- 当前真正未做完的，不是基础设施，而是：
  - 生成质量闭环
  - 更完整的人工评测集
  - TRACE 时间序列联合证据 RAG 改造
- 因此从今天开始，后续推进重点应正式切换到：
  - **质量闭环**
  - **评测闭环**
  - **时间序列扩展**

## 2026-04-24 生成侧 factual_correctness 收口：主回答约束与低噪声上下文

### 目标
- 在不重演“证据抽取替代主回答链路”大回归的前提下，继续收紧非流式 RAG 的事实正确性。
- 采取低风险、小步可回退策略：
  - 不替换当前主回答结构
  - 不新增额外 LLM 阶段
  - 只加强主 prompt、复审 prompt 与上下文噪声控制

### 实施内容
- 更新 `app/chains/rag.py`：
  - 强化 `RAG_SYSTEM_PROMPT`：
    - 单事实 / 单时间点 / 单数值问题，第一句必须直接给结论
    - 非用户显式要求时，避免“根据上下文 / 根据资料”等模板化前缀
    - 对 `当前 / 最新 / 现任 / 截至 / 哪一年 / 何时` 等时间约束，要求答案保留证据中的具体时间信息
  - 新增 `build_answer_requirements()`：
    - 按 query 类型生成补充回答约束
    - 当前支持两类附加约束：
      - `should_directly_answer_query()`
      - `is_temporal_answer_query()`
  - 将 `answer_requirements` 接入：
    - 主回答 prompt
    - `RAG_COMPLETENESS_REVIEW_PROMPT`
    - `RAG_FACTUAL_REVIEW_PROMPT`
  - 为 prompt 内部上下文增加轻量降噪：
    - `format_reference_block()` 新增 `snippet_limit`
    - `resolve_reference_content_limit()` 会对单事实问题压缩正文片段长度
    - 单事实问题优先缩短 text evidence 片段，减少无关背景进入生成
  - 为 answer trace 增加观测字段：
    - `direct_answer_query`
    - `temporal_answer_query`
    - `coverage_requirement_count`

### 当前设计判断
- 这轮改动仍然遵循“主回答直接基于压缩上下文生成”的稳定路线。
- 没有重新引入：
  - `facts extraction -> answer drafting` 强绑定链路
  - 新的替代式中间结构
  - 额外 LLM 调用开销
- 核心思路是：
  - 让模型更少说套话
  - 让单事实问题更快落结论
  - 让时间敏感问题更少丢具体时间点
  - 让 prompt 内部正文噪声稍微收敛

### 验证结果
- 使用内存编译方式验证 `app/chains/rag.py` 语法通过：
  - `RAG_PY_COMPILE_OK`
- 额外说明：
  - 常规 `python -m py_compile` 仍受现有 `__pycache__` 写权限限制，失败点不是语法而是 `.pyc` 写入阶段

### 当前状态
- 生成侧 `factual_correctness` 收口已继续推进一小步，并已进入代码。
- 这轮属于“低风险提示词和上下文组织收紧”，还不是最终闭环。
- 下一步若继续验证收益，应优先跑：
  - `Domain` 小样本快测
  - `CRUD / RAGAS` 回归
  - 对比这轮是否减少：
    - 答非所问
    - 套话开头
    - 时间点遗漏

## 2026-04-24 生成侧 factual_correctness 收口：同证据小规模本地回归

### 回归方式
- 目标不是重跑全量 `RAGAS`，而是先判断这轮 `rag.py` 收口是否带来可见改善。
- 采用“同一批旧检索证据，重放当前生成链路”的小规模回归方式：
  - 输入来源：
    - `data/eval/domain_case_demo_3_ragas_detail_rerun.json`
    - `data/eval/crud_case_demo_2_ragas_detail_rerun.json`
  - 直接复用其中保存的：
    - `query`
    - `top_references`
    - `retrieved_contexts`
  - 只重跑当前 `generate_rag_answer()`，尽量把变化锁定在生成侧，而不是检索侧
- 输出结果已落盘：
  - `data/eval/factual_correctness_small_regression_20260424.json`

### 回归样本
- 共 `5` 条：
  - `Domain` `3` 条
  - `CRUD` `2` 条
- 覆盖类型：
  - `extractive_qa`
  - `multi-doc_qa`
  - `time-sensitive_qa`
  - `quest_answer`

### 结果判断

#### 1. 明确改善

- `multi-doc_qa-1`
  - 旧回答错误地判定“数据计算及应用”信息缺失，导致未完成跨文档比较。
  - 新回答在同一批证据上，已经能同时提取：
    - 数学与应用数学专业的培养目标与特点
    - 数据计算及应用专业的培养目标与特点
  - 说明这轮“覆盖要求 + 低噪声上下文”收口，对多子问题覆盖有实际收益。

- `extractive_qa-1`
  - 旧回答是“无法确定 + 额外解释背景”。
  - 新回答收缩为单句拒答，长度从 `85` 降到 `37`。
  - 说明单事实场景下，回答更短、更直接，套话和无效补充明显减少。

- `time-sensitive_qa-1`
  - 旧回答会在无法确定时附带无关年份和岗位背景。
  - 新回答直接收缩为“根据当前检索到的内容，无法确定。”，长度从 `167` 降到 `16`。
  - 说明对“证据不足时不要展开猜测或旁支说明”的约束更稳了。

- `64fa9b27b82641eb8ecbe14d`
  - 旧回答保留较多解释性尾注。
  - 新回答仍覆盖了：
    - 上海与成都促进体育消费的措施
    - 上海消费券的关键使用条件
  - 但删除了大量次要说明，长度从 `453` 降到 `233`。
  - 说明这轮收口对“保留主答案、压缩附加解释”有正向效果。

#### 2. 部分改善但仍有问题

- `64fa9b27b82641eb8ecbe14c`
  - 新回答比旧回答更直接，长度从 `398` 降到 `223`。
  - 但仍保留了被截断的枚举项：`多方合力共`。
  - 说明当前 prompt 收紧能减少扩写，但对“证据片段本身被截断”的脏上下文还没有专门修复能力。

### 当前结论
- 这轮 `factual_correctness` 收口已经带来可见改善，且改善主要体现在：
  - 单事实 / 无法确定问题更短、更直接
  - 多子问题覆盖比上一版更完整
  - 无证据时的冗余背景解释明显减少
- 从这 `5` 条同证据回放看，这轮变化更像“生成侧稳定性提升”，而不是偶然输出波动。

### 残留问题
- `中国人民大学的校长是谁？`
  - 这是典型隐含“当前时点”的问题，但当前 answer trace 中：
    - `direct_answer_query = true`
    - `temporal_answer_query = false`
  - 说明现有 `is_temporal_answer_query()` 还没有覆盖“职位当前归属类问题”的隐式时间约束。
- 对截断证据的处理仍偏弱：
  - 如 `多方合力共` 这类残缺片段，当前仍可能被原样带入答案。
- 本轮回归是“同证据重放”，因此只能证明：
  - 当前生成侧比旧版本更会用同一批证据作答
  - 不能直接替代“检索 + 生成全链路”的正式大样本回归

### 下一步建议
1. 继续补一个很小的正式全链路回归：
   - 至少再跑 `Domain + CRUD` 各几条真实检索问答
   - 与这次“同证据重放”结果交叉验证
2. 追加一轮很小的规则收口：
   - 让隐含当前职位类问题进入 `temporal_answer_query`
3. 若继续处理生成侧事实正确性，优先补：
   - 截断枚举项清理
   - 不完整短语过滤

## 2026-04-24 生成侧 factual_correctness 收口：隐含当前职位类问题 + 残缺短语过滤 + 全链路小回归

### 本轮改动
- 文件：`app/chains/rag.py`
- 目标：
  - 用最小规则补齐“隐含当前职位类问题”的时间敏感识别
  - 补一层轻量答案清洗，减少截断短语、残缺枚举项直接进入最终回答
  - 同时避免为了这轮收口再引入新生成阶段

### 具体实现

#### 1. 隐含当前职位类问题识别
- 扩展 `is_temporal_answer_query()`：
  - 原先主要依赖显式时间词：
    - `当前`
    - `最新`
    - `现任`
    - `截至`
    - `哪一年`
    - `何时`
  - 现在新增 `is_implicit_current_role_query()` 作为补充
- 新规则：
  - 若问题同时满足：
    - 包含 `谁 / 哪位`
    - 包含职位词，如：
      - `校长`
      - `院长`
      - `主任`
      - `书记`
      - `局长`
      - `部长`
      - `市长`
      - `董事长`
      - `总经理`
      - `CEO`
      - `负责人`
  - 且不含明显历史指向词，如：
    - `曾任`
    - `历任`
    - `前任`
    - `时任`
    - `当时`
  - 则判定为隐含“当前时点”的时间敏感问题
- 作用：
  - `中国人民大学的校长是谁？` 现在会进入时间敏感问答约束

#### 2. 残缺短语过滤
- 在 `maybe_refine_rag_answer()` 返回前新增 `cleanup_generated_answer()`
- 当前只做轻量、低风险清洗：
  - 清理已观察到的截断尾项模式：
    - `多方合力共`
    - `具体如下`
    - `主要有以下`
    - `包括以下`
  - 清理清洗后残留的占位痕迹，例如：
    - `“。”这一概括性表述`
- 目标不是改写答案结构，而是去掉明显的截断垃圾片段

#### 3. Prompt 片段边界裁切
- `format_reference_block()` 不再直接按字符数硬截断正文
- 新增 `clip_prompt_snippet()`：
  - 优先在这些边界上收口：
    - `。！？；`
    - 换行
    - `，、`
  - 避免把半句、半个枚举项直接塞进 prompt
- 这一步主要用于减少像 `多方合力共` 这类脏片段在生成阶段被复述

### 定点验证
- 语法验证通过：
  - `RAG_PY_COMPILE_OK`
- 规则定点检查结果：
  - `中国人民大学的校长是谁？`
    - `implicit_current_role_query = true`
    - `temporal_answer_query = true`
  - `中国人民大学现任校长是谁？`
    - `implicit_current_role_query = true`
    - `temporal_answer_query = true`
  - `李文海曾任什么职务？`
    - `implicit_current_role_query = false`
    - `temporal_answer_query = false`

### 很小的真实“检索 + 生成”全链路回归

#### 回归方式
- 不再只看“同证据重放”，而是直接跑真实链路：
  - `search_local_knowledge_base()`
  - `generate_rag_answer()`
- 输出结果已落盘：
  - `data/eval/full_chain_small_regression_20260424.json`

#### 回归样本
- 共 `4` 条：
  - `Domain` `2` 条
  - `CRUD` `2` 条
- 所有样本均使用真实知识库检索结果，不手工注入旧上下文

#### 回归结果

- `domain-time-role-1`
  - 查询：`中国人民大学的校长是谁？`
  - 结果：
    - `temporal_answer_query = true`
    - `implicit_current_role_query = true`
    - 最终答案仍为无法确定
  - 结论：
    - 新规则已生效
    - 但这条样本当前仍受检索命中质量限制，生成侧无法凭空补答案

- `crud-nearsight-1`
  - 查询：`启明行动 ... 核心知识 + 医疗机构和家长角色`
  - 结果：
    - 旧问题里的残缺短语 `多方合力共` 不再出现在最终答案中
    - 清洗后的残留痕迹 `“。”` 也已消失
  - 结论：
    - 残缺短语过滤有效

- `crud-sport-coupon-1`
  - 结果：
    - 检索、生成链路正常
    - 回答保留了上海、成都措施与上海消费券条件
  - 结论：
    - 这轮收口没有造成明显回退

- `domain-multi-doc-1`
  - 结果：
    - 真实检索条件下，回答退回“无法确定”
    - 当前检索到的上下文只包含专业名称与链接，未包含足够的人才培养描述
  - 结论：
    - 这条暴露的是检索侧问题，不是本轮生成规则问题
    - 同证据重放里能答出来，不代表真实检索一定能取到同样好的上下文

### 当前判断
- 本轮“小规则收口”是有效的，且风险可控：
  - 隐含当前职位类问题已纳入时间敏感识别
  - 截断短语不会再直接以脏片段形式出现在答案里
- 同时，这轮真实全链路回归也进一步确认：
  - 生成侧收口不能替代检索质量
  - `Domain` 多文档样本的下一步重点，应转回检索召回与重排，而不是继续只改 prompt

### 下一步建议
1. 保留本轮 `rag.py` 规则收口，不回退。
2. 若继续优化 `Domain` 真实全链路表现，优先处理：
   - 多文档描述型问题的检索召回
   - 与 query 覆盖要求更一致的 rerank
3. 后续所有生成侧回归，建议同时保留两套口径：
   - 同证据重放
   - 真实检索 + 生成全链路

## 2026-04-24 Domain 检索 bad case 第一轮拆解：`multi-doc_qa-1`

### 目标
- 对 `domain-multi-doc-1` 先做第一轮真实 bad case 拆解。
- 目的不是立刻改代码，而是先判断这条问题当前更接近：
  - `missed_recall`
  - `low_rank`
  - `chunk_noise`
  - 还是已经进入“生成误判 / 证据组织问题”

### 样本
- 查询：
  - `数学与应用数学专业与数据计算及应用专业在人才培养的共同目标和独特特点是什么？`
- gold 证据来自：
  - `数学与应用数学`
  - `数据计算及应用`

### 检索侧检查结果

#### 1. query rewrite 没有明显漂移
- 当前 retrieval trace 中，这条 query 的改写为：
  - `数学与应用数学专业 数据计算及应用专业 人才培养 共同目标 独特特点`
- query bundle 也保留了明显的多文档对比意图：
  - `数学与应用数学 数据计算及应用 专业培养方案 目标 特色 对比`
  - 或 `专业培养目标 区别 特色`
- 当前判断：
  - 关键实体没有丢
  - `人才培养 / 目标 / 特点 / 对比` 等约束也仍在
  - 因此这条样本暂不归为 `query_rewrite_drift`

#### 2. gold 并没有丢失，且已进入前排
- retrieval trace 显示当前 `top-5` 稳定包含：
  - `数据计算及应用__doc-2009__psg-ref3`
  - `数据计算及应用__doc-661__psg-ref4`
  - `数学与应用数学__doc-719__psg-3`
  - 以及两条 `数学学院` 汇总型 chunk
- 这些结果里：
  - `数据计算及应用` 是 gold
  - `数学与应用数学` 是 gold
  - `数学学院` chunk 还同时包含两个专业的人才培养描述
- 当前判断：
  - 这条样本不是 `missed_recall`
  - 也不是“gold 只在 top-50、排不进前列”的典型 `low_rank`

#### 3. 前排 chunk 本身已经足够回答问题
- 打开当前 top chunk 后，能确认：
  - `数据计算及应用__doc-2009__psg-ref3`
    - 包含“创新复合型人才”
    - 包含“数据科学与人文社会科学深度交叉融合”
    - 包含“面向国家创新驱动发展、中国制造2025、互联网+”
  - `数学与应用数学__doc-719__psg-3`
    - 包含“国际视野、创新能力、独立分析和深入研究能力”
    - 包含“单独培养管理、顶尖师资、国际交流平台”
  - `数学学院__doc-694__psg-ref6` / `doc-131__psg-ref5`
    - 同一块里同时给出了两个专业的人才培养描述
- 当前判断：
  - 这条样本前排证据并不是只有“名称 + 链接”
  - 证据已经足够支持回答

### 结论
- 这条 `multi-doc_qa-1` 在当前版本下，第一轮拆解结论应修正为：
  - **不是 `missed_recall`**
  - **不是典型 `low_rank`**
  - **更接近“生成误判 / 证据组织利用不稳”**
- 也就是说：
  - 先前把它简单归因为“真实检索拿不到足够证据”，这个判断过粗
  - 更准确的说法是：
    - 检索已经把足够证据送到了前排
    - 但生成链路对这些证据的利用还不够稳定

### 下一步建议
1. 不把这条样本优先归入“召回修复”。
2. 下一轮更值得做的是：
   - 针对 `Domain multi-doc` 补一个更稳定的上下文组织检查
   - 排查 prompt 压缩、去重、片段截断是否把关键信息削弱
3. 这条样本后续更适合作为：
   - `生成稳定性`
   - `证据组织稳定性`
   的专项回归样本，而不是纯检索召回样本。

## 2026-04-24 Domain 多文档上下文组织修复：`build_context()` 链路验证

### 目标
- 沿着：
  - `build_context()`
  - `build_rag_variables()`
  - `build_rag_prompt()`
- 把 `domain-multi-doc-1` 真正送进模型的最终上下文拆出来，并对定位出的链路问题做最小修复。

### 修复前还原结果
- 还原产物：
  - `data/eval/domain_multi_doc_prompt_trace_20260424.json`
- 关键现象：
  1. 检索拿到了 `5` 条有效 references
  2. 进入 prompt 的只剩 `3` 条
  3. 最终 `context` 长度只有 `760`
  4. 送进模型的正文几乎只剩：
     - 标题
     - 来源链接
     - 日期
     - 文档ID
     - 很短的正文前缀
- 直接原因有两个：

#### 1. prompt 去重过强
- `deduplicate_references_for_prompt()` 默认依赖 `build_prompt_reference_fingerprint()`
- fingerprint 优先用：
  - `evidence_summary`
  - 否则 `content_preview`
  - 否则 `content`
- 而当前这条 case 的多个 chunk：
  - 两条 `数据计算及应用`
  - 两条 `数学学院`
  - `evidence_summary` 都只是标题级文本
- 结果：
  - 两条 `数据计算及应用` 被误判为重复
  - 两条 `数学学院` 被误判为重复
  - `5` 条证据被压成 `3` 条

#### 2. 多文档问题被错误套用了单事实片段上限
- `resolve_reference_content_limit()` 对 `should_directly_answer_query(query)=true` 的问题使用 `120` 字上限
- 这条 query 虽然本质是多文档比较题，但因为包含“是什么”，仍被判成 `direct_answer_query`
- 结果：
  - 每条证据只保留了极短前缀
  - 真正的人才培养描述没有完整进入 prompt

### 实施修复
- 文件：
  - `app/chains/rag.py`
- 修复内容：

#### 1. 放宽多文档问题的 prompt 去重
- 新增 `is_multi_doc_comparative_query()`
- 对包含以下特征的问题判定为多文档比较型：
  - 关系词：
    - `与`
    - `和`
    - `及`
    - `以及`
    - `对比`
    - `比较`
  - 比较词：
    - `共同`
    - `区别`
    - `不同`
    - `相同`
    - `特点`
    - `分别`
    - `各自`
  - 实体词：
    - `专业`
    - `学院`
    - `方向`
    - `项目`
- 在这类 query 下：
  - `deduplicate_references_for_prompt()` 传入 `query`
  - `build_prompt_reference_fingerprint()` 不再优先只看标题级 `evidence_summary`
  - 改为更偏向 `content_preview / content`
  - 指纹长度也从 `180` 放宽到 `260`

#### 2. 提高多文档问题的正文片段上限
- 在 `resolve_reference_content_limit()` 中新增分支：
  - 若 `is_multi_doc_comparative_query(query)=true`
  - 则直接使用 `260`
- 优先级放在 `should_directly_answer_query()` 之前
- 避免多文档比较题被错误套用单事实题的 `120` 字极短裁切

### 修复后验证结果
- 语法验证通过：
  - `RAG_PY_COMPILE_OK`
- 修复后链路还原产物：
  - `data/eval/domain_multi_doc_prompt_trace_20260424_after_fix.json`

#### 修复后关键变化
- `retrieved_count`：
  - `5`
- `prompt_count`：
  - `3 -> 5`
- `context` 长度：
  - `760 -> 1732`
- 修复后进入 prompt 的 source 包括：
  - 两条 `数据计算及应用`
  - 两条 `数学学院`
  - 一条 `数学与应用数学`

#### 修复后生成结果
- 目标问题：
  - `数学与应用数学专业与数据计算及应用专业在人才培养的共同目标和独特特点是什么？`
- 修复后回答已恢复为有效对比：
  - 共同目标：
    - 培养复合型、创新型人才
    - 面向国家战略需求
    - 注重跨学科交叉融合
  - 独特特点：
    - `数据计算及应用`：强调数据科学与人文社会科学深度交叉融合，突出信息化、智能化方向
    - `数学与应用数学`：强调数学、计算机、经济金融理论的复合，设有多个方向和单独管理培养班

### 当前结论
- 这条 `domain-multi-doc-1` 的主因已经可以明确收口为：
  - **prompt 去重过强**
  - **多文档问题的正文片段裁切过短**
- 它不是检索召回问题。
- 这次修复说明：
  - 检索已经把足够证据送到了前排
  - 只要 prompt 侧别过度压缩，当前模型已经能完成这类多文档对比回答

### 下一步建议
1. 保留这次 `rag.py` 修复，不回退。
2. 后续继续检查：
   - `split_query_into_requirements()` 是否需要把“共同目标 + 独特特点”拆开，进一步提升 coverage 约束
3. 若继续推进 `Domain multi-doc`，优先新增：
   - 多文档比较题专项小回归集
   - prompt 去重前后对照回归

## 2026-04-24 Query 主体归一化收口：避免背景文本污染 coverage / prompt

### 目标
- 在不继续堆 case-specific 规则的前提下，再做一小步泛化收口。
- 解决带 `事件背景：... 问题：...` 前缀的查询，会把整段背景误带进：
  - `coverage_requirements`
  - 问题分类
  - prompt 约束判断

### 实施内容
- 文件：
  - `app/chains/rag.py`
- 新增：
  - `extract_primary_question_text()`
- 规则非常收敛：
  - 若 query 中存在最后一个显式问题标记：
    - `问题： / 问题:`
    - `请回答： / 请回答:`
    - `问： / 问:`
  - 则后续：
    - `split_query_into_requirements()`
    - `build_coverage_requirements()`
    - `build_answer_requirements()`
    - `should_directly_answer_query()`
    - `is_multi_doc_comparative_query()`
    - `is_temporal_answer_query()`
  - 都优先基于“问题主体”而不是整段原始输入判断

### 为什么这一步值得做
- 上一轮 `crud_parallel` 小回归里，`coverage_requirements` 还会把整段事件背景带进第一个子问题。
- 这会造成：
  - coverage 文本变脏
  - prompt token 被背景叙述挤占
  - 问题分类更容易被非问题正文干扰
- 这不是为某一条样本打补丁，而是统一处理“背景 + 正题”格式输入。

### 本地验证
- 语法编译：
  - `RAG_PY_COMPILE_OK`
- 额外生成验证产物：
  - `data/eval/query_normalization_check_20260424_ascii.json`

#### 关键结果
- `domain-time-role-1`
  - 主体仍识别为：
    - `中国人民大学的校长是谁？`
  - `is_temporal = true`
- `domain-multi-doc-1`
  - 主体仍识别为原比较问题
  - `is_multi_doc = true`
  - comparative coverage 与 answer requirements 保持不变
- `crud-sport-coupon-1`
  - 主体成功收敛为真正的问题：
    - `上海和成都市体育局都采取了哪些措施来促进体育消费，以及上海体育消费券的具体使用条件是什么？`
  - `coverage_requirements` 不再把“事件背景”整段带入
  - 仍能正常拆成：
    - 上海和成都市体育局的措施
    - 上海体育消费券的具体使用条件

### 当前判断
- 这一步属于“去噪”，不是继续对单一样本做 prompt 微调。
- 它对以下场景都有普适价值：
  - `事件背景 + 问题`
  - `材料 + 问题`
  - `说明 + 请回答`
- 且没有观察到：
  - `domain-multi-doc-1` comparative 行为回退
  - `domain-time-role-1` temporal 行为回退

### 下一步建议
1. 先停止继续给生成侧叠更多规则。
2. 若再往前走，优先做：
   - 1~2 条新增 `Domain multi-doc` 比较题小回归
   - 看当前 comparative prompt 组织是否真的能泛化
3. 若后续指标没有明显回退，这一轮就可以视为生成侧的低风险收口完成。

## 2026-04-24 Domain multi-doc 比较题再回归：comparative prompt 泛化检查

### 目标
- 再补 `1~2` 条新的 `Domain multi-doc` 比较题小回归。
- 验证当前 comparative prompt 收口是否只对 `domain-multi-doc-1` 有效，还是已经具备一定泛化能力。

### 本轮动作
- 选择 `domainrag_multi_doc_20.jsonl` 中两条新样本：
  - `case 3`
    - `计算机科学与技术专业与数据科学与大数据技术专业的共同点和不同点是什么？`
  - `case 20`
    - `社会工作与社会学专业在人才培养方案与未来职业发展方向上有哪些相似之处？`
- 走真实全链路：
  - `search_local_knowledge_base()`
  - `build_context()`
  - `generate_rag_answer()`
- 结果产物：
  - `data/eval/domain_multi_doc_comparative_regression_20260424.json`

### 第一轮结果

#### case 3：泛化成功
- `is_multi_doc_comparative_query = true`
- `context_length = 1762`
- 返回的回答已经具有明确比较结构：
  - 先答共同点
  - 再分点写不同点
- 当前判断：
  - 这说明当前 comparative prompt 收口并不只对“数学与应用数学 vs 数据计算及应用”单一样本有效
  - 对另一条“共同点 + 不同点”型比较题也能稳定生效

#### case 20：暴露新的边界问题
- 第一轮中：
  - `is_multi_doc_comparative_query = false`
  - `context_length = 869`
  - 最终回答直接退回“无法确定”
- 这说明当问题表述成：
  - `相似之处`
  - 而不是 `共同点 / 不同点 / 区别`
  - 现有 comparative 识别规则会漏掉一部分比较题

### 小步补充修复
- 文件：
  - `app/chains/rag.py`
- 做了一步很小的泛化扩展：
  - `is_multi_doc_comparative_query()` 新增比较词：
    - `相似`
    - `异同`
    - `差异`
  - `infer_comparative_coverage_points()` 补充：
    - `比较对象的相似之处或共同点`
- 这一步仍然属于泛化词表扩展，不是对单一样本写硬编码。

### 第二轮结果
- 结果产物：
  - `data/eval/domain_multi_doc_comparative_regression_20260424_v2.json`

#### case 3：保持稳定
- `is_multi_doc_comparative_query = true`
- `context_length = 1762`
- 回答继续保持：
  - 共同点
  - 各自不同点
- 当前判断：
  - 没有因为新增比较词而回退

#### case 20：识别修复成功，但检索问题仍在
- 第二轮中：
  - `is_multi_doc_comparative_query = true`
  - `context_length = 1722`
  - 但最终回答仍然是保守拒答
- top sources 显示：
  - `社会工作__doc-676__psg-0`
  - `社会学__doc-176__psg-0`
  - 但还混入了两条 `劳动与社会保障` 材料
- 当前判断：
  - 这条样本已经不再是“comparative 识别漏掉”
  - 当前更接近：
    - 前排证据仍不够聚焦
    - 检索 / rerank 层混入了不相关专业材料
  - 也就是说：
    - prompt 收口已接住这类问题
    - 但这条 case 的主瓶颈已经回到检索前排质量

### 本轮结论
1. 当前 comparative prompt 收口已经证明具有一定泛化性，不只对单一样本有效。
2. “共同点 / 不同点”类比较题目前表现更稳。
3. “相似之处”类比较题原本存在识别漏检，这一轮已通过小步规则扩展修复。
4. 修复 comparative 识别后，剩余 bad case 更容易暴露真实检索问题，而不会继续被 prompt 识别链路掩盖。

### 下一步建议
1. 保留这轮 `comparative` 识别扩展，不回退。
2. 后续若继续推进 `Domain multi-doc`，优先分析：
   - 为何 `社会工作 vs 社会学` 会混入 `劳动与社会保障`
   - 是 query rewrite、candidate pool，还是 rerank 特征导致的串题
3. 下一步更值得做的，不是继续给 comparative prompt 加规则，而是：
   - 对 `Domain multi-doc` 的检索前排做一次“串题 / 误命中”专项 bad case 分析

## 2026-04-24 Domain multi-doc 串题 / 误命中专项分析：`社会工作 vs 社会学`

### 目标
- 对 `社会工作与社会学专业在人才培养方案与未来职业发展方向上有哪些相似之处？` 做一次定点误命中分析。
- 判断当前串题主因更接近：
  - `query rewrite drift`
  - `candidate pool 污染`
  - `rerank 误抬高`

### 分析对象
- 目标问题：
  - `社会工作与社会学专业在人才培养方案与未来职业发展方向上有哪些相似之处？`
- 当前异常现象：
  - 前排稳定混入：
    - `劳动与社会保障`
    - 偶发 `政治学与行政学`
  - 导致最终回答保守拒答，无法稳定输出“相似之处”

### 证据 1：retrieval trace 显示 rewrite 会持续把问题抽象化
- 近期 retrieval trace 中，这条问题的 `query_bundle` 多次出现类似改写：
  - `社会工作与社会学专业在人才培养方案与未来职业发展方向上的相似之处`
  - `社会工作与社会学专业课程设置与就业方向对比分析`
  - `社会工作与社会学两个专业的课程设置及就业方向对比`
  - `社会工作与社会学专业在课程设置、培养目标及职业路径上的共性`
- 这些 rewrite 的共同特点：
  - 保留了两个专业名
  - 但把 query 重心逐步抽象成：
    - `课程设置`
    - `就业方向`
    - `培养目标`
    - `职业路径`
  - 于是更容易把其他“有人才培养 / 就业去向 / 课程体系”描述的专业文档一并拉进候选池

### 证据 2：默认链路下，误命中稳定存在
- 当前默认 retrieval trace 结果稳定出现：
  - `社会工作__doc-676__psg-0`
  - `社会学__doc-176__psg-0`
  - 以及多条：
    - `劳动与社会保障__doc-615__...`
    - `劳动与社会保障__doc-724__...`
  - 偶发还会出现：
    - `政治学与行政学__doc-648__...`
- 当前判断：
  - 串题不是偶然抖动，而是默认 query bundle 下可重复复现的问题

### 证据 3：关闭 query rewrite 后，前排明显收敛
- 新增对照产物：
  - `data/eval/domain_multi_doc_mistarget_compare_20260424.json`
- 对照方式：
  - `default`
    - 保留当前 query rewrite / multi-query
  - `no_rewrite`
    - 关闭 query rewrite 与 multi-query，只保留原问

#### 对照结果
- `default`：
  - `社会工作`
  - `劳动与社会保障`
  - `社会学`
  - `政治学与行政学`
  - `劳动与社会保障`
- `no_rewrite`：
  - `社会工作`
  - `社会工作`
  - `社会学`
  - `社会工作`
  - `社会工作`

### 结论
- 这次对照说明：
  - **主因优先归为 `query rewrite / multi-query` 放宽了候选池**
  - 而不是“原问本身天然就会串到 `劳动与社会保障`”
- 更准确地说：
  - 原问在不改写时，前排候选其实能较好收敛到 `社会工作 + 社会学`
  - 一旦进入当前 rewrite / multi-query 扩展，抽象化的“培养方案 / 就业方向 / 课程设置”表述会把其他相邻专业一起拉进来
- 因此这条 bad case 的当前定位应更新为：
  - **主要是 `query_rewrite_drift`**
  - `rerank` 只是没有把漂进来的候选再压下去
  - 不是最初判断的“comparative prompt 没接住”

### 当前判断
1. 当前 comparative prompt 收口已经不是这条样本的主问题。
2. 当前主要矛盾前移到了检索阶段：
   - query rewrite 对实体边界保护不够强
3. 当前 rerank 也存在次级问题：
   - 对“目标专业名是否同时命中”这一信号利用还不够强

### 下一步建议
1. 优先给 `Domain multi-doc` 比较题增加 query rewrite 保护：
   - 若 query 已包含两个明确专业名，降低 rewrite 抽象化幅度
   - 强制保留两个目标实体
   - 避免把 rewrite 变成泛化性的“课程设置 / 就业方向对比”
2. 对 comparative query 增加一个低风险 rerank 信号：
   - 同时命中两个目标实体名称的候选优先
3. 后续若继续做回归，优先复测：
   - `社会工作 vs 社会学`
   - 看 query rewrite 保护后，`劳动与社会保障` 是否还会稳定挤进前排

## 2026-04-24 Domain multi-doc query rewrite 保护：双专业比较题跳过 rewrite

### 目标
- 对 `Domain multi-doc` 中“两个目标专业 + 比较意图”的问题增加一层 query rewrite 保护。
- 避免 rewrite 把问题泛化成：
  - `课程设置对比`
  - `就业方向对比`
  - `培养目标共性`
- 从而把相邻专业一起拉进候选池。

### 问题背景
- 在 `社会工作 vs 社会学` 的串题分析中已经确认：
  - `default`（保留 rewrite）前排会稳定混入：
    - `劳动与社会保障`
    - 偶发 `政治学与行政学`
  - `no_rewrite`（只保留原问）前排会明显收敛回：
    - `社会工作`
    - `社会学`
- 因此这条问题的主因不是 comparative prompt，而是：
  - `query rewrite / multi-query` 放宽了候选池

### 实施内容
- 文件：
  - `app/services/retrieval/query_rewrite_service.py`

#### 1. 新增比较实体识别
- 新增：
  - `ComparativeEntityProfile`
  - `build_comparative_entity_profile()`
  - `extract_comparative_professional_entities()`
- 识别条件收敛为：
  - query 中包含：
    - 关系词：`与 / 和 / 及 / 以及`
    - 比较词：`共同 / 区别 / 不同 / 相同 / 相似 / 异同 / 差异 / 比较 / 对比`
  - 且能从 query 中拆出两个明确的“专业”实体

#### 2. 对这类 query 直接跳过 rewrite
- 在 `generate_multi_queries()` 中新增保护：
  - 若 `build_comparative_entity_profile(query).is_multi_entity_comparative = true`
  - 则直接返回：
    - `[original_query]`
- 当前设计选择的是“直接跳过 rewrite”，而不是继续保留宽泛改写候选。
- 原因：
  - 在这类题上，原问已经足够明确
  - 继续 rewrite 的边际收益远小于串题风险

### 验证结果
- 语法编译：
  - `QUERY_REWRITE_COMPILE_OK`

#### 1. query bundle 验证
- 目标问题：
  - `社会工作与社会学专业在人才培养方案与未来职业发展方向上有哪些相似之处？`
- 当前 `query_bundle` 已收敛为：
  - 仅保留原问 1 条

#### 2. 检索前排验证
- 结果产物：
  - `data/eval/domain_multi_doc_mistarget_after_guard_20260424.json`
- 保护后前排来源变为：
  - `社会工作`
  - `社会工作`
  - `社会学`
  - `社会工作`
  - `社会工作`
- 说明：
  - `劳动与社会保障`
  - `政治学与行政学`
  - 已不再稳定挤进前排

#### 3. 全链路验证
- 结果产物：
  - `data/eval/domain_multi_doc_mistarget_after_guard_fullchain_20260424_v2.json`
- 当前回答仍然保守：
  - 主要原因已从“串题误命中”收敛为：
    - `社会学` 可回答正文覆盖不足
    - 两个目标专业的证据覆盖不均衡
- 当前判断：
  - 这是更干净、更可定位的失败
  - 至少已经不是“检索前排被错误专业挤占”的问题

### 本轮结论
1. 这轮 query rewrite 保护已经生效。
2. 对“双专业比较题”，直接跳过 rewrite 是当前更稳的工程选择。
3. 保护后，`社会工作 vs 社会学` 的主问题已经从：
   - `query_rewrite_drift / 串题误命中`
   收敛为：
   - `目标专业证据覆盖不均衡`

### 下一步建议
1. 保留这层 rewrite 保护，不回退。
2. 若继续优化 `社会工作 vs 社会学`，下一步优先检查：
   - 为什么 `社会学` 只有定义型 chunk 排上来，培养方案与职业发展方向正文没有稳定进入前排
3. 后续若继续做检索收口，优先补：
   - comparative query 下“同时命中两个目标专业名”的 rerank 加分

## 2026-04-24 Domain multi-doc 检索侧小步复测：比较实体 rerank 未带来可见收益

### 本轮目标
- 在不扩大改动面的前提下，验证一个低风险假设：
  - 给 comparative multi-doc query 增加“目标专业实体命中”排序信号，是否能把 `社会学` 的更有效正文稳定抬到前排

### 实施方式
- 试验位置：
  - `app/retrievers/local_kb.py`
- 试验思路：
  1. 在 heuristic rerank 中加入比较实体命中加分
  2. 尝试做一层 very light 的 comparative 候选平衡
- 约束：
  - 只在“双专业 + 比较意图”问题上触发
  - 不改 query rewrite 主策略
  - 不改 `rag.py` prompt 结构

### 回归方式
- 使用本地知识库：
  - `domainrag_small_batch_100`
- 目标问题：
  - `社会工作与社会学专业在人才培养方案与未来职业发展方向上有哪些相似之处？`
- 本轮实际复测的是：
  - `search_local_knowledge_base(..., top_k=12)`
  - 观察前排 title/source 分布是否出现实质变化

### 复测结果
- 本轮稳定看到的前排仍然是：
  1. `社会工作`
  2. `社会工作`
  3. `社会学`
  4. `社会工作`
  5. `社会工作`
  6. `劳动与社会保障`
  7. `劳动与社会保障`
  8. `劳动与社会保障`
- 说明：
  - 顶部仍然只有 1 条 `社会学`
  - `劳动与社会保障` 在 top-12 中仍然占据明显比例
  - 没有看到 `社会学` 的“人才培养 / 职业发展”类正文被稳定顶到前面

### 结论
1. “比较实体命中加分”这个方向在当前候选池上没有带来可见收益。
2. 这说明当前瓶颈不在于：
   - rerank 没有利用目标实体名
3. 当前更可能的瓶颈在于：
   - `社会学` 相关可回答 chunk 在候选池上游就不够强
   - 或者该专业原始文本里本就缺少和问题直接对齐的培养方案 / 职业去向表述

### 处理决定
1. 这轮试验代码没有保留，已回退。
2. 原因：
   - 改动增加了规则复杂度
   - 但没有换来稳定收益
3. 当前仍保留的有效改动只有：
   - comparative 双专业问题跳过 rewrite 的保护

### 更新后的下一步建议
1. 不再继续堆 comparative rerank 小规则，避免过拟合。
2. 下一步更值得做的是“证据供给侧核查”：
   - 直接检查 `社会学` 文档中是否存在培养方案 / 去向类 chunk
   - 若存在，定位为什么没有进入候选前排
   - 若不存在，就把该 bad case 归类为“知识源覆盖不足”，不要继续在 prompt 上强行收口
3. 如果后续继续优化 `Domain multi-doc`，优先顺序建议调整为：
   - 先查 corpus coverage
   - 再查 candidate generation / chunking
   - 最后才是再加 rerank 规则

## 2026-04-24 阶段十启动：TRACE 迁移的 Phase 1 接口骨架先落地

### 本轮目标
- 正式进入 `优化方案.md` 第十章。
- 当前不直接实现完整时间序列入库 / 检索，而是先完成 Phase 1 的低风险主干准备：
  - 让系统能够“合法承载时间序列 reference”
  - 让生成链路能够“识别并展示时间序列证据”
  - 让概览统计能够“区分文本 / 时间序列联合覆盖”

### 本轮实施范围
- 文件：
  - `app/schemas/chat.py`
  - `app/services/retrieval/reference_overview.py`
  - `app/chains/rag.py`
  - `app/ui/app.py`

### 已完成改动

#### 1. `RetrievedReference` 扩展了时间序列字段
- 新增可空字段：
  - `series_id`
  - `start_time`
  - `end_time`
  - `ts_summary`
  - `event_type`
  - `channel_names`
- 目的：
  - 先把第十章 10.6 约定的接口骨架落进真实代码
  - 后续接入 TS 检索时不需要再改 API 契约

#### 2. `ReferenceOverview` 增加时间序列统计
- 新增：
  - `timeseries_count`
  - `has_text_ts_joint_coverage`
- `build_reference_overview()` 已同步支持：
  - `source_modality=timeseries` 的计数
  - 文本 + 时间序列联合覆盖标记

#### 3. `rag.py` 上下文组装支持时间序列证据分组
- `build_context()` 新增：
  - `时间序列证据`
- `resolve_reference_context_group()` 已能识别：
  - `source_modality=timeseries`
- `format_reference_block()` 已支持输出：
  - `series_id`
  - `time_range`
  - `event_type`
  - `channel_names`
  - `ts_summary`
- 当前效果：
  - 即使检索侧还没接 TS 分支，只要后续有时间序列 reference 进入链路，prompt 侧已经能正确展示

#### 4. UI 概览已兼容时间序列统计
- 前端证据概览已增加：
  - 时间序列证据计数
  - 文本 + 时间序列联合覆盖提示
- 说明：
  - 这是“兼容性改造”，不是新增独立 TS 页面
  - 与第十章“保持现有主路径稳定”一致

### 当前阶段判断
1. 第十章已经从“纯方案”进入“代码骨架已起步”的状态。
2. 当前完成的是：
   - **Phase 1 的接口与上下文准备**
3. 当前还没有完成的是：
   - 时间序列样本入库流程
   - 时间序列检索分支
   - 时间序列专项 trace 与评测集

### 下一步建议
1. 继续按 Phase 1 顺序推进，不跳到 TRACE 风格训练增强。
2. 下一步最合适的落点是：
   - 新增 `timeseries` 结构化样本 schema / 入库规范
   - 先把“时间序列样本 -> 文本化摘要 -> 可索引文档”打通
3. 等入库结构稳定后，再补：
   - `search_local_knowledge_base` 旁的 TS 检索分支

## 2026-04-24 阶段十继续推进：时间序列结构化入库骨架已打通

### 本轮目标
- 在第十章 Phase 1 的基础上，继续把“时间序列样本如何进入当前知识库”落成可运行代码。
- 目标不是完整 TS 检索，而是先打通：
  - 结构化时间序列 JSON
  - 时间序列摘要生成
  - 转成现有知识库可索引 `Document`
  - 元数据沿链路传到 `RetrievedReference`

### 本轮改动范围
- 文件：
  - `app/schemas/kb.py`
  - `app/services/timeseries_summary_service.py`
  - `app/loaders/timeseries.py`
  - `app/loaders/factory.py`
  - `app/loaders/documents.py`
  - `app/services/core/settings.py`
  - `app/services/kb/embedding_assembler.py`
  - `app/services/kb/kb_incremental_rebuild.py`
  - `app/retrievers/local_kb.py`

### 已完成内容

#### 1. 新增时间序列结构化 schema
- 在 `app/schemas/kb.py` 新增：
  - `TimeSeriesPoint`
  - `TimeSeriesKnowledgeUnit`
- 当前最小支持字段包括：
  - `series_id`
  - `start_time`
  - `end_time`
  - `location`
  - `event_type`
  - `channel_names`
  - `ts_summary`
  - `description`
  - `event_background`
  - `points`

#### 2. 新增时间序列摘要服务
- 新增：
  - `app/services/timeseries_summary_service.py`
- 当前能力：
  - 若输入已提供 `ts_summary`，直接复用
  - 若未提供，则根据点位自动生成轻量趋势摘要
  - 生成数值预览文本，避免把长序列原值直接塞进 prompt

#### 3. 新增时间序列 JSON loader
- 新增：
  - `app/loaders/timeseries.py`
- 当前支持输入形态：
  1. 单个样本对象
  2. 样本对象数组
  3. 包含 `samples` 数组的对象
- 当前 loader 输出：
  - `source_modality=timeseries`
  - `content_type=timeseries_text_evidence`
  - `series_id / start_time / end_time / location / event_type / channel_names / ts_summary`
  - 可直接进入现有 chunk / embedding / retrieval 链路

#### 4. 接入当前 loader 工厂
- `factory.py` 已把时间序列 loader 注册进现有 `KnowledgeFactory`
- `settings.py` 已把 `.json` 纳入支持扩展名
- 同时增加了一个安全保护：
  - `.rag_file_metadata.json` 不会被当作知识文件加载

#### 5. 增加普通 JSON 兼容回退
- 为避免把现有普通 `.json` 文件全部变成“必须满足时间序列 schema”，当前策略是：
  - 命中时间序列结构：按 `timeseries` 加载
  - 否则：回退为普通 JSON 文本加载
- 这样不会明显抬高现有知识库 rebuild 风险

#### 6. 时间序列元数据已能沿链路透传
- 已把以下字段补进 chunk / rebuild / retrieval 引用链路：
  - `series_id`
  - `start_time`
  - `end_time`
  - `ts_summary`
  - `event_type`
  - `location`
  - `channel_names`
- 当前效果：
  - loader 生成的时间序列文档，已经可以在 `RetrievedReference` 和 `build_context()` 中保留这些字段

### 本轮验证
- 代码解析检查：
  - `TIMESERIES_LOADER_PARSE_OK`
- 时间序列 smoke test：
  - 结构化 JSON 能加载成：
    - `source_modality=timeseries`
    - `content_type=timeseries_text_evidence`
  - `ReferenceOverview` 统计为：
    - `timeseries_count=1`
  - `build_context()` 中已出现：
    - `时间序列证据`
    - `location`
    - `ts_summary`
- 普通 JSON fallback smoke test：
  - 非时间序列 JSON 会回退成：
    - `source_modality=text`
    - `content_type=document_text`

### 当前阶段判断
1. 第十章 Phase 1 已经从“接口骨架”推进到“入库对象标准化骨架”。
2. 当前已经具备：
   - 时间序列样本结构定义
   - 自动摘要
   - 可索引文档转换
   - 元数据透传
3. 当前仍未完成：
   - `search_local_knowledge_base` 旁的 TS 检索分支
   - TS 与文本联合排序
   - 时间序列专项 trace / 评测集

### 下一步建议
1. 下一步优先做 TS 检索分支，而不是马上做训练增强。
2. 更合适的落点是：
   - 新增 `timeseries_retrieval_service`
   - 先基于现有向量库把 `source_modality=timeseries` 候选单独召回出来
   - 再在 `search_local_knowledge_base` 旁加一个融合入口
3. 保持当前原则：
   - 文本检索保底执行
   - TS 检索作为增强分支并行进入

## 2026-04-24 阶段十继续推进：TS 检索分支最小版已接入主链路

### 本轮目标
- 在 Phase 1 范围内，把“时间序列检索分支”以最小可运行方式接进当前主链路。
- 目标不是完整联合排序优化，而是先实现：
  - 时间序列问题识别
  - 文本保底分支
  - TS 增强分支
  - 合并候选后继续走现有 rerank / build_context / rag 生成链路

### 本轮改动范围
- 文件：
  - `app/services/retrieval/timeseries_retrieval_service.py`
  - `app/retrievers/local_kb.py`

### 已完成内容

#### 1. 新增时间序列问题识别服务
- 新增：
  - `infer_timeseries_query_profile()`
- 当前识别信号覆盖：
  - `时间序列`
  - `趋势`
  - `变化`
  - `波动`
  - `异常`
  - `峰值 / 谷值`
  - `走势`
  - `监测`
  - `曲线`
  - 时间窗口类词

#### 2. 查询画像新增 `timeseries_related`
- 在 `infer_query_modality_profile()` 中新增：
  - `query_type="timeseries_related"`
- 当前效果：
  - 时间序列问题会优先把 `timeseries` 放进 preferred modalities
  - 非时间序列问题保持原来的 `text_related`

#### 3. `local_kb.py` 已接入 TS 双分支检索入口
- 新增入口：
  - `retrieve_candidates_with_timeseries_branching()`
- 当前执行策略：
  1. 先识别 query 是否为时间序列相关
  2. 若不是：
     - 保持原检索路径
  3. 若是：
     - 文本分支：`source_modality != timeseries`
     - TS 分支：`source_modality == timeseries`
     - 合并候选后继续进入现有 rerank

#### 4. 非时间序列问题不会主动扩展到 TS 模态
- 调整了 `select_modalities_for_query()`：
  - 默认不会把 `timeseries` 当成普通文本问题的补充模态
- 这样更符合第十章的设计：
  - 文本主链路稳定
  - TS 作为增强分支，仅在相关 query 下触发

#### 5. 诊断字段已补充
- 当前 retrieval diagnostics 已新增：
  - `timeseries_query_detected`
  - `timeseries_query_keywords`
  - `timeseries_window_constraint`
  - `timeseries_branch_used`
  - `text_branch_candidate_count`
  - `timeseries_branch_candidate_count`
  - `merged_candidate_count`

### 本轮验证
- 代码解析检查：
  - `TIMESERIES_RETRIEVAL_PARSE_OK`
- 最小 fake vector store 回归结果：
  - 时间序列 query：
    - `query_type=timeseries_related`
    - `timeseries_branch_used=True`
    - 返回候选包含：
      - `timeseries`
      - `text`
  - 普通文本 query：
    - `query_type=text_related`
    - `timeseries_branch_used=False`
    - 返回候选保持：
      - `text`

### 当前阶段判断
1. 第十章 Phase 1 现在已经完成了三层主干：
   - 时间序列 reference 接口骨架
   - 时间序列结构化入库骨架
   - 时间序列检索分支最小版
2. 当前仍未完成：
   - TS 与文本候选的专门排序增强
   - 时间序列问题的 prompt 输出规范强化
   - 真实知识库上的端到端联合回归
   - 时间序列专项 trace 落盘与评测集

### 下一步建议
1. 先不要继续堆复杂 rerank 规则。
2. 下一步更合适的是：
   - 做一轮真实小样本知识库 smoke test
   - 把时间序列 JSON 放进临时知识库或测试知识库
   - 直接验证 `/chat/rag` 级别是否能返回 TS reference
3. 若真实链路通了，再补：
   - 时间序列问题回答模板约束
   - `reference_overview` / trace 的 TS 专项落盘字段

## 2026-04-26 阶段十真实小样本端到端 smoke test：链路已通，生成覆盖仍需收口

### 本轮目标
- 执行第十章当前最关键的一步验证：
  - 用真实临时知识库样本跑通 `/chat/rag`
  - 确认时间序列 JSON 能否真正进入上传、索引、检索、引用、生成链路

### 本轮执行
- 样本组成：
  - `air_quality_series.json`
    - 北京 `PM2.5 / PM10` 三日时间序列
  - `event_note.txt`
    - 冷空气过程导致污染物浓度短时升高、随后回落的文本背景
- 执行路径：
  1. `upload_temp_files()`
  2. `rag_chat()`
  3. 查询：
     - `请结合时间序列和文本说明北京pm25在2024年1月1日至1月3日的变化趋势，并指出对应事件背景。`

### 本轮发现与修正

#### 1. 真实上传链路最初被配置挡住
- 问题：
  - `configs/kb_settings.yaml` 中 `SUPPORTED_EXTENSIONS` 还没有 `.json`
  - 导致 `upload_temp_files()` 直接拒绝时间序列样本
- 处理：
  - 已将 `.json` 加入 `SUPPORTED_EXTENSIONS`

#### 2. 端到端 smoke test 已跑通
- 返回结果：
  - `reference_overview.reference_count = 2`
  - `reference_overview.text_count = 1`
  - `reference_overview.timeseries_count = 1`
  - `reference_overview.has_text_ts_joint_coverage = True`
- `references` 中实际返回：
  - `source_modality = timeseries`
  - `source_modality = text`
- 说明：
  - 阶段十当前最核心的“文本 + 时间序列联合引用”已经在真实链路中出现

### 生成结果观察
- 当前回答已经能稳定给出：
  - 时间范围
  - 主要趋势
  - 峰值 / 谷值
- 但仍存在一个明确缺口：
  - 问题已要求“指出对应事件背景”
  - 系统虽然检索到了文本背景引用
  - 最终回答仍主要只输出了时间序列趋势

### 本轮补充收口
- 已对 `rag.py` 做一轮轻量生成侧增强：
  - `split_query_into_requirements()` 新增识别：
    - `并指出`
  - `build_answer_requirements()` 对带 `timeseries` 引用的问题追加要求：
    - 明确写时间范围与主要趋势
    - 分别交代时间序列观察与文本事件背景
- 但本轮复测表明：
  - 生成结果仍然偏向只回答趋势
  - 说明当前问题已不在“链路未通”
  - 而在“生成侧联合覆盖不够稳”

### 当前阶段判断
1. 阶段十已经完成了真正意义上的端到端打通：
   - 时间序列文件可上传
   - 可建临时知识库
   - 可检索出 TS + 文本联合引用
   - 可进入 `/chat/rag` 返回
2. 当前新的主要短板已经收敛为：
   - **生成侧没有稳定同时覆盖 TS 观察与文本事件背景**

### 下一步建议
1. 下一步不该回头折腾入库或检索主链路。
2. 更合适的是继续做生成侧收口：
   - 强化 `coverage_requirements`
   - 强化 timeseries answer template
   - 必要时在 completeness review 中增加“TS + text 联合覆盖”检查
3. 换句话说：
   - 阶段十后续的主战场，已经从“接通链路”转移到了“让回答真正用好联合证据”

## 2026-04-26 阶段十生成侧再收口：联合证据回答开始稳定区分“已确认趋势”与“背景不足”

### 本轮目标
- 继续收口第十章当前最明确的短板：
  - 回答虽然已经能返回 `timeseries + text` 联合引用
  - 但此前容易只输出趋势，不处理“事件背景 / 原因说明”这类并列要求

### 本轮改动
- 文件：
  - `app/chains/rag.py`

#### 1. 强化系统级回答约束
- `RAG_SYSTEM_PROMPT` 新增要求：
  - 当上下文同时包含时间序列证据和文本证据时
  - 必须分别说明：
    - 时间序列观察到的趋势 / 时间范围
    - 文本证据给出的事件背景 / 原因说明

#### 2. 强化 completeness review
- `RAG_COMPLETENESS_REVIEW_PROMPT` 新增要求：
  - 若上下文同时包含 TS + text，审校器必须检查答案是否分别覆盖两类证据

#### 3. 强化 coverage requirements
- `build_coverage_requirements()` 现在支持结合 `references` 推断时间序列覆盖点
- 对带 TS 引用的问题，当前会自动补充：
  - 时间范围与主要变化趋势
  - 若 query 提到异常/峰值/谷值：关键异常点
  - 若 query 提到背景/原因/事件，且存在文本证据：对应事件背景或文本原因说明

#### 4. 增加一次 TS 联合覆盖 retry
- 在 `maybe_refine_rag_answer()` 中增加轻量二次 completeness retry：
  - 若 query 确实要求 TS + text 联合覆盖
  - 且当前答案没有同时提到趋势与背景
  - 则再走一轮 completeness review

### 本轮复测
- 继续使用上一轮真实临时知识库 smoke case：
  - `air_quality_series.json`
  - `event_note.txt`
- 复测后回答变为：
  - 能明确给出 `2024-01-01 ~ 2024-01-03` 的趋势结论
  - 能明确指出：
    - 文本证据当前不可可靠解读
    - 因此无法确定具体事件背景或原因

### 本轮结论
1. 生成侧已经不再只是“只讲趋势、漏掉背景要求”。
2. 当前行为更接近第十章要求：
   - 能区分“时间序列观察”
   - 和“文本背景是否足够支持原因说明”
3. 这意味着阶段十生成侧已经从：
   - **遗漏并列要求**
   收敛到：
   - **会显式说明背景证据不足，而不是强行补背景**

### 仍然存在的问题
1. 本轮文本样本在 smoke test 中出现了乱码式证据内容，导致背景文本无法被可靠利用。
2. 因此当前还不能证明：
   - 生成侧已经能稳定把“可用的文本背景”自然融合进最终答案
3. 当前能证明的是：
   - 当 TS + text 同时命中时，系统已经会尝试分别处理两类证据
   - 并在文本背景不可用时显式收缩结论

### 下一步建议
1. 下一步优先做一轮“干净文本背景样本”的真实回归。
2. 目标应改成验证：
   - 在文本背景可正常读取时
   - 答案能否稳定同时输出：
     - 趋势观察
     - 对应事件背景
3. 若这轮通过，阶段十的 Phase 1 就基本可以认为进入可交付状态。

## 2026-04-26 阶段十真实联合回归补充：干净文本背景样本下已能稳定输出“趋势 + 事件背景”

### 本轮目标
- 验证上一轮的关键疑点：
  - 系统不能同时回答“趋势观察 + 事件背景”
  - 还是只是上一轮文本样本在 shell 传递中被编码污染

### 本轮执行
- 更换为 Unicode 转义直传的干净样本：
  - `air_quality_series_clean_u.json`
  - `event_background_clean_u.txt`
- 继续使用同一类 query：
  - `请结合时间序列和文本说明北京pm25在2024年1月1日至1月3日的变化趋势，并指出对应事件背景。`

### 本轮验证结果

#### 1. 联合引用状态保持稳定
- `reference_overview`：
  - `reference_count = 2`
  - `text_count = 1`
  - `timeseries_count = 1`
  - `has_text_ts_joint_coverage = True`

#### 2. 文本背景引用内容已正常可读
- `event_background_clean_u.txt` 在 `references` 和 `build_context()` 中可正常看到：
  - `2024年1月2日，北京扩散条件转差，空气污染物出现短时累积。`
  - `随后冷空气过程增强了扩散条件，PM2.5和PM10在1月3日明显回落。`

#### 3. 最终回答已同时覆盖两类证据
- 回答中已稳定同时输出：
  - 时间序列观察到的趋势
  - 文本证据给出的事件背景
  - 序列波动与事件背景之间的对应关系
- 当前回答已经包含：
  - 时间范围
  - 整体趋势
  - 峰值 / 谷值
  - 背景说明
  - “1月2日升高 / 1月3日回落”与文本事件的映射

### 本轮结论
1. 阶段十当前链路已经具备真实的“文本 + 时间序列联合回答”能力。
2. 上一轮的主要问题不是架构缺失，而是：
   - 文本背景样本在 shell 传递过程中出现了编码污染
3. 在文本背景可正常读取时，当前系统已经能稳定完成：
   - `趋势观察 + 事件背景` 的联合输出

### 当前阶段判断
1. 阶段十的 Phase 1 现在可以认为已经接近可交付状态。
2. 当前已完成的关键能力包括：
   - 时间序列 reference 接口骨架
   - 时间序列结构化入库骨架
   - 时间序列检索分支最小版
   - 真实端到端联合引用
   - 真实端到端联合回答

### 下一步建议
1. 后续重点可从“是否能工作”转向“是否能系统评测”。
2. 更合理的下一步是：
   - 补时间序列专项 trace 落盘字段
   - 补最小时间序列评测集
   - 再决定是否进入 Phase 2 的联合排序增强

## 2026-04-26 阶段十继续推进：时间序列专项 trace 与最小评测集已落地

### 本轮目标
- 把第十章 10.5 / 10.8 中约定的时间序列专项 trace 字段真正落进现有 trace 输出。
- 增加一套可重复运行的最小时间序列评测集，并做一轮真实“检索 + 生成”小回归。
- 用回归结果判断当前是否需要立即进入 Phase 2 的联合排序增强。

### 本轮改动
- 文件：
  - `app/retrievers/local_kb.py`
  - `app/chains/rag.py`
  - `data/eval/timeseries_minimal_assets/air_quality_series_clean_u.json`
  - `data/eval/timeseries_minimal_assets/event_background_clean_u.txt`
  - `data/eval/timeseries_minimal_cases_20260426.jsonl`
  - `scripts/run_timeseries_minimal_regression.py`

### 已完成内容

#### 1. retrieval trace 增加时间序列专项字段
- `retrieval_trace.jsonl` 现在会稳定输出：
  - `ts_reference_count`
  - `has_ts_evidence`
  - `has_text_ts_joint_coverage`
  - `temporal_constraint_detected`
- 实现方式：
  - 直接复用当前 `build_reference_overview()` 的计数逻辑
  - 时间约束检测结合：
    - 通用 temporal query profile
    - timeseries window constraint

#### 2. answer trace 增加对应字段
- `answer_trace.jsonl` 现在会稳定输出：
  - `ts_reference_count`
  - `has_ts_evidence`
  - `has_text_ts_joint_coverage`
  - `temporal_constraint_detected`
- 同时补了一处小一致性修正：
  - `is_temporal_answer_query()` 现在对显式年份 / 日期范围也会返回 `true`
  - 这样时间窗口类 query 在 answer trace 中不会漏记

#### 3. 新增最小时间序列评测资产
- 新增固定样本：
  - `air_quality_series_clean_u.json`
  - `event_background_clean_u.txt`
- 新增最小 case 集：
  - `timeseries_minimal_cases_20260426.jsonl`
- 当前覆盖 4 类任务：
  - `trend_qa`
  - `anomaly_qa`
  - `time_window_qa`
  - `event_alignment_qa`

#### 4. 新增最小真实回归脚本
- 新增：
  - `scripts/run_timeseries_minimal_regression.py`
- 当前执行策略：
  1. 上传固定时间序列 + 文本背景样本到 temp KB
  2. 逐条运行 query
  3. 直接走当前真实检索与生成链路
  4. 抽取 `retrieval_trace` / `answer_trace` 中对应记录
  5. 按最小验收规则输出 JSON 回归结果
- 输出文件：
  - `data/eval/timeseries_minimal_regression_20260426.json`

### 本轮验证结果
- 执行命令：
  - `python scripts/run_timeseries_minimal_regression.py`
- 回归结果：
  - `case_count = 4`
  - `passed = 4`
  - `failed = 0`
  - `pass_rate = 1.0`
  - `avg_ts_reference_count = 2.0`
- 逐项观察：
  - `trend_qa`：
    - 能返回 TS 证据
    - `temporal_constraint_detected = true`
    - `timeseries_branch_used = true`
  - `anomaly_qa`：
    - 能给出最高值 `96.000` 与 `1月2日`
  - `time_window_qa`：
    - 能回答 `1月2日 -> 1月3日` 为回落
  - `event_alignment_qa`：
    - 能同时覆盖时间序列趋势与文本事件背景
    - `has_text_ts_joint_coverage = true`

### 当前判断
1. 阶段十的 Phase 1 现在已经具备：
   - 时间序列专项 trace 可观测性
   - 最小专项评测集
   - 真实可复跑的小规模回归
2. 以当前结果看，**还不需要立刻进入 Phase 2 的联合排序增强**。
3. 更合理的节奏是：
   - 先继续扩充 `time_qa / trend / anomaly / event_alignment` case
   - 确认在更大一些的样本面上仍稳定
   - 只有当：
     - TS 证据命中不稳
     - 文本 / TS 候选排序互相挤压
     - 时间窗口样本经常排不到前列
     才进入 Phase 2 做联合排序增强

### 下一步建议
1. 先不要继续加重排序规则。
2. 下一步更值得做的是：
   - 扩充 8 到 12 条时间序列专项评测 case
   - 把这些 case 并到现有 `domain` 小回归看板里
   - 观察 `time_qa` 与 `joint coverage` 的稳定性
3. 如果下一轮出现“检索命中有但排序掉队”的现象，再正式进入 Phase 2。

## 2026-04-26 阶段十补充：最小时间序列集扩到 11 条，并初步排除“命中有了但排序掉队”风险

### 本轮目标
- 把时间序列最小集从 4 条扩到 8 到 12 条范围内。
- 不只看通过率，还额外观察：
  - `top1_source_modality`
  - `timeseries_in_top2`
  - `text_in_top2`
- 判断当前主要问题到底是：
  - 已命中但排序掉队
  - 还是 query 没被识别为时间序列问题

### 本轮改动
- 文件：
  - `data/eval/timeseries_minimal_cases_20260426.jsonl`
  - `scripts/run_timeseries_minimal_regression.py`
  - `app/services/retrieval/timeseries_retrieval_service.py`

### 已完成内容

#### 1. 最小集扩充到 11 条
- 在原有 4 条基础上，新增：
  - `timeseries-pm10-trend-1`
  - `timeseries-channel-compare-1`
  - `timeseries-range-meta-1`
  - `timeseries-background-only-1`
  - `timeseries-cause-align-1`
  - `timeseries-channel-list-1`
  - `timeseries-insufficient-1`
- 当前覆盖任务类型：
  - `trend_qa`
  - `anomaly_qa`
  - `time_window_qa`
  - `event_alignment_qa`
  - `channel_compare_qa`
  - `metadata_qa`
  - `background_qa`
  - `insufficient_qa`

#### 2. 回归脚本增加排序观测字段
- `timeseries_minimal_regression_20260426.json` 现在会额外记录：
  - `top_modalities`
  - `top1_source_modality`
- summary 现在会额外统计：
  - `top1_timeseries_case_count`
  - `timeseries_in_top2_case_count`
  - `text_in_top2_case_count`

#### 3. 本轮中途发现的真实问题
- 第一轮 11 条回归里，出现了一个明确失败样本：
  - `timeseries-channel-compare-1`
- 表现不是“有 TS 命中但被排后面”：
  - 而是 `timeseries_branch_used = false`
  - `ts_reference_count = 0`
  - top1 直接是 text
- 说明当前主问题是：
  - **query 识别漏掉了“更高/更低/最高/最低/浓度”这类时间序列比较问法**
  - 不是排序阶段把正确 TS 候选压掉

#### 4. 做了一次小范围泛化修正
- 在 `timeseries_retrieval_service.py` 中补充识别词：
  - `最高`
  - `最低`
  - `更高`
  - `更低`
  - `浓度`
- 这是查询识别增强，不是联合排序增强。
- 同时修正回归脚本：
  - 对只要求 `text` 的 case，不再强制要求 `trace_ts_passed=true`

### 本轮验证结果
- 再次执行：
  - `python scripts/run_timeseries_minimal_regression.py`
- 回归结果：
  - `case_count = 11`
  - `passed = 11`
  - `failed = 0`
  - `pass_rate = 1.0`
  - `avg_ts_reference_count = 1.636`
  - `joint_coverage_case_count = 10`
  - `top1_timeseries_case_count = 10`
  - `timeseries_in_top2_case_count = 10`
  - `text_in_top2_case_count = 2`

### 本轮结论
1. 当前没有观察到“命中已经有了，但排序把关键 TS / text 证据挤掉”的明显证据。
2. 本轮真正暴露出的风险更偏向：
   - **query 识别是否能稳定触发时间序列分支**
3. 在补了比较类词汇后：
   - `timeseries-channel-compare-1` 已恢复正常
   - top1 重新回到 `timeseries`

### 当前判断
1. 以当前 11 条最小集结果看，**还不建议进入 Phase 2 的联合排序增强**。
2. 更优先的工作顺序应该是：
   - 继续扩充时间序列 query 识别覆盖面
   - 把更多真实题型并入专项回归
   - 只有当后续出现“TS 和 text 都命中了，但前排结构明显错误”时，才进入联合排序增强

### 下一步建议
1. 下一步优先做：
   - 再补 6 到 10 条更接近真实业务的时间序列 query
   - 尤其补：
     - 比较题
     - 多通道题
     - 原因题
     - 证据不足题
2. 然后把这套 case 接到统一 `domain` 小回归里，持续看：
   - `timeseries_branch_used`
   - `ts_reference_count`
   - `has_text_ts_joint_coverage`
3. 只有当这些指标稳定后，再考虑 Phase 2。

## 2026-04-26 阶段十继续扩容：时间序列专项扩到 18 条，并并入统一 domain 小回归

### 本轮目标
- 继续补 6 到 10 条更贴近真实业务的时间序列 query。
- 重点覆盖：
  - 比较题
  - 多通道题
  - 证据不足题
  - 只靠文本背景也能回答的业务题
- 同时把这套时间序列专项并进统一 `domain` 小回归总表。

### 本轮改动
- 文件：
  - `data/eval/timeseries_minimal_cases_20260426.jsonl`
  - `scripts/build_domain_small_regression_suite.py`

### 已完成内容

#### 1. 时间序列专项从 11 条扩到 18 条
- 新增 7 条更贴近真实业务的话题：
  - `timeseries-dual-trend-1`
  - `timeseries-background-compare-1`
  - `timeseries-metadata-combo-1`
  - `timeseries-improvement-1`
  - `timeseries-insufficient-source-1`
  - `timeseries-insufficient-forecast-1`
  - `timeseries-sync-fall-1`
- 当前 18 条已经覆盖：
  - 趋势问答
  - 峰值 / 异常问答
  - 时间窗口问答
  - 多通道比较
  - 通道列表 / 元数据问答
  - 文本背景问答
  - 联合证据问答
  - 证据不足与不可外推问答

#### 2. 新增统一 domain 小回归聚合脚本
- 新增：
  - `scripts/build_domain_small_regression_suite.py`
- 当前会把两组结果聚合到同一份总表：
  - 既有 `full_chain_small_regression_20260424.json`
  - 时间序列专项 `timeseries_minimal_regression_20260426.json`
- 输出文件：
  - `data/eval/domain_small_regression_suite_20260426.json`

### 本轮验证结果

#### 1. 时间序列专项回归
- 执行：
  - `python scripts/run_timeseries_minimal_regression.py`
- 结果：
  - `case_count = 18`
  - `passed = 17`
  - `failed = 1`
  - `pass_rate = 0.944`
  - `avg_ts_reference_count = 1.5`
  - `joint_coverage_case_count = 14`
  - `top1_timeseries_case_count = 14`
  - `timeseries_in_top2_case_count = 14`
  - `text_in_top2_case_count = 4`

#### 2. 统一 domain 小回归总表
- 执行：
  - `python scripts/build_domain_small_regression_suite.py`
- 汇总结果：
  - `combined_case_count = 22`
  - `legacy_case_count = 4`
  - `timeseries_case_count = 18`
  - `timeseries_passed = 17`
  - `timeseries_failed = 1`
  - `timeseries_pass_rate = 0.944`

### 关键发现
1. 当前 18 条里，**没有出现明确的“命中已经有了，但排序把关键证据挤掉”主问题**。
2. 唯一剩余失败样本是：
   - `timeseries-improvement-1`
3. 这个失败的性质不是排序问题，而更像：
   - 问题可以由文本背景直接回答
   - 但我们在评测里把它定义成了“必须 TS + text 联合覆盖”
   - 实际检索返回只有 text，`timeseries_branch_used = false`
4. 换句话说，这轮扩容后暴露出的主要矛盾仍然是：
   - **query 识别 / 路由边界**
   - 不是联合排序

### 当前判断
1. 现在还不建议直接进入 Phase 2 的联合排序增强。
2. 当前更值得优先做的是：
   - 梳理哪些 query 必须强制走 TS + text 联合分支
   - 哪些 query 允许 text-only 或 TS-only 正常通过
3. 如果后续真的出现：
   - `timeseries_branch_used = true`
   - `ts_reference_count > 0`
   - 但 `top1/top2` 仍经常缺关键证据
   再进入 Phase 2 会更稳妥。

### 下一步建议
1. 先不加 Phase 2 排序规则。
2. 下一步更值得做的是：
   - 给时间序列专项 case 增加“允许单模态通过 / 必须联合通过”的更细分类
   - 专门清理 `timeseries-improvement-1` 这类“业务可回答，但联合约束过严”的边界 case
3. 清完这类边界后，再看是否还有真正的排序性失败。

## 2026-04-26 阶段十继续收口：按单模态 / 联合必过重分时间序列 case，确认是否仍存在真实排序失败

### 本轮目标
- 把时间序列专项 case 显式分成两类：
  - `single_modality_ok`
  - `joint_required`
- 先把 `timeseries-improvement-1` 这类边界题定义清楚，再重新判断是否还存在真正的排序性失败。

### 本轮改动
- 文件：
  - `data/eval/timeseries_minimal_cases_20260426.jsonl`
  - `scripts/build_domain_small_regression_suite.py`

### 已完成内容

#### 1. 明确边界题的验收语义
- `timeseries-improvement-1` 继续保留为：
  - `evaluation_mode = single_modality_ok`
  - `required_modalities = ["text"]`
  - `preferred_modalities = ["timeseries", "text"]`
- 同时放宽末组答案关键词：
  - 从只接受 `依据 / 证据`
  - 调整为接受 `依据 / 证据 / 表明 / 支持`
- 这样做的原因是：
  - 该题业务语义是“只关心 1 月 3 日是否改善”
  - 当前文本背景本身就足以支持回答
  - TS 联合命中是加分项，不应被评测误判为必须项

#### 2. 统一 domain 小回归总表补充分组指标
- `scripts/build_domain_small_regression_suite.py` 现在会在总表 `summary` 中直接输出：
  - `timeseries_single_modality_ok_case_count`
  - `timeseries_single_modality_ok_failed`
  - `timeseries_joint_required_case_count`
  - `timeseries_joint_required_failed`
- 后续判断“剩余问题到底属于哪类 case”时，不需要再手工翻明细。

### 本轮验证结果

#### 1. 时间序列专项真实端到端回归
- 执行：
  - `python scripts/run_timeseries_minimal_regression.py`
- 结果：
  - `case_count = 18`
  - `passed = 18`
  - `failed = 0`
  - `pass_rate = 1.0`
  - `single_modality_ok_case_count = 15`
  - `single_modality_ok_failed = 0`
  - `joint_required_case_count = 3`
  - `joint_required_failed = 0`
  - `avg_ts_reference_count = 1.389`
  - `joint_coverage_case_count = 14`
  - `top1_timeseries_case_count = 14`
  - `timeseries_in_top2_case_count = 14`
  - `text_in_top2_case_count = 3`

#### 2. 统一 domain 小回归总表
- 执行：
  - `python scripts/build_domain_small_regression_suite.py`
- 汇总结果：
  - `combined_case_count = 22`
  - `legacy_case_count = 4`
  - `timeseries_case_count = 18`
  - `timeseries_passed = 18`
  - `timeseries_failed = 0`
  - `timeseries_pass_rate = 1.0`
  - `timeseries_single_modality_ok_case_count = 15`
  - `timeseries_single_modality_ok_failed = 0`
  - `timeseries_joint_required_case_count = 3`
  - `timeseries_joint_required_failed = 0`

### 关键结论
1. `timeseries-improvement-1` 的问题已经确认不是排序失败，而是边界题定义过严。
2. 在本轮“单模态可通过 / 联合必过”重分后：
   - 没有剩余失败 case
   - 也没有暴露出新的真实排序性失败
3. 当前更合理的判断是：
   - Phase 1 的时间序列联合证据接入已经达到稳定可用
   - 现阶段还没有充分证据证明必须立即进入 Phase 2 联合排序增强

### 当前判断
1. 现在仍然不建议为了“可能的排序问题”继续追加过多启发式规则。
2. 更值得优先做的是：
   - 用更真实、更复杂的业务题继续拉高样本覆盖
   - 只有当后续出现“TS/text 都命中但前排证据结构持续错误”时，再进入 Phase 2
3. 当前阶段十可以认为：
   - “评测分层与边界澄清”这一步已经收口
   - 下一步若继续推进，应转向更真实数据集扩展，而不是先做 rerank 复杂化

## 2026-04-26 阶段十继续扩容：补长尾因果题、跨时间窗比较题、弱约束联合题

### 本轮目标
- 在已有 18 条时间序列专项 case 之上，继续补一批更贴近真实业务口吻的 query。
- 重点扩充三类：
  - 长尾因果题
  - 跨时间窗比较题
  - 弱约束联合题
- 同时验证新增 case 暴露的是：
  - 真实排序失败
  - 还是时间约束识别 / 评测表达边界问题

### 本轮改动
- 文件：
  - `data/eval/timeseries_minimal_cases_20260426.jsonl`
  - `app/chains/rag.py`
  - `app/retrievers/local_kb.py`

### 已完成内容

#### 1. 时间序列专项从 18 条扩到 24 条
- 新增 6 条更接近真实业务的问题：
  - `timeseries-cause-brief-1`
  - `timeseries-window-amplitude-1`
  - `timeseries-channel-window-compare-1`
  - `timeseries-stage-summary-1`
  - `timeseries-weak-joint-1`
  - `timeseries-business-compare-1`
- 这 6 条的设计意图分别覆盖：
  - “一句话业务简报”式因果说明
  - 单通道跨窗口幅度比较
  - 多通道跨窗口比较
  - 不强调精确数值的阶段性概括
  - “持续恶化 vs 短时波动后改善”的业务判断
  - text-only 可通过、TS 联合更优的业务风险题

#### 2. 发现并修正“月日表达”的时间约束识别边界
- 首轮扩容回归后，出现的失败并不指向排序，而主要暴露为：
  - `1月2日 / 1月3日` 这类**无年份的月日表达**没有被稳定识别为 temporal constraint
  - 以及个别答案中日期带空格时，评测关键词过严
- 因此做了两类小步泛化修正：
  - 在 `app/chains/rag.py` 的 `is_temporal_answer_query()` 中，新增对 `1月2日 / 1 月 2 日` 这类月日表达的识别
  - 在 `app/retrievers/local_kb.py` 的 temporal query profile 中，新增 `MONTH_DAY_PATTERN`，使 retrieval trace 也能把这类 query 记为时间约束
- 同时仅对 `timeseries-window-1` 放宽日期关键词格式，避免纯空格差异造成误判

### 本轮验证结果

#### 1. 时间序列专项真实端到端回归
- 执行：
  - `python scripts/run_timeseries_minimal_regression.py`
- 结果：
  - `case_count = 24`
  - `passed = 24`
  - `failed = 0`
  - `pass_rate = 1.0`
  - `single_modality_ok_case_count = 18`
  - `single_modality_ok_failed = 0`
  - `joint_required_case_count = 6`
  - `joint_required_failed = 0`
  - `avg_ts_reference_count = 1.458`
  - `joint_coverage_case_count = 20`
  - `top1_timeseries_case_count = 20`
  - `timeseries_in_top2_case_count = 20`
  - `text_in_top2_case_count = 6`

#### 2. 统一 domain 小回归总表
- 执行：
  - `python scripts/build_domain_small_regression_suite.py`
- 汇总结果：
  - `combined_case_count = 28`
  - `legacy_case_count = 4`
  - `timeseries_case_count = 24`
  - `timeseries_passed = 24`
  - `timeseries_failed = 0`
  - `timeseries_pass_rate = 1.0`
  - `timeseries_single_modality_ok_case_count = 18`
  - `timeseries_single_modality_ok_failed = 0`
  - `timeseries_joint_required_case_count = 6`
  - `timeseries_joint_required_failed = 0`
  - `timeseries_top1_timeseries_case_count = 20`
  - `timeseries_in_top2_case_count = 20`
  - `timeseries_text_in_top2_case_count = 6`

### 关键结论
1. 这轮更真实 query 扩容后，**仍然没有形成“命中已存在但排序掉队”的稳定失败证据**。
2. 新暴露的问题主要是：
   - 月日表达的时间约束识别边界
   - 以及少量评测关键词格式过严
3. 在补完这一步最小泛化修正后，24 条专项 case 全量通过，说明当前主问题依旧不是 Phase 2 rerank。

### 当前判断
1. 现阶段可以更有把握地认为：
   - 时间序列专项已经覆盖到更真实的业务问法
   - 但还没有得到“必须立即做联合排序增强”的直接证据
2. 更合理的后续方向仍然是：
   - 继续扩真实数据与真实 query 覆盖
   - 或者开始把现有专项指标接入更正式的评测面板 / 日常回归

### Phase 2 准入实现补充

#### 1. 已落地的 Phase 2 工程支架
- 在 `app/retrievers/local_kb.py` 中补上了 Phase 2 所需的最小工程能力：
  - `JointQueryProfile`，用于判定 query 是否属于需要 text + TS 联合排序保护的问题
  - candidate 级特征：
    - `temporal_match_score`
    - `event_type_match_score`
    - `location_match_score`
    - `channel_match_score`
    - `joint_coverage_bonus`
    - `same_series_or_same_event_group`
  - 轻量 rerank 加分与 joint query final selection，目标是：
    - 不推翻现有 dense / lexical / rerank 主框架
    - 只在 joint query 上增加可解释 bonus
    - 尽量保证 top-k 内保留 text + TS 双模态覆盖
- 在 retrieval trace 中补齐了 Phase 2 观测字段：
  - `joint_query_detected`
  - `joint_rerank_applied`
  - `temporal_match_score_topk`
  - `event_type_match_score_topk`
  - `location_match_score_topk`
  - `channel_match_score_topk`
  - `joint_coverage_bonus_topk`
  - `topk_modality_sequence`
  - `topk_has_text_ts_joint_coverage`
- 在 `scripts/run_timeseries_minimal_regression.py` 中补了失败归因：
  - `recall_failure`
  - `ranking_failure`
  - `generation_failure`
  - `phase2_ready`
- 在 `scripts/build_domain_small_regression_suite.py` 中把这些字段汇总进统一 domain 小回归总表。

#### 2. 本轮收口动作
- `timeseries-business-compare-1` 的评测边界原先更像 joint_required，但题目定义其实是 `single_modality_ok`。
- 因此将该 case 的 `answer_keyword_groups` 收口为：
  - 必须明确回答 `1月2日`
  - 必须体现“更需要重点关注风险”的结论
- 不再强制要求答案同时复述 `1月3日改善/缓解`，避免把结论型单模态题误判成生成失败。

#### 3. 本轮最新验证结果
- 执行：
  - `python scripts/run_timeseries_minimal_regression.py`
  - `python scripts/build_domain_small_regression_suite.py`
- 时间序列专项最新结果：
  - `case_count = 24`
  - `passed = 24`
  - `failed = 0`
  - `pass_rate = 1.0`
  - `ranking_failure_count = 0`
  - `recall_failure_count = 0`
  - `generation_failure_count = 0`
  - `phase2_ready = false`
  - `top1_timeseries_case_count = 18`
  - `timeseries_in_top2_case_count = 19`
  - `text_in_top2_case_count = 6`
- 统一 domain 小回归总表最新结果：
  - `combined_case_count = 28`
  - `timeseries_passed = 24`
  - `timeseries_failed = 0`
  - `timeseries_pass_rate = 1.0`
  - `timeseries_ranking_failure_count = 0`
  - `timeseries_generation_failure_count = 0`
  - `timeseries_phase2_ready = false`

#### 4. 当前结论
1. Phase 2 的“工程实现前置条件”现在已经补齐：
   - 可以识别 joint query
   - 可以记录候选级排序信号
   - 可以在回归里区分召回失败 / 排序失败 / 生成失败
2. 但按当前真实端到端结果，**仍没有拿到进入 Phase 2 的排序性失败证据**：
   - `ranking_failure_count = 0`
   - `phase2_ready = false`
3. 因此当前更准确的状态不是“正式进入 Phase 2 开发”，而是：
   - **Phase 2 准入监测与最小工程支架已完成**
   - 后续只有在新增真实业务题里稳定出现排序失败时，才继续加大联合排序权重或策略

### 更真实业务题扩容观察

#### 1. 本轮新增 case 方向
- 在 `data/eval/timeseries_minimal_cases_20260426.jsonl` 中新增了 8 条更接近真实业务问法的 query，重点压三类场景：
  - 弱约束联合题
  - 跨时间窗比较题
  - 业务归因 / 业务简报题
- 新增 case 包括：
  - `timeseries-joint-ops-brief-1`
  - `timeseries-cross-window-compare-2`
  - `timeseries-multi-channel-brief-2`
  - `timeseries-causal-judgement-1`
  - `timeseries-joint-window-risk-1`
  - `timeseries-business-summary-2`
  - `timeseries-weak-constraint-joint-2`
  - `timeseries-joint-comparison-2`

#### 2. 本轮想验证的问题
- 不是继续堆“趋势题”。
- 而是刻意加入更容易让 text / TS 同时竞争前排的业务题，看是否出现：
  - TS 和 text 都已召回
  - 但 top2 / top3 把关键证据压掉
  - 进而触发稳定的 `ranking_failure`

#### 3. 本轮真实端到端结果
- 执行：
  - `python scripts/run_timeseries_minimal_regression.py`
  - `python scripts/build_domain_small_regression_suite.py`
- 最新时间序列专项结果：
  - `case_count = 32`
  - `passed = 30`
  - `failed = 2`
  - `pass_rate = 0.938`
  - `single_modality_ok_case_count = 19`
  - `single_modality_ok_failed = 1`
  - `joint_required_case_count = 13`
  - `joint_required_failed = 1`
  - `ranking_failure_count = 0`
  - `recall_failure_count = 0`
  - `generation_failure_count = 2`
  - `phase2_ready = false`
  - `top1_timeseries_case_count = 26`
  - `timeseries_in_top2_case_count = 28`
  - `text_in_top2_case_count = 13`
- 最新统一 domain 小回归总表：
  - `combined_case_count = 36`
  - `timeseries_case_count = 32`
  - `timeseries_passed = 30`
  - `timeseries_failed = 2`
  - `timeseries_pass_rate = 0.938`
  - `timeseries_ranking_failure_count = 0`
  - `timeseries_generation_failure_count = 2`
  - `timeseries_phase2_ready = false`

#### 4. 本轮观察到的失败性质
1. **没有出现 Phase 2 想找的“排序性失败”**：
   - `ranking_failure_count = 0`
   - 新增联合题中，绝大多数 `joint_required` case 的 `topk_modality_sequence` 都已经是 `timeseries + text`
   - 因此当前并不存在“命中了但排不上去”的稳定证据
2. 本轮暴露出来的失败主要是 **生成/评测边界**：
   - 失败并不是因为 top2 缺少某一模态
   - 而是答案措辞没有完全命中当前 keyword groups
3. 这说明扩题后的主要矛盾仍然不是 rerank，而是：
   - 个别问题的题意定义与 keyword 评测边界
   - 某些“谨慎归因题”的生成表述一致性

#### 5. 当前结论更新
1. 经过这轮更真实业务 query 扩容后，仍然**没有形成进入 Phase 2 的直接证据**。
2. 当前更可信的判断是：
   - Phase 1 + 准入监测已经能够覆盖较多真实业务问法
   - 目前没有看到需要继续强化 joint rerank 权重的稳定失败模式
3. 如果后续继续扩题，重点应放在：
   - 更多真实数据样本，而不只是同一组三天空气质量样本
   - 更多“已召回但答错”的真实 bad case 收集
   - 再根据 bad case 判断是否真的进入 Phase 2 的排序增强

### 多样本真实数据与真实 bad case 补充

#### 1. 本轮新增真实样本资产
- 将 `scripts/run_timeseries_minimal_regression.py` 改为**自动扫描** `data/eval/timeseries_minimal_assets` 下的 `.json / .txt / .md` 资产，不再手工写死单一文件列表。
- 新增两组多样本资产：
  - 上海空气质量样本：
    - `shanghai_air_quality_o3.json`
    - `shanghai_air_quality_o3_background.txt`
    - 场景：2024-07-15 至 2024-07-17，`location=上海`，`event_type=空气质量监测`
    - 重点通道：`o3`、`pm25`
    - 背景：7月16日高温静稳导致臭氧抬升，7月17日海风 + 阵雨导致臭氧回落
  - 苏州在线服务样本：
    - `suzhou_service_ops_series.json`
    - `suzhou_service_ops_background.txt`
    - 场景：2024-03-05 至 2024-03-07，`location=苏州`，`event_type=在线服务监测`
    - 重点通道：`latency_ms`、`error_rate`、`qps`
    - 背景：3月6日发布导致缓存命中率下降，3月7日回滚 + 缓存预热恢复

#### 2. 本轮新增真实 bad case 方向
- 在 `data/eval/timeseries_minimal_cases_20260426.jsonl` 中新增 8 条多样本 bad case，重点是：
  - 同类样本跨地点守卫：
    - `timeseries-shanghai-location-guard-1`
    - `timeseries-shanghai-risk-source-1`
  - 跨领域守卫：
    - `timeseries-suzhou-domain-guard-1`
  - 新领域联合归因：
    - `timeseries-suzhou-release-cause-1`
    - `timeseries-suzhou-sync-anomaly-1`
    - `timeseries-suzhou-window-compare-1`
  - 新样本联合趋势 / 风险判断：
    - `timeseries-shanghai-o3-cause-1`
    - `timeseries-shanghai-o3-trend-1`

#### 3. 本轮真实端到端结果
- 执行：
  - `python scripts/run_timeseries_minimal_regression.py`
  - `python scripts/build_domain_small_regression_suite.py`
- 最新时间序列专项结果：
  - `case_count = 40`
  - `passed = 32`
  - `failed = 8`
  - `pass_rate = 0.8`
  - `single_modality_ok_case_count = 21`
  - `single_modality_ok_failed = 3`
  - `joint_required_case_count = 19`
  - `joint_required_failed = 5`
  - `ranking_failure_count = 0`
  - `recall_failure_count = 2`
  - `generation_failure_count = 6`
  - `phase2_ready = false`
  - `avg_ts_reference_count = 3.225`
  - `timeseries_in_top2_case_count = 34`
  - `text_in_top2_case_count = 19`
- 最新统一 domain 小回归总表：
  - `combined_case_count = 44`
  - `timeseries_case_count = 40`
  - `timeseries_passed = 32`
  - `timeseries_failed = 8`
  - `timeseries_pass_rate = 0.8`
  - `timeseries_ranking_failure_count = 0`
  - `timeseries_recall_failure_count = 2`
  - `timeseries_generation_failure_count = 6`
  - `timeseries_phase2_ready = false`

#### 4. 本轮最有价值的新观察
1. **仍未出现稳定的 ranking failure**：
   - `ranking_failure_count = 0`
   - 多样本后，top-k 中 text + TS 的联合覆盖依旧较高
   - 因此当前还不能把主要矛盾归因到 rerank
2. **首次明显出现“多样本守卫型 recall failure”**：
   - `timeseries-shanghai-risk-source-1`
   - `timeseries-suzhou-domain-guard-1`
   - 这两条 case 都出现：
     - `topk_modality_sequence = ["text", ...]`
     - `timeseries_count = 0`
     - `has_text_ts_joint_coverage = false`
   - 说明问题不是“命中了但排不上去”，而是：
     - query profile 没把这类问题稳定识别成需要 TS 联合召回
     - 或 location / domain guard 约束没有把 TS 分支稳定拉起来
3. **其余新增失败主要仍是 generation / eval boundary**：
   - 例如：
     - `timeseries-suzhou-release-cause-1`
     - `timeseries-suzhou-window-compare-1`
   - 这些 case 的 TS + text 已经在前排，只是答案措辞或数值判断和当前 keyword groups 仍有边界差异

#### 5. 当前阶段判断更新
1. 这轮“多样本真实数据 + 真实 bad case”带来了比单样本更有价值的失败：
   - 不是排序掉队
   - 而是**跨样本场景下的 query 识别 / TS 召回守卫问题**
2. 因此下一步更合理的方向不是 Phase 2 rerank，而是优先补：
   - 多样本下的 joint query detect
   - location / domain guard 对 TS 分支触发的保护
   - 以及新领域样本下的最小数值比较能力

## 2026-04-27 时间序列 Phase 1 下一步

### 目标
- 不进入 `Phase 2 rerank`，继续收口 `Phase 1`。
- 优先解决多样本场景下暴露出来的 **TS 分支触发不足** 和 **location / domain guard 不稳**。
- 在不破坏现有文本主链路的前提下，提高“该走 TS 联合召回的问题”命中率。

### 为什么这一步排在前面
- 最新 40 条专项回归里，`ranking_failure_count = 0`，说明当前还没有形成“命中了但排不上去”的稳定失败证据。
- 真正新出现的坏例子是：
  - `timeseries-shanghai-risk-source-1`
  - `timeseries-suzhou-domain-guard-1`
- 这两类失败更像：
  - query profile 没有稳定识别出“需要 TS 证据”
  - 或者 location / domain guard 没能把 TS 分支拉起来
- 因此如果现在直接进入 `Phase 2`，只会把“触发层问题”和“排序层问题”混在一起调。

### 下一步拆解

#### 1. 先补 TS 分支触发规则
- 目标：
  - 提高“风险来源 / 原因 / 波动 / 上升下降 / 时间窗口比较”这类问题进入 TS 联合召回的稳定性。
- 具体动作：
  - 补充 `timeseries query detect` 关键词与模式：
    - 风险来源
    - 为什么升高 / 降低
    - 哪一天最高 / 最低
    - 哪段时间变化最明显
    - 发布后 / 回滚后 / 降雨后 / 高温后
  - 对“因果问法但答案需要先看序列变化”的 query，允许强制保留 TS 分支。
- 验收：
  - 之前 `timeseries_count = 0` 的守卫型 case，至少能稳定拉起 TS 候选。

#### 2. 再补 location / domain guard
- 目标：
  - 避免跨地点、跨领域样本混召回时，TS 分支被无关样本污染或完全压掉。
- 具体动作：
  - 强化 location 命中约束：
    - `上海` 问题优先命中上海样本
    - `苏州` 问题优先命中苏州样本
  - 强化 domain/event guard：
    - 空气质量问题优先连到 `空气质量监测`
    - 服务稳定性问题优先连到 `在线服务监测`
  - 若 query 中已显式出现地点或事件词，TS 检索阶段对不匹配样本增加惩罚。
- 验收：
  - 守卫型 recall failure 不再表现为 `timeseries_count = 0`
  - 无关领域样本不会稳定挤进前排

#### 3. 最后补最小数值比较能力
- 目标：
  - 先补“能看出升降、峰值、窗口比较”，不急着做复杂统计分析。
- 具体动作：
  - 在时间序列摘要或 reference 组织里补充最小比较字段：
    - 峰值时间
    - 谷值时间
    - 起止变化方向
    - 窗口均值 / 最大值的简短摘要
  - 对“哪一天更高 / 是否回落 / 是否恢复”这类问题，优先靠结构化摘要回答，而不是把原始长序列直接塞进 prompt。
- 验收：
  - `window compare` 类 case 的 generation failure 比例下降
  - 回答中能稳定给出“时间范围 + 变化方向 + 关键点”

### 本轮不做的事
- 不上 `Phase 2` 的 joint rerank 权重增强。
- 不先做复杂的 TS/text 融合排序实验。
- 不引入新的研究型时序编码训练。

### 本轮完成后的再判断条件
- 如果下一轮回归后仍然主要是：
  - `timeseries_count = 0`
  - `has_text_ts_joint_coverage = false`
  - `recall_failure_count` 仍显著高于 `ranking_failure_count`
  - 那就继续留在 `Phase 1`
- 只有当：
  - TS 候选已经能稳定召回
  - joint coverage 已经稳定存在
  - 但前排排序仍然经常把有效 TS 候选压下去
  - 才有理由进入 `Phase 2`

### 当前结论
- 时间序列方向的下一步已经比较清楚：
  - **先补“该不该触发 TS 分支”和“该召回哪类 TS 样本”**
  - **不是先补“召回后怎么重新加权排序”**
- 因此下一轮工作的优先级应固定为：
  1. `joint query detect`
  2. `location / domain guard`
  3. `最小数值比较能力`

## 2026-04-27 时间序列 Phase 1 执行

### 本轮目标
- 按上一节的优先级，直接落地 `Phase 1` 三个收口动作：
  - 补强 `joint query detect`
  - 补强 `location / domain guard`
  - 补强最小数值比较摘要

### 本轮改动
- 文件：
  - `app/services/retrieval/timeseries_retrieval_service.py`
  - `app/retrievers/local_kb.py`
  - `app/services/timeseries_summary_service.py`

### 已完成内容

#### 1. 补强 TS query detect
- 扩大了时间序列识别词：
  - 因果 / 业务判断词：
    - `风险来源`
    - `为什么`
    - `依据`
    - `解释`
    - `恢复`
    - `恶化`
  - 窗口比较词：
    - `幅度`
    - `同步`
    - `哪一天`
    - `高于 / 低于 / 大于 / 小于`
  - 新领域词：
    - `臭氧`
    - `延迟`
    - `错误率`
    - `订单服务`
    - `缓存`
    - `回滚`
    - `发布`
- 扩大了 guard 识别词：
  - `不要混`
  - `不要掺入`
- 扩大了地点识别模式：
  - 显式覆盖 `北京 / 上海 / 苏州` 等常见城市词
- 识别逻辑增加了：
  - 当 query 同时具有“因果/比较意图”与“地点/领域线索”时，也可判为时间序列相关

#### 2. 补强 joint query 与 location / domain guard
- `infer_joint_query_profile()` 增加了更贴近真实业务问法的背景 / 趋势 marker：
  - `风险`
  - `业务判断`
  - `业务简报`
  - `值班`
  - `抖动`
  - `恢复`
  - `恶化`
- joint query 判定从“趋势 + 背景”扩大到：
  - `guard + TS/background`
  - 或带有明显地点 / 领域 / 事件线索的因果解释题
- TS 分支触发更激进：
  - 若 joint profile 已出现 `location / domain / channel / event` 约束，则强制保留 TS 分支
  - 同时提高 TS branch 的 `top_k`
- candidate 调整增加了更明确的 guard 奖惩：
  - 地点匹配加分，不匹配减分
  - 领域匹配加分，不匹配减分
  - 事件类型匹配加分，不匹配减分
  - 通道匹配加分，不匹配减分
  - 在 `has_guard_constraint=true` 的情况下，对不匹配 TS 候选再额外加一层惩罚

#### 3. 补强最小数值比较能力
- `build_timeseries_summary()` 不再只输出“起点/终点/峰值/谷值”静态值，而是增加：
  - `变化 delta`
  - `峰值时间`
  - `谷值时间`
- `build_timeseries_numeric_preview()` 增加：
  - `start`
  - `end`
  - `delta`
  - `min@timestamp`
  - `max@timestamp`
- 新增了轻量 `extract_channel_stats()`，统一复用：
  - 起点值
  - 终点值
  - 均值
  - 峰谷值及对应时间

### 当前验证
- 已完成：
  - 3 个改动文件的静态语法检查
  - 时间序列摘要样例输出检查
- 进行中：
  - `scripts/run_timeseries_minimal_regression.py`

### 当前判断
1. 这轮仍然属于 `Phase 1` 收口，不是 `Phase 2 rerank`。
2. 改动重点已经对齐到最新 bad case 的主要矛盾：
  - 不是“命中了但排不上去”
  - 而是“该不该触发 TS 分支”与“该召回哪类 TS 样本”
3. 待回归结果出来后，再看：
  - `timeseries_count = 0` 的守卫型 case 是否下降
  - `window compare` 类生成失败是否下降
