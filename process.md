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
  - `data/eval/reranker_compare_domain100_base_vs_v2_m3.json`
  - `data/eval/reranker_compare_domain100_base_vs_v2_m3.md`

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
