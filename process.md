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
