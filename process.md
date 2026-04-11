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
