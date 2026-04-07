# 基于 LangChain 的 RAG 智能体对话系统学习与搭建文档

## 1. 文档目标

这份文档有两个目的：

1. 以 `Langchain-Chatchat` 为样本，系统理解一个完整 RAG + Agent 应用是如何组织起来的。
2. 基于阅读结果，提炼一套更适合本地学习和快速复现的 `LangChain + Ollama + FAISS + FastAPI/Streamlit` 搭建方案。

本文默认读者具备基本 Python 开发能力，希望解决三个问题：

- 这个项目到底是怎么工作的？
- 一个通用的 LangChain RAG 智能体系统应该拆成哪些模块？
- 如果我自己从零开始，怎样最快搭起一个可运行版本？

> 说明
>
> `Langchain-Chatchat` 是完整应用框架，不是最小化 demo。阅读时应区分“成熟工程设计”和“教学版最小闭环”两层目标，避免一开始就把系统做得过重。

---

## 2. Langchain-Chatchat 项目阅读

## 2.1 项目定位

从 `README.md` 和后端源码可以看出，`Langchain-Chatchat` 的定位是：

- 基于大语言模型与 LangChain 的本地知识库问答系统。
- 既支持传统 RAG，又支持工具调用型 Agent。
- 通过 `FastAPI` 提供接口，通过 `Streamlit` 提供 WebUI。
- 支持多种模型平台、多种向量库、多种工具和多种知识来源。

它不是“只做一条问答链”的项目，而是一个可离线部署、可扩展、偏平台化的应用框架。

## 2.2 阅读重点与关键文件

官方贡献文档把项目描述成 monorepo，当前阅读时最关键的后端实现集中在 `libs/chatchat-server` 下。对学习 RAG 与 Agent 来说，下面这些文件最值得读：

| 路径 | 作用 |
| --- | --- |
| `libs/chatchat-server/chatchat/settings.py` | 统一管理数据目录、YAML 配置、模型平台、知识库参数、工具参数 |
| `libs/chatchat-server/chatchat/cli.py` | 命令行入口，负责 `init`、`start`、`kb` 等能力拼装 |
| `libs/chatchat-server/chatchat/startup.py` | 启动 API 和 WebUI 进程 |
| `libs/chatchat-server/chatchat/server/api_server/server_app.py` | FastAPI 应用装配入口 |
| `libs/chatchat-server/chatchat/server/api_server/chat_routes.py` | 对话相关路由 |
| `libs/chatchat-server/chatchat/server/api_server/kb_routes.py` | 知识库管理路由 |
| `libs/chatchat-server/chatchat/server/api_server/tool_routes.py` | 工具查询与调用路由 |
| `libs/chatchat-server/chatchat/server/chat/kb_chat.py` | 本地知识库 / 临时知识库 / 搜索引擎 RAG 问答主链路 |
| `libs/chatchat-server/chatchat/server/chat/file_chat.py` | 上传文件后临时构建 FAISS 并进行文件问答 |
| `libs/chatchat-server/chatchat/server/chat/chat.py` | Agent 对话与工具调用主链路 |
| `libs/chatchat-server/chatchat/server/knowledge_base/*` | 知识库服务、文档管理、向量库适配 |
| `libs/chatchat-server/chatchat/server/agent/tools_factory/*` | 内置工具实现 |
| `libs/chatchat-server/chatchat/server/agent/tools_factory/tools_registry.py` | 工具注册机制 |

## 2.3 配置与数据目录

`settings.py` 是理解全项目的起点。这里的关键点有三层。

### 1. 用 `CHATCHAT_ROOT` 管理运行时数据

项目通过环境变量 `CHATCHAT_ROOT` 确定配置与数据根目录：

```python
CHATCHAT_ROOT = Path(os.environ.get("CHATCHAT_ROOT", ".")).resolve()
```

这意味着：

- 配置文件不强绑定源码目录。
- 数据文件、日志、媒体文件、临时文件都能独立存放。
- 项目更像一个“可安装应用”，而不是只能在源码目录里运行的脚本。

### 2. 通过多个 YAML 文件拆分配置职责

核心配置分为三类：

- `basic_settings.yaml`
  - 服务地址
  - 知识库根目录
  - 数据库路径
  - 日志目录
  - 临时目录
- `kb_settings.yaml`
  - 默认向量库类型
  - 切分参数
  - 检索 `top_k`
  - 阈值
  - 搜索引擎配置
- `model_settings.yaml`
  - 默认 LLM
  - 默认 Embedding
  - Agent 模型
  - 模型平台列表
  - 各阶段模型参数

这种拆法很值得继承，因为它天然对应三类变化频率不同的信息：运行环境、RAG 参数、模型参数。

### 3. 支持多平台模型接入

`settings.py` 中的 `MODEL_PLATFORMS` 能同时描述 `Xinference`、`Ollama` 等模型平台。核心思想是：

- 业务代码不要直接写死某一个模型厂商。
- 先抽象模型平台，再配置模型名称。
- 对上层链路来说，只关心“我需要 LLM / Embedding / Reranker 能力”。

这对成熟系统很有价值，但对教学版来说可以裁剪。教学版不需要一开始就支持多平台自动检测，先固定 `Ollama` 即可。

## 2.4 初始化与启动流程

`cli.py` 和 `startup.py` 定义了项目从“空目录”到“可运行服务”的完整生命周期。

### 1. `chatchat init`

`cli.py` 中的 `init()` 主要做四件事：

- 创建数据目录。
- 复制示例知识库 `samples`。
- 初始化知识库数据库。
- 生成默认 YAML 配置模板。

这一点非常重要：`init` 不是只创建配置，而是在创建“应用运行所需的完整本地状态”。

### 2. `chatchat kb -r`

该命令负责把知识库目录中的文件重新向量化，构建或重建向量库。

### 3. `chatchat start -a`

`startup.py` 里定义了：

- `--api`
- `--webui`
- `-a/--all`

其中 `-a` 同时启动 API 和 WebUI。启动逻辑本质是多进程拉起：

- FastAPI 服务
- Streamlit 页面

这是完整应用常见的组织方式，但教学版可以更轻：先只启动 FastAPI，前端后补。

## 2.5 API 路由组织方式

`server_app.py` 统一装配 FastAPI 应用，并注册了多个 router：

- `chat_router`
- `kb_router`
- `tool_router`
- `openai_router`
- `server_router`
- `mcp_router`

对学习最重要的是前三个。

### 1. `chat_routes.py`

负责对话接口，核心包括：

- `POST /chat/kb_chat`
  - 知识库问答
- `POST /chat/file_chat`
  - 文件问答
- `POST /chat/chat/completions`
  - 兼容 OpenAI 风格的统一对话入口

这个统一入口的设计很值得注意：

- 如果传 `tools`，走 Agent。
- 如果传 `tool_choice`，可直接调用工具或让模型决定如何调用。
- 如果不传工具，则退化为普通对话。

这说明项目并没有把“聊天”“RAG”“工具调用”做成完全割裂的三套系统，而是尽量统一到一个 API 体验里。

### 2. `kb_routes.py`

知识库路由不只负责上传文件，还负责整个知识库生命周期：

- 列出知识库
- 创建知识库
- 删除知识库
- 上传文档
- 更新文档
- 重建向量库
- 搜索知识库
- 上传临时文件
- 搜索临时知识库

同时它还提供了按知识来源包装的 OpenAI 兼容入口：

- `POST /knowledge_base/{mode}/{param}/chat/completions`

这说明 `Langchain-Chatchat` 在接口层已经尝试把“知识库问答能力”包装成统一的聊天协议，而不只是提供一个内部专用 API。

这意味着 `Langchain-Chatchat` 里的知识库不是“一个文件夹 + 一个检索函数”，而是一个受控资源对象。

### 3. `tool_routes.py`

工具路由相对简单：

- `GET /tools`
  - 列出工具及其参数
- `POST /tools/call`
  - 直接调用某个工具

这种设计非常适合调试，也很适合前端做工具面板。

## 2.6 知识库管理链路

知识库相关逻辑分散在 `knowledge_base` 目录下，但可以抽象成四步：

1. 文件进入知识库目录。
2. 文档加载器把原始文件转成文本块。
3. 文本块被向量化并写入向量库。
4. 检索时从向量库取回相似文本，再交给 LLM。

源码层面的关键点：

- `kb_service/*`
  - 不同向量库后端的适配层，如 `faiss`、`milvus`、`chromadb` 等
- `kb_doc_api.py`
  - 知识库文档的上传、更新、搜索
- `kb_api.py`
  - 知识库资源本身的管理
- `kb_cache/*`
  - 向量库缓存

这个分层是成熟工程设计：

- 上层 API 不直接操作具体向量库。
- 通过 service 层隔离底层存储差异。

教学版可以继承“service 抽象”的思路，但只保留一个后端 `FAISS`。

## 2.7 RAG 问答链是怎么跑起来的

`kb_chat.py` 是最核心的 RAG 阅读文件之一。其主链路非常清晰：

1. 根据 `mode` 决定知识来源。
   - `local_kb`
   - `temp_kb`
   - `search_engine`
2. 执行检索：
   - 本地知识库走 `search_docs`
   - 临时知识库走 `search_temp_docs`
   - 搜索引擎走 `search_engine`
3. 将检索结果格式化为引用信息。
4. 把检索文本拼成 `context`。
5. 读取 `prompt_settings.yaml` 中的 RAG 模板。
6. 使用 `ChatPromptTemplate` 组装历史消息和当前问题。
7. 调用 LLM 生成答案。
8. 通过流式输出返回回答和引用。

可以把它压缩成一句话：

> 检索结果不是直接返回，而是先组织成上下文，再进入 Prompt，由 LLM 基于上下文回答。

这正是标准 RAG 的核心思想。

### `kb_chat.py` 值得学习的地方

- `mode` 统一了三类知识来源。
- 检索与生成被显式分开，逻辑清楚。
- 支持 `return_direct`，方便调试检索质量。
- 没有检索到文档时会切换到 `empty` prompt 模板。
- 输出里带引用，便于追溯答案依据。

### `kb_chat.py` 可以裁剪的地方

- 多种知识来源统一入口在教学版中不是必须。
- Langfuse 这类观测集成可以先不做。
- 搜索引擎对话可后置。

## 2.8 文件问答链路

`file_chat.py` 展示了另一种很实用的 RAG 场景：用户上传文件，系统立刻构建临时向量库并对话。

这条链路分两段：

### 1. 上传阶段

`upload_temp_docs()` 会：

- 接收上传文件
- 保存到临时目录
- 调用 `KnowledgeFile.file2text()` 解析文本
- 将文本块写入 `memo_faiss_pool` 对应的临时 FAISS

### 2. 对话阶段

`file_chat()` 会：

- 用 query 生成 embedding
- 在临时 FAISS 中检索
- 拼接 `context`
- 调用 LLM
- 返回答案与出处

这条链路对教学版非常值得保留，因为它直接覆盖了“上传 PDF 即问即答”这种最常见需求。

## 2.9 Agent 与工具调用链路

`chat.py` 是 `Langchain-Chatchat` 的 Agent 主入口。这里的核心思路不是“先写一个工具，再让模型随便调”，而是分成三层。

### 1. 模型配置层

`create_models_from_config()` 会根据 `LLM_MODEL_CONFIG` 创建不同阶段使用的模型，例如：

- `preprocess_model`
- `llm_model`
- `action_model`
- `postprocess_model`

说明这个系统允许不同阶段用不同模型，这属于完整平台级设计。

### 2. Agent 执行层

`create_models_chains()` 中会：

- 取历史消息
- 读取中间步骤
- 加载工具
- 构造 Agent Executor

这里用的是项目自定义的 `PlatformToolsRunnable.create_agent_executor()`，不是最基础的 LangChain demo 写法，说明项目为多模型适配和回调展示做了额外封装。

### 3. 工具层

工具实现位于 `server/agent/tools_factory/*`，如：

- `search_local_knowledgebase.py`
- `search_internet.py`
- `calculate.py`
- `arxiv.py`
- `shell.py`
- `text2sql.py`

工具注册则在 `tools_registry.py` 中完成，其核心思想是：

- 用装饰器 `@regist_tool` 自动注册工具。
- 为工具补齐标题、描述、参数 schema。
- 统一暴露给 Agent 与工具路由。

### 为什么这条链路值得学

因为它清楚地展示了 Agent 系统不是单个 prompt，而是至少包含：

- 可调用工具集合
- 工具元数据
- 历史对话
- 中间步骤状态
- Agent 执行器
- 回调与流式事件

## 2.10 哪些设计适合继承，哪些适合裁剪

### 值得继承的设计

- `CHATCHAT_ROOT` 形式的数据目录管理
- `basic/kb/model` 分文件配置
- 知识库 service 抽象
- RAG 与 Agent 分链路组织
- 工具注册机制
- API 与 WebUI 分离
- 引用回传与流式输出

### 教学版应主动裁剪的设计

- 多模型平台自动检测
- 多向量数据库后端
- MCP 连接
- 多模态图片、语音、文生图
- 复杂数据库对话
- 过多内置工具
- 完整的统一 OpenAI 兼容层

结论很简单：

> `Langchain-Chatchat` 适合拿来学架构，不适合原样照搬做第一次实现。

---

## 3. 从样本项目抽象出通用 RAG 智能体架构

## 3.1 RAG 主链路

教学版系统建议严格围绕下面这条主链路展开：

```text
文档导入
-> 文本解析
-> 文本切分
-> 向量化
-> 写入向量库
-> 用户提问
-> 问题向量化
-> 相似检索
-> 组装 Prompt
-> LLM 生成答案
-> 返回答案与引用
```

这一链路对应的最小模块为：

- `Document Loader`
- `Text Splitter`
- `Embedding`
- `Vector Store`
- `Retriever`
- `RAG Chain`
- `API/UI`

## 3.2 Agent 主链路

Agent 部分建议单独理解，不要与 RAG 混成一个概念。

```text
用户问题
-> 判断是否需要工具
-> 选择工具
-> 生成工具参数
-> 执行工具
-> 将工具结果回填上下文
-> 生成最终回答
```

教学版中建议只保留 2 到 3 个工具：

- `search_local_knowledge`
  - 复用本地知识库检索
- `calculate`
  - 处理数学表达式
- `current_time`
  - 演示非知识库类工具

这样既能展示 Agent 的本质，也能保持系统离线可复现。

## 3.3 将 Langchain-Chatchat 映射为通用模块

| 通用模块 | Langchain-Chatchat 中的对应实现 |
| --- | --- |
| `Document Loader` | `file_rag/document_loaders/*` |
| `Text Splitter` | `file_rag/text_splitter/*` |
| `Embedding` | `settings.py` 中的模型配置 + `server.utils` 中的 Embedding 获取逻辑 |
| `Vector Store` | `knowledge_base/kb_service/*` |
| `Retriever` | `search_docs`、`search_temp_docs`、FAISS 检索 |
| `RAG Chain` | `server/chat/kb_chat.py` |
| `Temp File RAG` | `server/chat/file_chat.py` |
| `Tool Registry` | `server/agent/tools_factory/tools_registry.py` |
| `Agent Executor` | `server/chat/chat.py` |
| `API/UI` | `api_server/*` + `webui.py` |

## 3.4 教学版系统边界

为了降低学习成本，本文定义的教学版边界如下：

### 第一阶段必须完成

- 本地知识库问答
- 上传文件后临时问答
- 至少 2 个工具的 Agent 对话
- FastAPI 接口
- 可选 Streamlit 页面

### 第二阶段再考虑

- 多模型平台切换
- 多向量库切换
- 搜索引擎联网问答
- SQL / PromQL / Shell 等高风险工具
- 多模态能力
- OpenAI 兼容 SDK 完整适配

---

## 4. 教学版参考接口约定

这一节不代表 `Langchain-Chatchat` 原有接口，而是后续自己搭建系统时建议采用的统一约定。

## 4.1 CLI 约定

建议的 CLI 接口如下：

| 命令 | 职责 | 对照 Chatchat |
| --- | --- | --- |
| `init` | 创建数据目录、生成配置文件、初始化元数据 | `chatchat init` |
| `rebuild_kb` | 重新切分、向量化、构建本地知识库 | `chatchat kb -r` |
| `start_api` | 启动 FastAPI 服务 | `chatchat start --api` |
| `start_ui` | 启动 Streamlit 页面 | `chatchat start --webui` |

建议不要一开始就做成一个大而全的 CLI，而是只保留这四个固定入口。

## 4.2 HTTP 接口约定

### 1. `POST /chat/rag`

职责：

- 本地知识库问答
- 临时文件问答

建议请求体：

```json
{
  "query": "请总结这份制度文件的核心要求",
  "source_type": "local_kb",
  "knowledge_base_name": "policies",
  "knowledge_id": "",
  "top_k": 4,
  "score_threshold": 0.5,
  "history": [],
  "stream": true
}
```

约定：

- `source_type=local_kb` 时使用 `knowledge_base_name`
- `source_type=temp_kb` 时使用 `knowledge_id`

### 2. `POST /chat/agent`

职责：

- 启用工具调用的智能体对话

建议请求体：

```json
{
  "query": "先帮我查知识库里的评分规则，再计算总分",
  "knowledge_base_name": "policies",
  "top_k": 4,
  "score_threshold": 0.5,
  "allowed_tools": [
    "search_local_knowledge",
    "calculate",
    "current_time"
  ],
  "max_steps": 4,
  "history": [],
  "stream": false
}
```

说明：

- 当前实现使用 `allowed_tools` 控制可调用工具集合。
- 若不传 `allowed_tools`，默认允许使用所有已注册工具。
- `max_steps` 用于限制单轮问题中的最大工具步数，避免死循环。
- 当前实现没有 `conversation_id` 字段。
- 当前实现在 `stream=true` 时已支持基于 SSE 的真实流式输出，非流式模式仍保持 JSON 响应。
- 当前 `/chat/agent` 响应中已包含 `tool_calls`、`steps`、`references` 三类轨迹数据。

### 3. `POST /knowledge_base/upload`

职责：

- 上传临时问答文件
- 上传长期知识库文件

当前实现说明：

- `scope=temp` 时创建临时 `knowledge_id` 并立即构建临时向量库。
- `scope=local` 时写入 `data/knowledge_base/<knowledge_base_name>/content/`。
- `scope=local` 支持 `overwrite_existing` 与 `auto_rebuild`。

建议表单字段：

- `files`
- `scope`
  - `temp`
  - `local`
- `knowledge_base_name`
  - `scope=temp` 时保持空字符串
  - `scope=local` 时必填
- `overwrite_existing`
  - `scope=local` 时控制是否覆盖同名文件
- `auto_rebuild`
  - `scope=local` 时控制上传后是否立即重建索引

### 4. `POST /knowledge_base/rebuild`

职责：

- 重建指定知识库的切片和向量库

建议请求体：

```json
{
  "knowledge_base_name": "policies",
  "chunk_size": 800,
  "chunk_overlap": 150,
  "embedding_model": "bge-m3:latest"
}
```

### 5. `GET /tools`

职责：

- 返回工具列表、参数 schema、工具说明

## 4.3 配置文件约定

建议保留与 `Langchain-Chatchat` 一致的三份配置文件。

### 1. `basic_settings.yaml`

只放运行环境与路径：

```yaml
PROJECT_NAME: rag-agent-system
API_HOST: 127.0.0.1
API_PORT: 8000
WEBUI_PORT: 8501
DATA_ROOT: ./data
KB_ROOT_PATH: ./data/knowledge_base
TEMP_ROOT_PATH: ./data/temp
LOG_PATH: ./data/logs
```

### 2. `kb_settings.yaml`

只放知识库与检索参数：

```yaml
DEFAULT_VS_TYPE: faiss
TEXT_SPLITTER_NAME: ChineseRecursiveTextSplitter
CHUNK_SIZE: 800
CHUNK_OVERLAP: 150
VECTOR_SEARCH_TOP_K: 4
SCORE_THRESHOLD: 0.5
TEMP_KB_TTL_MINUTES: 120
TEMP_KB_CLEANUP_ON_STARTUP: true
TEMP_KB_TOUCH_ON_ACCESS: true
```

### 3. `model_settings.yaml`

只放模型与平台参数：

```yaml
LLM_PROVIDER: ollama
OLLAMA_BASE_URL: http://127.0.0.1:11434
OPENAI_COMPATIBLE_BASE_URL: ""
OPENAI_COMPATIBLE_API_KEY: ""
DEFAULT_LLM_MODEL: qwen2.5:7b
DEFAULT_EMBEDDING_MODEL: bge-m3:latest
AGENT_MODEL: qwen2.5:7b
TEMPERATURE: 0.3
MAX_TOKENS: 2048
```

当前版本中，Embedding 仍默认走 Ollama；但“与大模型对话”这条链路已经支持按 `LLM_PROVIDER` 在 `ollama` 和 `openai_compatible` 之间切换。也就是说，RAG 回答、Agent 对话、查询改写都可以切到云端 API 模型，而不需要改业务接口。

## 4.4 数据对象约定

### `DocumentChunk`

| 字段 | 含义 |
| --- | --- |
| `chunk_id` | 切片唯一标识 |
| `doc_id` | 所属文档标识 |
| `source` | 来源文件名 |
| `source_path` | 来源文件完整路径 |
| `extension` | 文件扩展名 |
| `chunk_index` | 在原文中的切片序号 |
| `page` | 页码，若存在 |
| `title` | 标题，若存在 |
| `content_length` | 切片长度 |
| `content_preview` | 切片预览 |

### `RetrievedDoc`

| 字段 | 含义 |
| --- | --- |
| `chunk_id` | 命中的切片 ID |
| `raw_score` | FAISS 原始距离分数 |
| `relevance_score` | 归一化后的相关度分数 |
| `content` | 命中的文本 |
| `source` | 来源文件名或链接 |
| `source_path` | 来源路径 |

### `ChatRequest`

| 字段 | 含义 |
| --- | --- |
| `query` | 用户问题 |
| `history` | 历史对话 |
| `source_type` | `local_kb` 或 `temp_kb` |
| `knowledge_base_name` | 本地知识库名称 |
| `knowledge_id` | 临时知识库 ID |
| `top_k` | 检索条数 |
| `stream` | 是否流式返回 |

### `ToolCallRecord`

| 字段 | 含义 |
| --- | --- |
| `tool_name` | 工具名 |
| `arguments` | 工具入参 |
| `output` | 工具输出 |
| `status` | `success/error/skipped` |
| `error_message` | 错误信息 |

### `ChatResponse`

| 字段 | 含义 |
| --- | --- |
| `answer` | 最终回答 |
| `references` | 引用文档列表 |
| `source_type` | `local_kb` 或 `temp_kb` |
| `knowledge_base_name` | 知识库名称或临时知识库 ID |
| `used_context` | 是否实际使用了检索上下文 |
| `stream` | 是否声明为流式 |

补充说明：

- 当前实现还额外定义了 `AgentChatRequest` 与 `AgentChatResponse`，用于 `/chat/agent`。
- `AgentChatRequest` 中包含 `allowed_tools` 与 `max_steps`。
- `AgentChatResponse` 中包含 `tool_calls`、`steps`、`references`、`used_tools` 等字段。

---

## 5. 基于 LangChain 的教学版系统搭建方案

这一部分是本文的落地重点。目标不是复刻 `Langchain-Chatchat` 的全部能力，而是在 Windows 上搭建一套最小可运行、结构清楚、便于继续扩展的 RAG 智能体对话系统。

## 5.1 环境准备

### 操作系统基线

- Windows 10/11
- Python 3.10 或 3.11
- Ollama

### 推荐准备

1. 安装 Python
2. 安装 Ollama
3. 确认 PowerShell 能正常执行虚拟环境脚本

### 模型建议

默认采用“中文友好的本地模型”路线：

- LLM 示例
  - `qwen2.5:7b`
  - 或其他中文问答能力较强的指令模型
- Embedding 示例
  - `bge-m3:latest`
  - 或其他兼容中文检索的 embedding 模型

> 不要把模型名当成系统设计本身。模型可以替换，链路设计不变。

## 5.2 依赖安装

建议在独立虚拟环境中完成。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install langchain langchain-core langchain-community langchain-text-splitters
pip install langchain-ollama faiss-cpu fastapi "uvicorn[standard]" streamlit
pip install pydantic-settings python-multipart pypdf docx2txt markdown beautifulsoup4
```

如果后续需要更丰富的文件解析，可再补：

```powershell
pip install unstructured openpyxl
```

拉取模型：

```powershell
ollama pull qwen2.5:7b
ollama pull bge-m3:latest
```

## 5.3 项目目录设计

建议的教学版目录如下：

```text
rag-agent-system/
├── app/
│   ├── api/
│   │   └── main.py
│   ├── agents/
│   ├── chains/
│   ├── loaders/
│   ├── retrievers/
│   ├── services/
│   ├── tools/
│   ├── schemas/
│   └── ui/
│       └── app.py
├── configs/
│   ├── basic_settings.yaml
│   ├── kb_settings.yaml
│   └── model_settings.yaml
├── data/
│   ├── knowledge_base/
│   ├── temp/
│   ├── logs/
│   └── vector_store/
├── scripts/
│   ├── init.py
│   ├── rebuild_kb.py
│   ├── start_api.py
│   ├── start_ui.py
│   └── validate_phase7.py
└── requirements.txt
```

这种结构同时借鉴了 `Langchain-Chatchat` 的“配置与运行时数据分离”思想，又避免一开始拆成过多包。

## 5.4 配置说明

建议先只支持 `Ollama + FAISS` 这一条主路径。

### 设计原则

- `basic_settings.yaml` 只放路径和端口
- `kb_settings.yaml` 只放切分与检索参数
- `model_settings.yaml` 只放模型名和推理服务地址

### 为什么这样拆

因为这三类配置的变更频率不同：

- 部署时常改的是端口和路径
- 调优时常改的是切分和检索参数
- 模型切换时常改的是模型名和基地址

## 5.5 知识入库流程

教学版建议把本地知识库处理成固定四步。

### 第一步：上传或拷贝文件

本地知识库文件统一放入：

```text
data/knowledge_base/<knowledge_base_name>/content/
```

例如：

```text
data/knowledge_base/policies/content/
```

### 第二步：文档解析

根据文件类型选用加载器：

- PDF -> `PyPDFLoader`
- DOCX -> `Docx2txtLoader` 或自定义 loader
- Markdown -> 直接读取文本
- TXT -> 直接读取文本

### 第三步：文本切分

中文场景建议优先做：

- 递归切分
- 合理重叠
- 标题保留

对应参数可参考：

- `chunk_size: 800`
- `chunk_overlap: 150`

### 第四步：向量化与写入 FAISS

处理结果建议以两份数据形式存放：

- 原始文档与切片元数据
- FAISS 索引文件

这样后续能支持：

- 查看引用来源
- 重建向量库
- 调整切片参数后重新入库

## 5.6 RAG 对话链设计

教学版 `/chat/rag` 建议复用 `Langchain-Chatchat` 的基本思路，但简化实现。

### 建议链路

1. 接收问题与知识来源信息
2. 根据 `source_type` 选择本地知识库或临时知识库
3. 可选执行查询改写，提炼更适合检索的核心意图
4. 执行混合检索
   - 向量检索负责语义召回
   - 词法检索负责型号、货号、专有名词匹配
5. 对候选片段执行启发式重排序
6. 对命中的小块做邻块扩展，按大块上下文喂给模型
7. 组装上下文 `context`
8. 拼接 RAG Prompt
9. 调用 `ChatOllama`
10. 返回答案与引用

### 当前项目已落地的增强链路

当前代码实现已经把教学版 RAG 从“单路向量检索”升级为更接近企业项目常用的黄金组合：

- `查询改写`
  - 使用 `QUERY_REWRITE_MODEL` 或默认聊天模型，对口语化问题做检索友好的短查询改写
  - 当前实现会保留型号、数字、缩写等关键约束，并在改写失败时自动回退到原问题
- `混合检索`
  - 向量检索仍然基于 `FAISS`
  - 词法检索在本地 `docstore` 上执行 BM25 风格打分，重点补足型号、货号、专有名词召回
- `重排序`
  - 先用 RRF 融合向量结果和词法结果
  - 当前版本已接入本地 `CrossEncoder` 模型重排，启发式特征仅作为轻量 tie-break 和回退策略
- `小块查大块答`
  - 检索阶段仍以切片为粒度
  - 生成阶段会按 `doc_id + chunk_index` 扩展邻近切片，把更完整的原文段落送进 Prompt

当前相关配置集中在 `kb_settings.yaml` 和 `model_settings.yaml`：

- `ENABLE_QUERY_REWRITE`
- `ENABLE_HYBRID_RETRIEVAL`
- `ENABLE_HEURISTIC_RERANK`
- `ENABLE_MODEL_RERANK`
- `ENABLE_SMALL_TO_BIG_CONTEXT`
- `HYBRID_DENSE_TOP_K`
- `HYBRID_LEXICAL_TOP_K`
- `HYBRID_RERANK_TOP_K`
- `HYBRID_RRF_K`
- `RERANK_CANDIDATES_TOP_N`
- `RERANK_SCORE_THRESHOLD`
- `RERANK_FALLBACK_TO_HEURISTIC`
- `SMALL_TO_BIG_EXPAND_CHUNKS`
- `QUERY_REWRITE_MODEL`
- `RERANK_MODEL`
- `RERANK_DEVICE`

### 当前版本的模型重排实现

当前项目已在本地接入 `BAAI/bge-reranker-base`，默认配置如下：

- `RERANK_MODEL: ./data/models/bge-reranker-base`
- `RERANK_DEVICE: cpu`
- `ENABLE_MODEL_RERANK: true`

实现方式是：

- 先做混合召回，得到候选片段
- 再用 `sentence-transformers` 的 `CrossEncoder` 对候选做二次精排
- 启发式重排不再是主逻辑，只在模型不可用时回退，或者在模型分数极其接近时做轻微 tie-break

这意味着当前教学版已经不是“纯启发式重排”的最小实现，而是具备了一个真正可用的企业级精排模块。

### Prompt 原则

RAG Prompt 不要写得过度花哨，建议只保留核心约束：

- 只能优先依据给定上下文回答
- 若上下文不足，要明确说明
- 回答尽量给出依据
- 不要编造不存在的事实

### 返回结果原则

无论是否流式输出，都建议返回：

- `answer`
- `references`

因为“可追溯”是 RAG 系统区别于普通聊天的关键价值。

## 5.7 Agent 工具接入设计

教学版不建议直接复刻 `Langchain-Chatchat` 的完整 Agent 执行器，而是保留思想、简化结构。

### 推荐最小工具集

#### 1. `search_local_knowledge`

职责：

- 根据 query 调用本地知识库检索
- 把命中的文档片段作为工具输出返回

它实际上是把 RAG 检索能力变成 Agent 可调用工具。

#### 2. `calculate`

职责：

- 处理简单数学表达式

#### 3. `current_time`

职责：

- 返回当前时间

### 为什么这样选

- 都能本地运行
- 没有外部网络依赖
- 能清楚演示“模型决定何时调用工具”
- 风险远低于 `shell`、`sql`、联网搜索

### 工具注册建议

借鉴 `tools_registry.py` 的做法：

- 每个工具都应具备
  - 工具名
  - 描述
  - 参数 schema
  - 调用函数
- 系统应能统一枚举所有工具

## 5.8 FastAPI 接口层设计

教学版 API 层建议分成三组 router：

- `chat`
- `knowledge_base`
- `tools`

### `chat` router

- `POST /chat/rag`
- `POST /chat/agent`

### `knowledge_base` router

- `POST /knowledge_base/upload`
- `POST /knowledge_base/rebuild`
- `GET /knowledge_base/list`

### `tools` router

- `GET /tools`

这样既保留清晰边界，又不会像完整应用那样扩出太多接口。

## 5.9 可选 Streamlit 前端

前端不是第一优先级，但非常适合教学验证。

建议页面只做三块：

### 1. 知识库管理页

- 刷新知识库列表
- 查看知识库文件
- 选择知识库
- 触发重建

### 2. RAG 问答页

- 输入问题
- 查看答案
- 查看引用片段

### 3. Agent 对话页

- 选择工具
- 输入问题
- 展示工具调用过程

这样已经足够覆盖本文定义的三类场景。

## 5.10 启动与验证流程

建议把运行闭环固定为下面这组步骤。

### 第一步：启动 Ollama

```powershell
ollama list
```

若模型未准备好，继续执行：

```powershell
ollama pull qwen2.5:7b
ollama pull bge-m3:latest
```

### 第二步：初始化项目

```powershell
python .\scripts\init.py
```

目标：

- 创建 `data/`
- 创建 `configs/`
- 初始化本地元数据

### 第三步：导入知识库并重建

```powershell
python .\scripts\rebuild_kb.py --kb-name policies
```

### 第四步：启动 API

```powershell
python .\scripts\start_api.py
```

### 第五步：可选启动 Streamlit

```powershell
python .\scripts\start_ui.py
```

### 第六步：验证三类场景

#### 场景 A：本地知识库问答

- 上传制度文档到 `policies`
- 重建向量库
- 向 `/chat/rag` 发起问题
- 验证答案是否带引用

#### 场景 B：临时文件问答

- 通过 `/knowledge_base/upload` 上传文件，`scope=temp`
- 获取临时 `knowledge_id`
- 向 `/chat/rag` 发起问题，`source_type=temp_kb`

#### 场景 C：带工具的 Agent 问答

- 向 `/chat/agent` 发送问题
- 通过 `allowed_tools` 允许 `search_local_knowledge`、`calculate`、`current_time`
- 验证返回结果中有工具调用记录

### 第七步：执行项目级验证脚本

```powershell
python .\scripts\validate_phase7.py
```

该脚本会统一验证：

- `GET /knowledge_base/list`
- `POST /knowledge_base/rebuild`
- 本地知识库 RAG 问答
- 企业级 RAG 优化链路
- 临时文件上传与临时 RAG 问答
- 临时知识库自动清理与过期访问报错
- Agent 知识检索工具与时间工具
- Streamlit 页面基础交互

## 5.11 当前版本测试结果与已知限制

截至当前项目版本，已经完成以下闭环验证：

- 本地知识库 `phase2_demo` 可被列出、可被重建；重建统计会随样例文档内容变化，当前环境最近一次验证结果为 `files_processed=5`、`chunks=301`
- `POST /chat/rag` 在 `source_type=local_kb` 下可稳定命中 `intro.txt`
- `POST /knowledge_base/upload` 已支持长期知识库上传；`scope=local` 下可直接写入知识库目录并按需自动重建
- 本地长期知识库 `phase7_local_upload_demo` 已通过“上传 -> 自动重建 -> 问答”闭环验证
- 本地长期知识库 `phase7_rag_optimized_demo` 已通过“查询改写 + 混合检索 + 重排序 + 小块查大块答”闭环验证
- 本地长期知识库 `phasef_reranker_demo` 已通过“启发式首条命中错误，模型重排纠正为正确文档”的对比验证
- `POST /knowledge_base/upload` 可创建临时 `knowledge_id`，并在 `source_type=temp_kb` 下完成问答
- 上传临时文件后，响应中会返回 `expires_at`、`ttl_minutes`、`touch_on_access`
- 手动清理和启动清理都已接入，当前通过 `python .\scripts\cleanup_temp_kb.py` 可手动触发
- `POST /chat/agent` 已支持多工具连续编排，可按顺序触发 `search_local_knowledge`、`calculate`、`current_time`
- `GET /tools` 可列出 3 个核心工具
- Streamlit 页面已完成知识库刷新、RAG 请求、Agent 请求三类交互验证

最近一次 `python .\scripts\validate_phase7.py` 结果要点如下：

- 本地知识库问答：`200`，`used_context=true`，首条引用 `intro.txt`
- 长期知识库上传：`200`，`phase7_local_upload_demo` 上传后自动重建成功
- RAG 检索优化：`200`，`phase7_rag_optimized_demo` 已验证 `ZX900 -> ZX-900` 检索命中，首条引用包含保修信息和相邻售后上下文
- 模型重排对比：`phasef_reranker_demo` 中，启发式首条来源为 `distractor.txt`，模型重排首条来源纠正为 `warranty.txt`
- 长期知识库上传问答：`200`，首条引用 `phase7_local_upload.txt`
- 长期知识库重名处理：重复上传同名文件时返回 `skipped_files`
- 临时文件问答：`200`，`used_context=true`，首条引用 `phase7_temp.txt`
- 临时知识库清理：未过期目录不会被误删；手动清理过期目录可返回 `removed=1`
- 过期访问报错：`410 temp_knowledge_expired`
- Agent 知识检索：`200`，工具名 `search_local_knowledge`
- Agent 时间查询：`200`，工具名 `current_time`
- Agent 多步编排：`200`，已验证 `search_local_knowledge -> calculate`
- Agent 最大步数保护：`max_steps=1` 时会返回终止提示，不会继续调用第二个工具
- 流式输出：RAG 已验证 `reference -> token -> done`，Agent 已验证 `tool_call -> step -> token -> done`，多步场景可出现两次 `tool_call`
- 错误响应：缺少 `knowledge_base_name` 时返回 `422`，结构为 `code/message/details`
- UI：3 个 tab 正常渲染，长期知识库上传按钮存在，Agent `max_steps` 控件存在，RAG 与 Agent 页面交互均返回成功提示
- 检索评测：`python .\\scripts\\eval_retrieval.py --compare-model-rerank --show-cases` 当前结果为
  - 启发式：`Top1 source accuracy = 0.8571`，`MRR = 0.9286`
  - 模型重排：`Top1 source accuracy = 1.0`，`MRR = 1.0`

当前版本仍保留以下限制：

- `/chat/rag` 与 `/chat/agent` 在 `stream=true` 时使用 SSE 流式输出；若客户端不支持事件流解析，可继续使用非流式模式
- 临时知识库已支持 TTL 自动清理；在当前 Windows 环境下若目录被系统占用，会退化为软清理并在后续启动继续重试物理删除
- Agent 执行器当前采用稳定的自定义工具路由，不是模型原生 tool calling
- 当前多工具编排仍以启发式规则为主，适合教学和可控演示，不等同于通用 ReAct Planner
- 当前已接入本地 `Cross-Encoder Reranker`，但评测集还比较小；如果后续要继续提高稳定性，优先应该扩充评测样本和增加检索链路日志观测，而不是继续堆新功能
- 当前知识库重建已经升级为“增量优先”流程，但首次全量构建或“大量修改/删除文件”的场景仍可能较慢；如果页面仍报 `HTTPConnectionPool(host='127.0.0.1', port=8000): Read timed out. (read timeout=180)`，通常是重建耗时超出前端等待时间，而不是检索链路逻辑错误

---

## 6. 教学版与 Chatchat 的差异说明

为了避免误解，必须明确两者不是同一复杂度。

| 维度 | Langchain-Chatchat | 教学版系统 |
| --- | --- | --- |
| 模型平台 | 多平台 | 单平台 Ollama |
| 向量库 | 多后端 | 仅 FAISS |
| 工具数量 | 多工具 | 2 到 3 个核心工具 |
| API 体系 | 多路由 + OpenAI 兼容 | 少量固定接口 |
| 前端 | 完整 WebUI | 可选轻量页面 |
| 目标 | 可部署应用框架 | 学习与快速复现 |

一句话总结：

> Chatchat 适合学习“一个完整系统怎么设计”；教学版适合学习“一个系统最小闭环如何搭起来”。

---

## 7. 常见问题

## 7.1 为什么不直接复刻 Langchain-Chatchat？

因为第一次做 RAG 智能体系统时，最大的风险不是功能太少，而是复杂度失控。直接复刻完整工程会把学习重心从“理解链路”变成“处理工程细节”。

## 7.2 为什么教学版先选 FAISS？

因为：

- 本地可运行
- 零额外服务
- 足够支撑学习阶段
- 与临时文件问答场景天然匹配

## 7.3 为什么建议把 RAG 与 Agent 分成两条链？

因为两者关注点不同：

- RAG 的核心是“检索质量 + 引用可信度”
- Agent 的核心是“工具选择 + 工具编排”

分开实现更容易调试，也更容易定位问题。

## 7.4 为什么工具数量要少？

因为 Agent 的复杂度会随着工具数、工具描述质量、工具参数复杂度快速上升。第一次实现时，工具越少，越容易看清模型为什么会调用或不会调用。

## 7.5 中文场景最容易踩的坑是什么？

通常有三个：

- 切片过粗，导致检索上下文过脏
- Embedding 模型不适合中文
- 回答只看最终结果，不看引用命中质量

## 7.6 为什么知识库重建会报 `Read timed out (read timeout=180)`？

这个错误在当前项目里通常不是“API 服务挂了”，而是“前端等待重建结果超时了”。

当前项目在 Phase G 之前，知识库重建链路是同步全量执行的：

1. 重新读取整个知识库目录
2. 重新解析所有文档
3. 重新切片全部文本
4. 重新调用 embedding 模型
5. 重新构建整个 FAISS 索引

而 Streamlit 页面里的重建请求当前使用固定超时，重建一旦超过这个时间，页面就会先报：

```text
知识库重建失败: HTTPConnectionPool(host='127.0.0.1', port=8000): Read timed out. (read timeout=180)
```

### 当前版本的长期修复

这个问题现在已经完成第一版根治，不再总是全量重建。当前实现会在 [kb_incremental_rebuild.py](/e:/南京航空航天大学/aaa大创/智能体案例/Mini-Agent-RAG2/app/services/kb_incremental_rebuild.py) 中维护：

- `build_manifest.json`
  - 记录每个源文件的 `relative_path`、`size`、`mtime`、`sha256`
- `cache/chunks/*.json`
  - 缓存每个文件拆分后的 chunk 和对应 embedding
- `index_mode`
  - `full`：首次构建、配置变化、文件修改或删除时的安全回退
  - `reuse`：文件未变化时直接复用已有索引和缓存
  - `append`：只有新增文件时，在现有索引上追加新向量

也就是说，当前版本已经支持：

1. 文件级增量检测
2. 文件哈希缓存
3. chunk 与 embedding 复用
4. `FAISS.add_embeddings(...)` 追加索引
5. 批量 embedding 与并行解析

这一版优化之后，重复重建和“只新增文件”的场景会明显更快，页面超时概率也会显著下降。

### 临时解决方案

- 优先用 CLI 重建，而不是页面按钮：

```powershell
python .\scripts\rebuild_kb.py --kb-name your_kb_name
```

- 不要每上传一个文件就自动重建，改成“批量上传一次，再统一重建一次”。
- 适当调大 `chunk_size`、减小 `chunk_overlap`，减少 chunk 数量。
- 大型 `pdf/docx` 尽量先转成 `txt/md` 再入库。
- 如果只是联调页面，可以把 [app.py](/e:/南京航空航天大学/aaa大创/智能体案例/Mini-Agent-RAG2/app/ui/app.py) 中重建请求的超时时间从 `180` 调大。

### 当前仍需注意的边界

- 第一次构建一个大型知识库时，依然需要真正跑完整解析、切片、embedding 和建索引
- 如果文件被修改或删除，当前实现会为了索引一致性安全回退到 `full`
- 因此，极大的知识库首次导入、或者一次性覆盖很多文件时，依然更适合优先用 CLI 重建

也就是说，这个超时问题在当前项目里已经从“默认会遇到的全量重建问题”，收敛成了“首次全量或大规模变更时仍可能遇到的性能边界”。短期手段仍然可用，但长期解法已经实际落地成增量重建链路。

---

## 8. 最终建议

如果你的目标是“学习 RAG 技术并做出一份可指导后续开发的文档”，正确路线不是直接复制项目代码，而是分三步走：

1. 先读懂 `Langchain-Chatchat` 的配置、RAG、文件问答、Agent、工具注册五条主线。
2. 再抽象出一套只保留 `Ollama + FAISS + FastAPI + 2~3 个工具` 的教学版架构。
3. 最后围绕三个场景闭环实现：
   - 本地知识库问答
   - 临时文件问答
   - 带工具的 Agent 问答

只要这三个场景跑通，你就已经完成了一个真正有工程意义的 LangChain RAG 智能体系统基础版。后续再逐步扩展模型平台、向量库、多模态和更复杂工具，路线会更稳。
