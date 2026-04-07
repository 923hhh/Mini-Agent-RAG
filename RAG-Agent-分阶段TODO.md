# RAG-Agent 分阶段 TODO

## 1. 文档说明

本 TODO 文档用于指导“按 [LangChain-RAG-Agent-学习与搭建文档](./LangChain-RAG-Agent-学习与搭建文档.md) 落地实现系统”，不是文档润色清单。

默认执行基线固定为：

- Windows
- Python 虚拟环境
- Ollama
- FAISS
- FastAPI
- 可选 Streamlit

执行规则如下：

- 按 `Phase 0 -> Phase 7` 顺序推进，不跳阶段并行开发核心能力。
- 每个阶段完成后先做本阶段验证，再进入下一个阶段。
- 若当前阶段的完成标准未满足，不进入下一阶段。
- 若出现模型、Embedding、文件解析或工具调用阻塞，优先执行该阶段的“阻塞后回退方案”。

完成状态约定：

- `[ ]` 未开始
- `[x]` 已完成
- `[-]` 当前阻塞

---

## 2. 总里程碑

| Phase | 目标 | 前置依赖 | 预计产出 |
| --- | --- | --- | --- |
| Phase 0 | 环境与工程骨架 | 无 | 目录结构、虚拟环境、依赖清单、基础配置模板 |
| Phase 1 | 配置与初始化能力 | Phase 0 | `init` 初始化逻辑、配置加载逻辑 |
| Phase 2 | 本地知识库入库能力 | Phase 1 | 文档导入、切分、Embedding、FAISS、`rebuild_kb` |
| Phase 3 | 本地 RAG 对话接口 | Phase 2 | `POST /chat/rag` 的 `local_kb` 路径 |
| Phase 4 | 临时文件问答能力 | Phase 3 | 临时上传、临时向量库、`temp_kb` 路径 |
| Phase 5 | Agent 与工具调用 | Phase 3 | `POST /chat/agent`、3 个核心工具 |
| Phase 6 | FastAPI 整理与可选 Streamlit 页面 | Phase 4、Phase 5 | 稳定 API 路由、最小前端页面 |
| Phase 7 | 联调、测试与文档回填 | Phase 6 | 联调结果、测试结论、运行说明、已知问题 |

里程碑判断标准：

- Phase 3、Phase 4、Phase 5 是三条核心业务能力线。
- Phase 6 之前不追求复杂 UI，不扩展多模型平台，不增加高风险工具。
- Phase 7 结束时必须完成三个核心场景闭环：
  - 本地知识库问答
  - 临时文件问答
  - 带工具的 Agent 对话

---

## 3. 分阶段 TODO

## Phase 0 环境与工程骨架

### 目标

建立后续实现所需的最小工程骨架，固定技术基线和目录结构。

### 前置依赖

- 无

### 任务清单

- [x] 创建项目目录结构：
  - `app/`
  - `configs/`
  - `data/`
  - `scripts/`
- [x] 建立 Python 虚拟环境。
- [x] 固化依赖安装方式：
  - `langchain`
  - `langchain-community`
  - `langchain-ollama`
  - `faiss-cpu`
  - `fastapi`
  - `uvicorn`
  - `streamlit`
  - `pydantic-settings`
  - 文件解析相关依赖
- [x] 生成基础配置文件模板：
  - `basic_settings.yaml`
  - `kb_settings.yaml`
  - `model_settings.yaml`
- [x] 创建运行时目录：
  - `data/knowledge_base`
  - `data/temp`
  - `data/logs`
  - `data/vector_store`
- [x] 确认 Ollama 已安装且能拉取目标模型。

### 交付物

- 目录结构就绪
- 虚拟环境可用
- 配置模板可读
- 本地运行时目录齐全

### 完成标准

- 项目能进入基础 Python 环境。
- 配置文件和数据目录齐全。
- 依赖安装路径明确，后续阶段不需要重新决定环境方案。

### 验证方式

- 执行虚拟环境激活命令成功。
- 执行依赖安装命令无关键报错。
- 检查 `configs/` 与 `data/` 结构完整。
- 执行 `ollama list` 能正常返回。

### 阻塞后回退方案

- 若 Ollama 或模型拉取失败，先保留目录和配置模板，只冻结环境层，不进入对话功能开发。
- 若某类文件解析依赖安装失败，先只支持 `txt` 和 `md`，把 PDF/DOCX 解析延后到 Phase 2 末尾。

---

## Phase 1 配置与初始化能力

### 目标

实现 `init` 所需的最小初始化逻辑，让系统能一键生成本地运行结构。

### 前置依赖

- Phase 0 完成

### 任务清单

- [x] 实现配置加载逻辑，按学习文档约定读取：
  - `basic_settings.yaml`
  - `kb_settings.yaml`
  - `model_settings.yaml`
- [x] 实现数据目录创建逻辑。
- [x] 实现默认配置文件生成或校验逻辑。
- [x] 实现最小 `init` 命令入口。
- [x] 约定初始化后输出信息：
  - 配置路径
  - 数据目录
  - 默认模型
  - 默认向量库类型
- [x] 记录初始化失败时的报错位置与提示文案。

### 交付物

- 可运行的 `init`
- 配置加载模块
- 初始化输出信息

### 完成标准

- 一条命令可生成 `configs/` 与 `data/` 的最小运行结构。
- 初始化逻辑不依赖手工创建目录。
- 配置加载规则已固定，后续阶段不再改配置文件命名。

### 验证方式

- 在空目录状态执行 `init`。
- 检查配置文件是否存在且可解析。
- 检查 `data/` 目录是否完整创建。
- 重复执行 `init` 时不出现破坏性报错。

### 阻塞后回退方案

- 若 CLI 暂时未封装完成，允许先以 `python .\scripts\init.py` 形式落地。
- 若自动生成模板失败，先提供静态模板文件并在 TODO 中标记为待收敛项，但不改命名约定。

---

## Phase 2 本地知识库入库能力

### 目标

实现从本地知识库目录读取文档、切分文本、生成向量并写入 FAISS 的完整链路。

### 前置依赖

- Phase 1 完成

### 任务清单

- [x] 确定知识库目录约定：
  - `data/knowledge_base/<knowledge_base_name>/content/`
- [x] 实现基础文档加载能力：
  - `txt`
  - `md`
  - `pdf`
  - `docx`
- [x] 实现文本切分逻辑：
  - `chunk_size`
  - `chunk_overlap`
  - 中文友好切分策略
- [x] 接入 Ollama Embedding。
- [x] 实现 FAISS 向量库写入与持久化。
- [x] 实现知识块元数据存储：
  - 来源文件
  - 切片标识
  - 页码或标题信息
- [x] 实现 `rebuild_kb` 入口。
- [x] 支持对单个知识库重建索引。

### 交付物

- 本地知识库入库脚本或命令
- FAISS 索引文件
- 元数据文件

### 完成标准

- 指定知识库目录中的文档可以被重建为可检索索引。
- 向量库与元数据可持久化，重启后仍可加载。
- 切分参数与 Embedding 模型来自配置文件，不写死在业务逻辑中。

### 验证方式

- 准备至少 2 份样例文档并放入本地知识库。
- 执行 `rebuild_kb`。
- 检查是否生成 FAISS 索引和元数据。
- 随机抽查切片结果是否合理。

### 阻塞后回退方案

- 若 PDF 或 DOCX 解析不稳定，先只保证 `txt` 和 `md` 跑通本阶段验收。
- 若 Embedding 模型不稳定，先固定一个可用模型，不做模型切换能力。

---

## Phase 3 本地 RAG 对话接口

### 目标

实现 `POST /chat/rag` 的 `local_kb` 路径，支持本地知识库检索增强问答。

### 前置依赖

- Phase 2 完成

### 任务清单

- [x] 定义 `POST /chat/rag` 请求与响应结构。
- [x] 实现 `source_type=local_kb` 路由逻辑。
- [x] 实现 query 向量化与 FAISS 检索。
- [x] 实现检索结果到 `context` 的拼接逻辑。
- [x] 实现 RAG Prompt 模板。
- [x] 接入 LLM 生成回答。
- [x] 返回答案与引用列表：
  - `answer`
  - `references`
- [x] 支持 `top_k` 和 `score_threshold`。
- [x] 支持 `history` 与 `stream` 参数。

### 交付物

- `/chat/rag` 的 `local_kb` 能力
- RAG Prompt 模板
- 引用返回能力

### 完成标准

- 针对本地知识库问题可稳定返回带引用的回答。
- 检索与生成已形成闭环。
- API 能区分“未命中文档”和“正常命中文档”两种情况。

### 验证方式

- 对本地知识库发起至少 3 个问题。
- 检查回答是否与知识内容一致。
- 检查返回中是否包含引用来源。
- 检查无命中时是否给出合理提示。

### 阻塞后回退方案

- 若流式输出实现复杂，先交付非流式版本，但保留 `stream` 字段。
- 若多轮历史导致效果不稳定，先只支持空历史或单轮历史，后续再增强。

---

## Phase 4 临时文件问答能力

### 目标

实现文件上传后构建临时向量库，并通过 `/chat/rag` 的 `temp_kb` 路径进行问答。

### 前置依赖

- Phase 3 完成

### 任务清单

- [x] 实现 `POST /knowledge_base/upload` 的 `scope=temp` 路径。
- [x] 为上传文件分配临时 `knowledge_id`。
- [x] 实现临时目录保存逻辑。
- [x] 实现上传文件的切分、Embedding、临时 FAISS 构建。
- [x] 实现 `source_type=temp_kb` 的 `/chat/rag` 路径。
- [x] 返回临时文件问答结果与引用。
- [x] 约定临时数据生命周期和清理策略。

### 交付物

- 文件上传接口
- 临时知识库 ID
- 临时问答链路

### 完成标准

- 上传单个或多个文件后可立即问答。
- 临时问答不依赖长期知识库目录。
- 临时知识库与本地知识库路径在逻辑上隔离。

### 验证方式

- 上传 1 个文件并提问。
- 上传多个文件并提问。
- 检查是否返回临时知识库 ID。
- 检查 `/chat/rag` 的 `temp_kb` 路径能返回答案与引用。

### 阻塞后回退方案

- 若多文件合并逻辑复杂，先支持单文件临时问答。
- 若临时清理策略未定，先只做进程内临时缓存并记录后续清理 TODO。

---

## Phase 5 Agent 与工具调用

### 目标

实现 `POST /chat/agent`，并接入 `search_local_knowledge`、`calculate`、`current_time` 三个工具。

### 前置依赖

- Phase 3 完成

### 任务清单

- [x] 定义 `/chat/agent` 请求与响应结构。
- [x] 建立工具注册机制，至少包含：
  - 名称
  - 描述
  - 参数 schema
  - 调用函数
- [x] 实现 `search_local_knowledge` 工具：
  - 复用本地知识库检索能力
- [x] 实现 `calculate` 工具。
- [x] 实现 `current_time` 工具。
- [x] 实现 `GET /tools`。
- [x] 接入 Agent 执行器。
- [x] 返回工具调用记录：
  - 工具名
  - 参数
  - 输出
  - 状态

### 交付物

- `/chat/agent`
- 3 个核心工具
- `/tools`

### 完成标准

- 至少 1 个问题会触发工具调用。
- 工具调用记录可返回。
- `search_local_knowledge` 能正确复用本地知识库检索结果。

### 验证方式

- 提交需要知识检索的问题。
- 提交需要数学计算的问题。
- 提交需要当前时间的问题。
- 检查返回结果中是否包含工具调用过程。

### 阻塞后回退方案

- 若完整 Agent 执行器接入难度过高，先实现“指定工具调用”模式，再逐步放开自动工具选择。
- 若模型对工具调用表现不稳定，先减少工具数量，不增加额外工具。

---

## Phase 6 FastAPI 整理与可选 Streamlit 页面

### 目标

整理 API 路由边界，并补上最小可用的 Streamlit 页面。

### 前置依赖

- Phase 4 完成
- Phase 5 完成

### 任务清单

- [x] 固化 `chat` 路由：
  - `POST /chat/rag`
  - `POST /chat/agent`
- [x] 固化 `knowledge_base` 路由：
  - `POST /knowledge_base/upload`
  - `POST /knowledge_base/rebuild`
  - 可选 `GET /knowledge_base/list`
- [x] 固化 `tools` 路由：
  - `GET /tools`
- [x] 统一错误返回格式。
- [x] 增加最小 Streamlit 页面：
  - 知识库管理
  - RAG 问答
  - Agent 对话
- [x] 在页面中展示引用与工具调用记录。

### 交付物

- 稳定 API 路由
- 最小 Streamlit 页面

### 完成标准

- API 可独立使用。
- 页面可覆盖知识库管理、RAG、Agent 三类操作。
- 前后端最小联调可用。

### 验证方式

- 使用接口工具逐个调用关键 API。
- 启动 Streamlit 页面并完成三类操作。
- 检查页面是否能展示答案、引用、工具调用结果。

### 阻塞后回退方案

- 若 Streamlit 页面开发阻塞，不影响 API 收敛；先保证 API 独立可用。
- 若知识库列表接口暂未稳定，页面中先采用手工输入知识库名称。

---

## Phase 7 联调、测试与文档回填

### 目标

完成全链路联调、问题收敛和文档更新，让系统达到可演示、可复盘状态。

### 前置依赖

- Phase 6 完成

### 任务清单

- [x] 对照学习文档逐项核对已实现能力。
- [x] 运行三个核心场景联调：
  - 本地知识库问答
  - 临时文件问答
  - 带工具的 Agent 对话
- [x] 汇总已知问题：
  - 模型限制
  - Embedding 兼容性
  - 文件解析边界
  - 工具调用不稳定场景
- [x] 补齐运行说明。
- [x] 补齐测试结果与失败案例。
- [x] 在 TODO 文档中回填完成状态。

### 交付物

- 联调记录
- 测试结论
- 运行说明
- 已知问题列表

### 完成标准

- 三类核心场景全通过。
- TODO 文档能标记完成状态并回链到学习文档。
- 系统当前能力和限制都已被记录。

### 验证方式

- 逐条执行测试清单。
- 复查接口命名、配置命名和目录结构是否与学习文档一致。
- 复查 README 或运行说明是否足够支持再次复现。

### 阻塞后回退方案

- 若某一条高级能力不稳定，不阻塞整体交付；保留核心三场景，明确记录限制与后续修复计划。
- 若页面联调问题较多，先以 API 自测结果作为主验收依据。

---

## 4. 固定接口清单

以下接口和命名直接复用学习文档，不重新设计。

### CLI

- [x] `init`
- [x] `rebuild_kb`
- [x] `start_api`
- [x] `start_ui`

### HTTP

- [x] `POST /chat/rag`
- [x] `POST /chat/agent`
- [x] `POST /knowledge_base/upload`
- [x] `POST /knowledge_base/rebuild`
- [x] `GET /tools`

### 配置文件

- [x] `basic_settings.yaml`
- [x] `kb_settings.yaml`
- [x] `model_settings.yaml`

### 数据对象

- [x] `DocumentChunk`
- [x] `RetrievedDoc`
- [x] `ChatRequest`
- [x] `ChatResponse`
- [x] `ToolCallRecord`

---

## 5. 风险与阻塞项

### 模型与 Embedding 风险

- [ ] Ollama 中 LLM 模型可用，但 Embedding 模型兼容性不足。
- [ ] 不同模型的中文工具调用表现差异较大。
- [ ] Embedding 质量不足会直接拉低 RAG 效果。

### 文件解析风险

- [ ] PDF 解析质量不稳定。
- [ ] DOCX 解析依赖较多，环境差异大。
- [ ] 表格、扫描件、图片类文档暂不纳入第一阶段核心验收。

### Agent 风险

- [ ] 工具描述不准确会导致工具选择失败。
- [ ] 工具过多会显著降低稳定性。
- [ ] 先期不引入 `shell`、`sql`、联网搜索等高风险工具。

### 工程风险

- [ ] Phase 3 之前不要引入多模型平台切换。
- [ ] Phase 6 之前不要引入复杂前端。
- [ ] 不要在核心能力未闭环前扩展 OpenAI 兼容完整层。

---

## 6. 测试清单

## Phase 0-1 基础测试

- [x] 虚拟环境可激活。
- [x] 配置文件可加载。
- [x] `init` 可重复执行。
- [x] 数据目录完整创建。

## Phase 2 入库测试

- [x] `txt` 文档可入库。
- [x] `md` 文档可入库。
- [x] `pdf` 文档可入库。
- [x] `docx` 文档可入库。
- [x] FAISS 索引可持久化。

## Phase 3 本地 RAG 测试

- [x] `POST /chat/rag` 的 `local_kb` 可用。
- [x] 返回中包含引用。
- [x] 无命中时返回合理提示。
- [x] `top_k`、`score_threshold` 生效。

## Phase 4 临时文件问答测试

- [x] 上传单文件后可提问。
- [x] 上传多文件后可提问。
- [x] 能返回临时 `knowledge_id`。
- [x] `temp_kb` 路径与 `local_kb` 路径行为区分清楚。

## Phase 5 Agent 测试

- [x] `GET /tools` 可列出工具。
- [x] `search_local_knowledge` 可调用。
- [x] `calculate` 可调用。
- [x] `current_time` 可调用。
- [x] `/chat/agent` 至少能完成一次工具调用。

## Phase 6-7 联调测试

- [x] API 独立可用。
- [x] Streamlit 页面可访问。
- [x] 页面能完成知识库管理。
- [x] 页面能完成 RAG 问答。
- [x] 页面能完成 Agent 对话。
- [x] 三个核心场景全通过。

---

## 7. 执行记录模板

### Phase 执行记录模板

```markdown
## Phase X

- 开始时间：
- 完成时间：
- 当前状态：未开始 / 进行中 / 已完成 / 阻塞
- 实际完成内容：
- 未完成内容：
- 遇到的问题：
- 处理结果：
- 是否达到完成标准：是 / 否
- 下一步动作：
```

### 单次问题记录模板

```markdown
## 问题标题

- 所属阶段：
- 发现时间：
- 现象：
- 影响范围：
- 临时方案：
- 根因分析：
- 最终结论：
```

### 复盘要求

- [x] 每完成一个 Phase，回填一次执行记录。
- [x] 每出现一个阻塞问题，单独记录。
- [x] Phase 7 结束时输出最终复盘结论。

## Phase 0

- 开始时间：已执行
- 完成时间：已完成
- 当前状态：已完成
- 实际完成内容：
  - 创建 `app/`、`configs/`、`data/`、`scripts/` 骨架目录
  - 创建 `.venv` 并安装 Phase 0 依赖
  - 创建 `requirements.txt`
  - 创建三份配置模板
  - 创建 `scripts/activate.ps1`
  - 验证 Ollama 本地模型可见
- 未完成内容：
  - `init` 命令实现留到 Phase 1
- 遇到的问题：
  - `.venv` 初次创建时 `ensurepip` 失败
  - 默认未生成标准 `Activate.ps1`
- 处理结果：
  - 补全 `pip` 后完成依赖安装
  - 新增项目内 `scripts/activate.ps1` 作为统一激活入口
- 是否达到完成标准：是
- 下一步动作：
  - 进入 Phase 1，开始实现配置加载和初始化逻辑

## Phase 1

- 开始时间：已执行
- 完成时间：已完成
- 当前状态：已完成
- 实际完成内容：
  - 新增配置加载模块，统一读取 `basic_settings.yaml`、`kb_settings.yaml`、`model_settings.yaml`
  - 新增初始化服务，负责默认配置生成、配置校验、数据目录创建
  - 新增 `python .\scripts\init.py` 入口
  - 初始化输出已固定包含配置目录、数据目录、默认模型、默认向量库类型
  - 验证了当前目录幂等初始化和空目录初始化
- 未完成内容：
  - CLI 封装为统一命令留到后续阶段按需扩展
- 遇到的问题：
  - 空目录 smoke test 产生的临时目录清理时出现 Windows 权限异常
- 处理结果：
  - Phase 1 主能力不受影响，初始化逻辑验证已完成
  - 临时测试目录问题作为环境清理问题记录，不阻塞进入 Phase 2
- 是否达到完成标准：是
- 下一步动作：
  - 进入 Phase 2，开始实现本地知识库入库能力

## Phase 2

- 开始时间：已执行
- 完成时间：已完成
- 当前状态：已完成
- 实际完成内容：
  - 新增知识库目录约定与路径解析能力
  - 新增 `txt`、`md`、`pdf`、`docx` 文档加载能力
  - 新增中文友好文本切分逻辑
  - 接入 `OllamaEmbeddings`
  - 新增 FAISS 写入与 `save_local` 持久化
  - 新增 `metadata.json` 保存切片元数据
  - 新增 `python .\\scripts\\rebuild_kb.py --kb-name <name>` 入口
  - 对单个知识库 `phase2_demo` 完成实际重建验证
- 未完成内容：
  - 问答检索链路留到 Phase 3
- 遇到的问题：
  - PowerShell 在一次性 here-string 写测试数据时语法报错
  - 一次检查命令短暂出现向量库路径未找到
- 处理结果：
  - 修正测试数据写法后重建成功
  - 重新核对解析路径后确认实际向量库与元数据均已生成
- 是否达到完成标准：是
- 下一步动作：
  - 进入 Phase 3，开始实现本地 RAG 对话接口

## Phase 3

- 开始时间：已执行
- 完成时间：已完成
- 当前状态：已完成
- 实际完成内容：
  - 新增 `/chat/rag` 请求与响应模型
  - 新增本地 FAISS 检索服务和分数归一化逻辑
  - 新增 RAG Prompt 链和本地知识库问答生成逻辑
  - 新增 FastAPI 应用入口与 `/health`
  - 新增 `scripts/start_api.py`
  - 完成 `local_kb` 路径下的真实接口调用验证
- 未完成内容：
  - `temp_kb` 路径留到 Phase 4
  - 真正的流式输出留到后续阶段
- 遇到的问题：
  - FAISS 原始距离分数与配置中的 `score_threshold` 语义不一致
- 处理结果：
  - 将原始距离分数转换为 `relevance_score = 1 / (1 + distance)` 后再过滤
  - 保留 `stream` 字段，但当前统一返回非流式 JSON
- 是否达到完成标准：是
- 下一步动作：
  - 进入 Phase 4，开始实现临时文件问答能力

## Phase 4

- 开始时间：已执行
- 完成时间：已完成
- 当前状态：已完成
- 实际完成内容：
  - 新增 `POST /knowledge_base/upload` 的 `scope=temp` 上传路径
  - 上传后为文件分配临时 `knowledge_id`
  - 新增 `data/temp/<knowledge_id>/content` 与 `data/temp/<knowledge_id>/vector_store` 目录约定
  - 复用已有切分、Embedding、FAISS 入库链路构建临时向量库
  - 新增 `manifest.json` 与 `metadata.json` 保存临时知识库元信息
  - 扩展 `/chat/rag`，支持 `source_type=temp_kb` + `knowledge_id` 查询
  - 完成单文件上传问答和多文件上传问答两组真实接口验证
- 未完成内容：
  - 自动过期清理和 TTL 策略留到后续阶段再收敛
- 遇到的问题：
  - 临时知识库生命周期在 Phase 4 内不适合提前引入自动清理逻辑，否则会干扰接口联调
- 处理结果：
  - 先固定 `manual_cleanup_until_phase5` 清理策略
  - 先保证临时问答链路稳定可用，再把自动清理作为后续增强项
- 是否达到完成标准：是
- 下一步动作：
  - 进入 Phase 5，开始实现 Agent 与工具调用能力

## Phase 5

- 开始时间：已执行
- 完成时间：已完成
- 当前状态：已完成
- 实际完成内容：
  - 新增 `/chat/agent` 请求与响应模型
  - 新增 `search_local_knowledge`、`calculate`、`current_time` 三个工具及参数 schema
  - 新增工具注册表和自定义 Agent 执行器
  - 新增 `GET /tools` 路由
  - 扩展 `/chat/agent`，支持工具选择、工具执行、工具调用记录返回
  - 为 `DocumentChunk`、`RetrievedDoc`、`ToolCallRecord` 等固定数据对象补齐代码落点
  - 完成知识库检索、数学计算、当前时间三组真实接口验证
- 未完成内容：
  - 模型原生 tool calling 暂未启用，当前阶段使用稳定的自定义执行器
  - 多工具连续调用和复杂规划能力留到后续增强
- 遇到的问题：
  - `python -m compileall` 在当前 Windows 环境下写入既有 `__pycache__` 时出现权限错误
  - PowerShell here-string 内联测试脚本会把中文字面量转成 `?`，导致首次工具选择验证失真
- 处理结果：
  - 改用 `python -B` 做无字节码导入检查，确认导入与路由注册正常
  - 测试脚本改用 Unicode 转义后，`GET /tools`、`/chat/agent` 三类工具调用全部验证通过
- 是否达到完成标准：是
- 下一步动作：
  - 进入 Phase 6，开始整理 FastAPI 路由并补最小 Streamlit 页面

## Phase 6

- 开始时间：已执行
- 完成时间：已完成
- 当前状态：已完成
- 实际完成内容：
  - 固化 `chat`、`knowledge_base`、`tools` 三组 FastAPI 路由
  - 新增 `POST /knowledge_base/rebuild` 和 `GET /knowledge_base/list`
  - 新增统一错误响应结构与全局异常处理
  - 新增 `scripts/start_ui.py`
  - 新增最小 Streamlit 页面，覆盖知识库管理、RAG 对话、Agent 对话
  - 页面中已展示引用内容和工具调用记录
  - 完成 API 回归验证、Streamlit 静态渲染验证和带真实本地 API 的页面交互验证
- 未完成内容：
  - 更复杂的知识库长期文件上传管理仍可在后续阶段继续扩展
  - 更完整的页面样式和多页结构留到后续增强
- 遇到的问题：
  - `rag_chat` 中局部抛出的 `HTTPException` 一度被通用异常分支包成 `500`
  - `AppTest` 默认 3 秒超时，不足以覆盖实际 LLM 请求
  - PowerShell 内联脚本中的中文标签匹配再次受到编码问题影响
- 处理结果：
  - 单独放行 `HTTPException`，恢复统一错误结构下的正确状态码
  - 页面交互测试改用更长超时
  - 交互测试脚本统一改用 Unicode 转义，成功完成知识库刷新、RAG 请求和 Agent 请求验证
- 是否达到完成标准：是
- 下一步动作：
  - 进入 Phase 7，开始做最终联调、测试结论整理和文档回填

## Phase 7

- 开始时间：已执行
- 完成时间：已完成
- 当前状态：已完成
- 实际完成内容：
  - 对照学习文档逐项核对现有 CLI、HTTP、配置文件、数据对象和运行路径
  - 修正文档中的旧接口示例、旧模型名、旧 UI 路径和旧启动命令
  - 新增 `python .\\scripts\\validate_phase7.py` 项目级验证脚本
  - 完成三类核心场景的最终联调：
    - 本地知识库问答
    - 临时文件问答
    - 带工具的 Agent 对话
  - 完成 Streamlit 页面基础交互验证
  - 在学习文档中补齐运行说明、测试结果和当前版本限制
- 未完成内容：
  - 模型原生 tool calling 留作后续增强
- 遇到的问题：
  - 学习文档中的部分接口示例与当前落地实现已经不一致
  - PowerShell 内联脚本中的中文字符在自动化验证中仍可能失真
- 处理结果：
  - 已将学习文档对齐到当前代码实现
  - 最终验证脚本统一采用可复用的 Python 文件和 Unicode 转义，避免终端编码影响
- 是否达到完成标准：是
- 下一步动作：
  - 当前版本进入可演示、可复盘状态；后续如继续推进，可从多工具编排与执行器增强开始

## 问题标题：PowerShell 内联脚本中文编码失真

- 所属阶段：Phase 5 - Phase 7
- 发现时间：已处理
- 现象：通过 PowerShell here-string 把测试脚本送入 Python 时，中文字面量会被替换为 `?`
- 影响范围：自动化接口验证、Streamlit UI 交互测试
- 临时方案：将测试问题、按钮标签和字段名改用 Unicode 转义
- 根因分析：终端和 PowerShell 管道编码在内联脚本场景下不稳定
- 最终结论：最终以独立 Python 验证脚本为主，避免继续依赖 here-string 传递中文测试数据

## 问题标题：Windows 环境下 compileall 写入 __pycache__ 权限异常

- 所属阶段：Phase 5
- 发现时间：已处理
- 现象：执行 `python -m compileall .\\app` 时，多个模块在写入现有 `__pycache__` 时返回 `WinError 5`
- 影响范围：静态编译检查流程
- 临时方案：改用 `python -B` 做导入和接口 smoke test
- 根因分析：当前工作目录中的既有 `__pycache__` 文件在 Windows 下存在写入权限异常
- 最终结论：该问题不影响应用运行，Phase 7 最终验证统一采用无字节码导入检查和真实接口调用验证

## 问题标题：知识库重建接口 `read timeout=180`

- 所属阶段：Phase C 完成后运行期问题，已在 Next Phase G 完成首版根治
- 发现时间：已记录
- 现象：在页面点击“执行重建”或通过长连接调用 `/knowledge_base/rebuild` 时，前端报错 `知识库重建失败: HTTPConnectionPool(host='127.0.0.1', port=8000): Read timed out. (read timeout=180)`
- 影响范围：
  - Streamlit 页面中的知识库重建操作
  - 大型长期知识库上传后自动重建
  - 文档较多、PDF 较多或 embedding 较慢时的同步重建请求
- 临时方案：
  - 优先使用 CLI 执行重建：`python .\\scripts\\rebuild_kb.py --kb-name <knowledge_base_name>`
  - 不要每上传一个文件就自动重建，改为“批量上传一次 + 手动统一重建一次”
  - 适当增大 `chunk_size`、减小 `chunk_overlap`，减少 chunk 数量
  - 大型 PDF/Word 文档尽量先转成 `txt/md` 后再入库
  - 若必须走页面联调，可把 `app/ui/app.py` 中重建请求的超时时间调大到高于 `180`
- 根因分析：
  - 早期 `/knowledge_base/rebuild` 是同步全量重建链路
  - 每次都会重新读取全部文件、重新切片、重新调用 embedding、重新全量构建 FAISS
  - Streamlit 页面当前对重建请求使用固定 `timeout=180`
  - 当“解析 + 切片 + embedding + 建索引”总耗时超过 180 秒时，前端会先超时
- 最终结论：
  - 这个问题本质上是“全量同步重建 + 固定前端超时”共同导致的性能问题
  - Phase G 已落地首版长期方案：文件级增量检测、`sha256 + mtime + size` 清单缓存、chunk/embedding 缓存、`FAISS.add_embeddings(...)` 追加索引、批量 embedding、并行解析
  - 当前重复重建和“仅新增文件”场景的超时概率已明显下降
  - 首次全量构建或“大量修改/删除文件”仍可能耗时较长，这类场景依然建议优先用 CLI 重建

## 最终复盘结论

- 当前项目已经完成教学版定义的三个核心闭环：本地知识库问答、临时文件问答、带工具的 Agent 对话
- CLI、FastAPI、Streamlit 三层入口已经形成可复用的最小工程骨架
- 项目级验证脚本已经固化，后续继续扩展功能时可以复用同一套回归路径
- 当前版本最主要的非功能缺口不在“能不能用”，而在“复杂 Agent 规划”这类增强项

---

## 8. 最终验收门槛

- [x] 接口命名、配置命名、数据对象命名与学习文档一致。
- [x] 技术基线仍保持 `Windows + Ollama + FAISS + FastAPI + 可选 Streamlit`。
- [x] 本地知识库问答闭环已完成。
- [x] 临时文件问答闭环已完成。
- [x] 带工具的 Agent 对话闭环已完成。
- [x] TODO 文档可直接作为执行清单使用，不需要实现者再做阶段顺序或接口命名决策。

---

## 9. 后续增强执行计划

以下内容不属于当前已完成的 Phase 0 - Phase 7，而是当前版本之后的增强路线。建议按顺序推进，不要并行混做。

### Next Phase A 真实流式输出

#### 目标

让 `/chat/rag` 与 `/chat/agent` 从“保留 `stream` 字段”升级为真正可用的 SSE 流式输出。

#### 核心任务

- [x] 为 `/chat/rag` 增加真实流式响应实现。
- [x] 为 `/chat/agent` 增加真实流式响应实现。
- [x] 统一流式事件结构，至少区分：
  - token
  - reference
  - tool_call
  - done
  - error
- [x] 保证非流式模式继续兼容现有响应结构。
- [x] 更新 Streamlit 页面，支持逐步显示回答内容。

#### 完成标准

- `stream=true` 时可持续收到增量输出，不再等待整段回答完成后一次性返回。
- RAG 场景下引用信息仍可回传。
- Agent 场景下工具调用过程可在流式过程中展示。

#### 执行结果

- [x] `POST /chat/rag` 在 `stream=true` 下已返回 `reference -> token -> done` 事件序列。
- [x] `POST /chat/agent` 在 `stream=true` 下已返回 `tool_call -> step -> token -> done`，多步场景下还可包含 `reference` 与第二次 `tool_call`。
- [x] Streamlit 页面默认开启流式输出，并可逐步显示回答内容。
- [x] `python .\\scripts\\validate_phase7.py` 已纳入流式 API 检查。

### Next Phase B 临时知识库自动清理

#### 目标

把当前 `manual_cleanup_until_phase5` 的手工清理策略升级为可配置、可观测的自动清理机制。

#### 核心任务

- [x] 为临时知识库增加过期时间配置，例如 TTL 分钟数。
- [x] 在 `manifest.json` 中记录创建时间、最后访问时间、过期时间。
- [x] 增加清理任务入口，支持手动触发和启动时清理。
- [x] 清理 `data/temp/<knowledge_id>/content` 与 `vector_store` 整体目录。
- [x] 增加安全检查，避免误删非临时目录。

#### 完成标准

- [x] 过期临时知识库可自动清理。
- [x] 未过期的临时知识库不会被误删。
- [x] 清理后查询过期 `knowledge_id` 时返回明确错误。

#### 执行结果

- [x] `kb_settings.yaml` 已新增 `TEMP_KB_TTL_MINUTES`、`TEMP_KB_CLEANUP_ON_STARTUP`、`TEMP_KB_TOUCH_ON_ACCESS`。
- [x] 临时知识库 `manifest.json` 已记录 `created_at`、`last_accessed_at`、`expires_at`、`ttl_minutes`。
- [x] 已新增 `python .\\scripts\\cleanup_temp_kb.py` 手动清理入口。
- [x] `start_api.py` 与 `start_ui.py` 已在启动时执行过期临时知识库清理。
- [x] `/chat/rag` 访问过期 `knowledge_id` 时返回 `410 temp_knowledge_expired`。
- [x] `python .\\scripts\\validate_phase7.py` 已覆盖“未过期不删、手动清理、启动清理、过期访问报错”四类校验。
- [x] Windows 下若目录被系统占用，当前实现会退化为“软清理 + 后续启动继续重试物理删除”，保证临时知识库已不可访问。

### Next Phase C 长期知识库上传管理

#### 目标

补齐长期知识库文件上传和管理能力，让 `POST /knowledge_base/upload` 同时支持 `scope=local`。

#### 核心任务

- [x] 扩展 `POST /knowledge_base/upload`，支持 `scope=local`。
- [x] 支持上传到 `data/knowledge_base/<knowledge_base_name>/content/`。
- [x] 增加知识库文件覆盖、重名处理和基础校验策略。
- [x] 在上传后支持按需触发自动重建或提示调用 `/knowledge_base/rebuild`。
- [x] 在 Streamlit 的知识库管理页补上长期知识库上传入口。

#### 完成标准

- [x] 用户不需要手工复制文件到项目目录，也能完成长期知识库入库。
- [x] 上传后的知识库可以通过 `/knowledge_base/rebuild` 完成索引更新。
- [x] 页面中可以完成长期知识库文件上传和重建闭环。

#### 执行结果

- [x] `POST /knowledge_base/upload` 已同时支持 `scope=temp` 和 `scope=local`。
- [x] `scope=local` 时会把文件写入 `data/knowledge_base/<knowledge_base_name>/content/`。
- [x] 已支持 `overwrite_existing` 和 `auto_rebuild` 两个表单参数。
- [x] 重名文件在不覆盖模式下会进入 `skipped_files`，覆盖模式下会进入 `overwritten_files`。
- [x] 上传后可直接选择自动重建；若未重建，响应会通过 `requires_rebuild=true` 提示后续操作。
- [x] Streamlit 的知识库管理页已新增长期知识库上传表单。
- [x] `python .\\scripts\\validate_phase7.py` 已覆盖“长期知识库上传 -> 自动重建 -> 本地 RAG 问答 -> 重名跳过”完整链路。

### Next Phase D 多工具连续编排

#### 目标

把当前“单次问题最多选择一个工具”的稳定执行器升级为支持多工具连续调用的 Agent 编排能力。

#### 核心任务

- [x] 为 Agent 增加多步中间状态记录。
- [x] 支持单轮问题内连续调用多个工具。
- [x] 支持“知识检索 -> 计算”这类串行工具链。
- [x] 为工具调用增加最大步数和防死循环保护。
- [x] 在响应中返回完整中间步骤轨迹。
- [x] 在 UI 中按顺序展示多步工具调用过程。

#### 完成标准

- [x] 至少支持一个需要两个工具连续执行的真实问题。
- [x] 工具调用过程和最终答案都能正确返回。
- [x] 出现异常或超步数时能安全终止，不进入死循环。

#### 执行结果

- [x] Agent 请求模型已新增 `max_steps`，响应已新增 `steps` 中间步骤轨迹。
- [x] 自定义执行器已改为多步循环，不再限制“单轮问题最多一个工具”。
- [x] 已支持 `search_local_knowledge -> calculate` 串行工具链。
- [x] 已增加重复调用检测与 `max_steps` 超步数保护。
- [x] `/chat/agent` 流式模式已新增 `step` 事件，并在 `done` 中返回完整 `steps`。
- [x] Streamlit Agent 页面已可展示多步工具调用记录和步骤轨迹。
- [x] `python .\\scripts\\validate_phase7.py` 已覆盖“多步检索计算、max_steps 终止、流式多步事件”三类校验。

### Next Phase E 企业级 RAG 检索优化

#### 目标

把当前 RAG 从“单路向量检索 + 直接拼接上下文”升级为更稳定的企业级黄金组合：查询改写 + 混合检索 + 重排序 + 小块查大块答。

#### 核心任务

- [x] 增加查询改写服务，对口语化问题做检索友好的短查询改写。
- [x] 在原有 `FAISS` 语义检索之外，增加基于本地 `docstore` 的词法检索。
- [x] 用 RRF 融合向量召回和词法召回。
- [x] 增加启发式重排序，利用覆盖率、短语匹配、型号标准化匹配等特征筛掉干扰项。
- [x] 基于 `doc_id + chunk_index` 把命中的小块扩展为更完整的大块上下文。
- [x] 保持 `/chat/rag` 与 `search_local_knowledge` 工具接口不变，只替换底层检索实现。
- [x] 为优化链路补充独立回归验证。

#### 完成标准

- [x] 用户口语化提问时，系统可先完成查询改写再进入检索。
- [x] 检索链路不再只依赖单一向量相似度。
- [x] 型号、货号、专有名词场景下，召回稳定性优于纯向量检索。
- [x] 回答阶段不再只喂单个小块，而是返回扩展后的大块引用。
- [x] 原有 API、Agent 工具和 UI 不需要改接口即可复用新链路。

#### 执行结果

- [x] `kb_settings.yaml` 已新增 `ENABLE_QUERY_REWRITE`、`ENABLE_HYBRID_RETRIEVAL`、`ENABLE_HEURISTIC_RERANK`、`ENABLE_SMALL_TO_BIG_CONTEXT` 及相关参数。
- [x] `model_settings.yaml` 已新增 `QUERY_REWRITE_MODEL`。
- [x] 已新增 `app/services/query_rewrite_service.py`，用于检索前的查询改写。
- [x] `app/retrievers/local_kb.py` 已改为统一的优化检索入口，支持本地知识库和临时知识库。
- [x] `/chat/rag` 已接入基于 `history` 的检索改写路径。
- [x] `python .\\scripts\\validate_phase7.py` 已新增 `phase7_rag_optimized_demo` 校验，验证 `ZX900 -> ZX-900` 命中和扩展上下文返回。

### Next Phase F 独立 Reranker 模型接入

#### 目标

把当前“启发式重排序”为主的链路升级为“模型重排序优先，启发式兜底”的双层方案，同时保留现有 `/chat/rag` 和 `search_local_knowledge` 接口不变。

#### 核心任务

- [x] 在配置层新增独立 `RERANK_MODEL`、`RERANK_DEVICE` 以及 Rerank 开关参数。
- [x] 新增可选的模型重排服务层，支持 `sentence-transformers` `CrossEncoder`。
- [x] 在检索器中接入“启发式排序 -> 模型重排 -> 回退启发式”的执行路径。
- [x] 保持模型依赖缺失时自动回退，不能影响当前项目可用性。
- [x] 新增最小检索评测数据集与评测脚本。
- [x] 在验证脚本中增加 Phase F 的可选依赖探测和回退校验。
- [x] 安装并启用真实 `bge-reranker-base` 模型后完成效果实测。
- [ ] 补充 10 到 20 条更完整的企业检索评测样本。

#### 完成标准

- [x] 配置文件已支持控制模型重排开关和候选数。
- [x] 在未安装 `sentence-transformers` 的环境下，系统仍可稳定查询。
- [x] 已有一份可执行的检索评测脚本，可输出 `Hit@K`、`MRR`、`Top1 source accuracy`。
- [x] 在启用真实 reranker 后，至少 1 组 case 的首条命中优于当前启发式排序。
- [x] `ENABLE_MODEL_RERANK=true` 在真实模型环境下完成全量回归。

#### 执行结果

- [x] `kb_settings.yaml` 已新增 `ENABLE_MODEL_RERANK`、`RERANK_CANDIDATES_TOP_N`、`RERANK_SCORE_THRESHOLD`、`RERANK_FALLBACK_TO_HEURISTIC`。
- [x] `model_settings.yaml` 已新增 `RERANK_MODEL`、`RERANK_DEVICE`，当前默认指向本地目录 `./data/models/bge-reranker-base`。
- [x] 已新增 `app/services/rerank_service.py`，封装可选 `CrossEncoder` 重排和依赖缺失回退。
- [x] `app/retrievers/local_kb.py` 已接入 Phase F 的重排调用路径。
- [x] 已新增 `data/eval/rag_eval.jsonl` 与 `python .\\scripts\\eval_retrieval.py`。
- [x] `python .\\scripts\\validate_phase7.py` 已新增 `model_rerank_probe`、`model_rerank_pair_probe` 和真实本地 reranker 检查。
- [x] 已在当前 `.venv` 中安装 `sentence-transformers`、`torch`，并成功加载本地 `bge-reranker-base` 权重。
- [x] `python .\\scripts\\eval_retrieval.py --compare-model-rerank --show-cases` 已验证：
  - 启发式：`Top1 source accuracy = 0.8571`，`MRR = 0.9286`
  - 模型重排：`Top1 source accuracy = 1.0`，`MRR = 1.0`
- [x] `phasef_reranker_demo` 已验证“启发式首条命中错误，模型重排后修正为 `warranty.txt`”。

### 推进顺序建议

- [x] 先做 `真实流式输出`，因为它影响 API 和 UI 的交互体验。
- [x] 再做 `临时知识库自动清理`，因为它属于当前实现的运行时稳定性缺口。
- [x] 然后做 `长期知识库上传管理`，补齐知识库生命周期管理。
- [x] 最后做 `多工具连续编排`，因为它对执行器稳定性要求最高。
- [x] 当前版本已补充 `企业级 RAG 检索优化`，后续若再继续增强，应优先考虑独立 Reranker 模型与召回缓存。
- [x] `独立 Reranker 模型接入` 已完成；后续优先项改为扩充评测集和加入检索日志观测。

### Next Phase G 知识库增量重建与性能优化

#### 目标

把当前“每次全量读取、全量切片、全量 embedding、全量重建 FAISS”的知识库重建链路，升级为“增量检测 + 缓存复用 + 追加索引优先”的高性能重建方案，并从根本上降低 `read timeout=180` 类问题的出现概率。

#### 核心任务

- [x] 增加增量重建能力，只处理新增或修改过的文件。
- [x] 为每个源文件记录 `relative_path + size + mtime + sha256`，未变化文件直接跳过。
- [x] 增加 chunk 级缓存，复用未变化文件的切片结果。
- [x] 增加 embedding 缓存，未变化 chunk 不重复请求 embedding 模型。
- [x] 对“仅新增文件”场景优先采用 `FAISS.add_embeddings(...)` 追加索引。
- [x] 对“修改/删除文件”场景采用安全回退策略，必要时触发受控全量重建。
- [x] 增加批量 embedding 和并行解析能力，降低大文档重建耗时。
- [x] 为重建过程输出更细的统计信息，例如 `files_reused`、`files_rebuilt`、`chunks_reused`、`chunks_embedded`、`index_mode`。

#### 完成标准

- [x] 未变化文件不会重复解析、切片、embedding。
- [x] 新增文件支持追加索引，不再总是全量重建。
- [x] 修改/删除文件时不会产生脏索引。
- [x] 同一知识库第二次重建显著快于第一次。
- [x] 重复重建和“仅新增文件”场景下，页面和 API 的超时概率已明显下降。

#### 执行建议

- [x] 第一版先做“文件级增量检测 + 文件哈希缓存 + 新增文件追加索引”。
- [x] 第二版再做“chunk 缓存 + embedding 缓存”。
- [x] 第三版再做“批量 embedding + 并行解析 + 更细的重建日志”。

#### 执行结果

- [x] `kb_settings.yaml` 已新增 `ENABLE_INCREMENTAL_REBUILD`、`ENABLE_FILE_HASH_CACHE`、`ENABLE_CHUNK_CACHE`、`ENABLE_APPEND_INDEX`、`EMBEDDING_BATCH_SIZE`、`DOC_PARSE_WORKERS`。
- [x] `settings.py` 已新增 `build_manifest.json` 与 `cache/chunks/` 路径辅助函数。
- [x] 已新增 `app/services/kb_incremental_rebuild.py`，封装增量检测、文件清单、chunk/embedding 缓存、批量向量化和索引模式选择。
- [x] `app/services/kb_ingestion_service.py` 已切换为统一的增量优先重建入口；原有 API、脚本和上传自动重建接口保持不变。
- [x] 当前重建结果会返回 `build_manifest_path`、`incremental_rebuild`、`index_mode`、`files_reused`、`files_rebuilt`、`files_deleted`、`chunks_reused`、`chunks_embedded`。
- [x] 当前已支持三种索引模式：
  - `full`：首次构建、配置变化、文件修改/删除时的安全回退
  - `reuse`：文件完全未变化时直接复用已有索引与缓存
  - `append`：仅新增文件时在现有索引上追加
- [x] `python .\\scripts\\validate_phase7.py` 已新增 `incremental_rebuild_full`、`incremental_rebuild_reuse`、`incremental_rebuild_append`、`incremental_rebuild_modified_full` 校验。
- [x] 已在 `phaseg_incremental_demo` 和 `phaseg_incremental_demo_validation` 上实际验证：
  - 首次构建返回 `index_mode=full`
  - 二次无变更重建返回 `index_mode=reuse`
  - 新增文件后返回 `index_mode=append`
  - 修改文件后安全回退为 `index_mode=full`，同时继续复用未变化文件缓存

---

## 7. 通用多模态 RAG 增强路线

### 路线说明

Phase 0 到 Phase 7 解决的是“RAG + Agent 基础闭环”，后续增强不再以“能不能跑通”为主，而转向“多模态质量、检索稳定性、回答可解释性、评测可量化”四个方向。

执行原则如下：

- 先做入库结构化，再做检索增强，不反过来。
- 先让多模态证据可分辨、可追踪，再追求更复杂的回答生成。
- 先补评测与观测，再扩展更多模态或更复杂的模型接入。
- 新增强化阶段默认不破坏现有 `/chat/rag`、`/chat/agent`、`/knowledge_base/*` 接口。

建议推进顺序：

- `Next Phase H` 多模态内容结构化入库
- `Next Phase I` 分路召回与跨模态融合检索
- `Next Phase J` 证据感知回答与可解释引用
- `Next Phase K` 多模态评测与检索观测
- `Next Phase L` 产品化能力与知识库配置分层

### 当前基线判断

当前项目已具备以下多模态基础：

- [x] 图片 OCR 入库
- [x] 图片视觉描述入库
- [x] 图片格式知识库上传与重建
- [x] 扫描类 PDF 的 OCR 预处理脚本
- [x] `epub` 文件读取与入库

当前主要不足：

- [x] 图片 OCR、视觉描述、原始文件信息仍以“拼接文本”为主，字段化程度不足。
- [x] 检索阶段尚未显式区分文本、OCR、视觉描述三类证据来源。
- [x] 回答阶段尚未按模态组织上下文，也未显式输出证据类型与不确定性。
- [x] 缺少针对图文联合问答的专项评测集与日志观测。

### Next Phase H 多模态内容结构化入库

#### 目标

把当前“文档正文 + OCR + 视觉描述拼成一段文本再入库”的方式，升级为“按模态拆字段、按来源保留证据、按文件类型走不同入库分支”的结构化入库方案。

#### 核心任务

- [x] 为知识块元数据增加多模态字段：
  - `content_type`
  - `source_modality`
  - `ocr_text`
  - `image_caption`
  - `original_file_type`
  - `evidence_summary`
- [x] 调整图片入库逻辑，区分：
  - 原始图片占位信息
  - OCR 文本
  - 视觉描述
  - 融合摘要
- [x] 对 PDF、DOCX、EPUB 补充更稳定的结构化字段：
  - `title`
  - `section_title`
  - `section_path`
  - `page/page_end`
  - `headers`
- [ ] 为不同文件类型保留独立 loader 输出风格，不强行统一为同一种文本模板。
- [x] 为多模态 chunk 增加统一的“证据预览”字段，方便检索后展示。
- [ ] 保持现有向量库存储接口兼容，避免一次性重写底层存储层。

#### 完成标准

- [x] 任意一个图片类 chunk 都能区分“来自 OCR”还是“来自视觉描述”。
- [x] 任意一个文档类 chunk 都能稳定返回标题、章节或页码中的至少一种结构信息。
- [x] `metadata.json` 中可直接看出每条 chunk 的证据来源和模态类型。
- [x] 不修改前端接口的情况下，现有重建和问答流程仍可运行。

#### 验证方式

- [x] 抽查 `metadata.json`，确认图片和文档 chunk 的关键字段存在。
- [x] 用图片知识库执行一次重建，检查 OCR 和视觉描述是否被分字段保留。
- [x] 用 `pdf/docx/epub` 各准备一个样例，检查章节元数据是否稳定。

#### 阻塞后回退方案

- [ ] 若一次性改动元数据范围过大，先只对图片类 chunk 增加 `source_modality`、`ocr_text`、`image_caption` 三个字段。
- [ ] 若历史 `metadata.json` 兼容性不足，先在新知识库上启用新字段，不强制迁移旧索引。

### Next Phase I 分路召回与跨模态融合检索

#### 目标

在现有混合检索基础上，增加“按模态分路召回、按问题类型动态融合”的能力，让系统不再把所有内容当成同一种文本处理。

#### 核心任务

- [x] 为检索器增加多路候选池：
  - 文档正文召回
  - OCR 文本召回
  - 视觉描述召回
- [x] 为查询增加轻量分类：
  - 文本知识问题
  - 图片内容问题
  - 图文联合问题
- [x] 根据查询类型动态调整召回权重，而不是固定使用同一套权重。
- [x] 在融合阶段保留命中来源，支持输出：
  - 命中文本证据
  - 命中 OCR 证据
  - 命中视觉描述证据
- [x] 为图片类问题增加查询扩展逻辑，例如从用户问题中提取 OCR 关键词、对象词、状态词。
- [x] 补充文件类型 / 模态类型 / 标题路径过滤能力，减少无关 chunk 干扰。

#### 完成标准

- [ ] 检索结果中可区分不同模态来源，不再只返回一组混合 chunk。
- [ ] 图片相关问题时，OCR 和视觉描述命中率高于正文通道。
- [ ] 图文联合问题时，至少可同时返回两类不同来源的证据。
- [ ] 保持现有 `search_local_knowledge` 以及 `/chat/rag` 接口不变。

#### 验证方式

- [ ] 准备 5 到 10 条文本问题、图片问题、图文联合问题各自测试。
- [ ] 记录各类问题下命中的模态分布是否符合预期。
- [ ] 对比改造前后，同一图片类问题的 Top1 / Top3 命中稳定性。

#### 阻塞后回退方案

- [ ] 若一次性做动态权重过于复杂，先实现“按 `source_modality` 分组召回后简单合并”。
- [ ] 若图像检索暂时没有独立 embedding，先基于 OCR / caption 文本分路召回。

#### 执行结果

- [x] `app/retrievers/local_kb.py` 已新增基于 `source_modality` 的文档分组逻辑。
- [x] 当知识库中 `source_modality` 覆盖率足够高时，向量召回会按模态分组执行并融合；旧索引则自动回退到原始全局召回。
- [x] 词法召回已支持按模态分组建立候选池并合并到统一排序链路。
- [x] 检索器已增加轻量查询模态画像，用于区分“文本知识问题”和“图片相关问题”。
- [x] `RetrievedReference` 已可返回 `source_modality`、`content_type`、`image_caption`、`ocr_text` 等字段，便于后续 UI 和回答阶段复用。
- [x] 已增加图片类问题的查询扩展逻辑，会自动补充 `图片内容 / 图像描述 / 图片文字` 等检索词。
- [x] 已在图片 loader 中过滤明显无效的拒答式视觉描述，避免把低质量 caption 写入知识库。
- [x] 已在排序阶段加入文件类型、模态类型和标题路径命中偏置，用于降低无关 chunk 干扰。

### Next Phase J 证据感知回答与可解释引用

#### 目标

让回答阶段显式区分“直接证据”和“模型推断”，并能按模态组织上下文，减少多模态混合后的幻觉与证据丢失。

#### 核心任务

- [x] 为回答上下文增加分段组织：
  - 文本证据
  - OCR 证据
  - 视觉描述证据
- [x] 调整 RAG Prompt，要求模型：
  - 优先引用直接证据
  - 不混淆 OCR 识别结果与视觉推断
  - 证据不足时明确说明不确定
- [x] 为返回结果增加证据标识，至少可标明：
  - `evidence_type`
  - `source_modality`
  - `used_for_answer`
- [x] 在 UI 中区分展示不同模态的引用，不再全部按统一文本展开。
- [x] 对图片类问题补充专门提示词模板，不与普通文档问答完全共用。

#### 完成标准

- [ ] 回答中能够明确区分“可以确认的内容”和“可能的推断”。
- [ ] 引用区可看出每条证据来自正文、OCR 或视觉描述。
- [ ] 图片相关问题下，回答误把 OCR 错字当事实的情况明显减少。

#### 验证方式

- [ ] 选取 5 条图片问答样例，人工检查回答与引用的一致性。
- [ ] 检查模型在证据不足时是否会明确输出“不确定”。
- [ ] 检查 UI 中是否能按模态区分引用展示。

#### 阻塞后回退方案

- [ ] 若 UI 改造暂时不做，先在 API 返回里增加 `source_modality` 与 `evidence_type` 字段。
- [ ] 若完整多模板 Prompt 管理复杂，先只为图片类问题增加单独系统提示词。

#### 执行结果

- [x] `app/chains/rag.py` 已将回答上下文从平铺引用改为按“文本证据 / OCR 证据 / 视觉描述证据”分段组织。
- [x] 每条上下文块现会携带 `source`、`section_title`、`source_modality`、`content_type`、`evidence_summary` 等信息。
- [x] RAG 系统提示词已补充“区分直接证据与推断、不要混淆 OCR 与视觉描述”的约束。
- [x] 当前未改动 `/chat/rag` 接口结构，已存在的 `source_modality / ocr_text / image_caption / evidence_summary` 字段继续透传到回答链路。
- [x] `RetrievedReference` 已新增 `evidence_type` 与 `used_for_answer` 字段，检索返回会直接标明证据类型与是否参与回答。
- [x] `Streamlit` 引用面板已展示 `evidence_type / used_for_answer / source_modality`，便于人工核查证据来源。
- [x] 回答链路已新增图片类问题专用系统提示词，并依据查询特征与证据模态分布在“普通 RAG Prompt / 图片类 RAG Prompt”之间自动切换。

### Next Phase K 多模态评测与检索观测

#### 目标

补齐多模态 RAG 的评测和观测能力，让系统优化从“主观感觉”转成“可量化迭代”。

#### 核心任务

- [x] 新增通用多模态评测集，至少覆盖：
  - 文本问答
  - 图片 OCR 问答
  - 图片描述问答
  - 图文联合问答
- [x] 为评测脚本增加指标：
  - `Hit@K`
  - `MRR`
  - `Top1 source accuracy`
  - `evidence_type accuracy`
- [x] 为检索阶段增加日志观测：
  - 查询类型
  - 各路召回数量
  - 融合后排序
  - 最终命中模态
- [x] 为回答阶段增加基础观测：
  - 是否使用 OCR 证据
  - 是否使用视觉描述证据
  - 是否命中多模态混合上下文
- [x] 整理失败案例库，记录典型问题：
  - OCR 噪声过大
  - 图片描述偏泛化
  - 图文证据冲突

#### 完成标准

- [x] 至少有一份可重复执行的多模态评测集和脚本。
- [x] 关键多模态问题的检索命中情况可被日志复盘。
- [x] 可以从日志中定位问题是出在“入库、检索、重排还是回答”。

#### 验证方式

- [x] 执行多模态评测脚本并输出对比结果。
- [x] 选取 3 个失败样例，检查是否能通过日志定位链路问题。
- [x] 对比一次优化前后，确认至少有一项指标出现提升。

#### 阻塞后回退方案

- [ ] 若完整评测集暂时不够，先从 10 到 20 条高质量样本起步。
- [ ] 若日志量过大，先只记录开发模式下的详细检索日志。

#### 执行结果

- [x] `scripts/eval_retrieval.py` 已升级为可评估多模态 case，支持 `expected_source`、`expected_evidence_type`、`expected_source_modality` 与 `expected_modalities_present`。
- [x] 已新增 `data/eval/multimodal_rag_eval.jsonl`，覆盖文本问答、图片 OCR、图片描述、图文联合四类样例。
- [x] 已新增 `data/eval/multimodal_failure_cases.jsonl`，记录当前已知弱项与失败原因。
- [x] `app/retrievers/local_kb.py` 已增加 `retrieval_trace.jsonl` 观测，当前会记录查询类型、可用模态、候选数、最终证据模态分布等信息。
- [x] `app/chains/rag.py` 已增加 `answer_trace.jsonl` 观测，当前会记录 `prompt_kind`、证据类型分布、是否命中 OCR / 视觉描述上下文等信息。
- [x] `python .\\scripts\\eval_retrieval.py --case-file .\\data\\eval\\multimodal_rag_eval.jsonl` 已完成实测，当前结果为：
  - `Hit@K = 0.75`
  - `MRR = 0.6875`
  - `Top1 source accuracy = 0.625`
  - `Top1 evidence_type accuracy = 0.8333`
  - `Top1 source_modality accuracy = 0.2`
  - `modality_presence_accuracy = 0.0`
- [x] 当前评测已明确暴露：文本问答稳定，但 `image_vision` 与 `multimodal_joint` 仍是主要短板，后续优先优化方向已清晰。
- [x] 已继续基于评测结果优化图片类召回，新增图片问句识别修正、非文本查询的重排候选放宽，以及图片模态保底覆盖逻辑。
- [x] 二次评测结果显示：
  - `MRR` 从 `0.6875` 提升到 `0.75`
  - `Top1 source accuracy` 从 `0.625` 提升到 `0.75`
  - `Top1 evidence_type accuracy` 从 `0.8333` 提升到 `1.0`
  - `Top1 source_modality accuracy` 从 `0.2` 提升到 `0.4`
- [x] 当前 `image_ocr` 与 `image_vision` 已可稳定把图片证据顶到 Top1。
- [x] 当前 `multimodal_joint` 仍未提升，已确认主要瓶颈不是“没有图片候选”，而是样例图片本身几乎没有有效 OCR / caption，导致图文联合语义仍偏向正文证据。

### Next Phase L 产品化能力与知识库配置分层

#### 目标

把多模态 RAG 从“实验性闭环”继续推进到“可复用通用框架”，重点补齐知识库策略配置、上传体验和可观测性。

#### 核心任务

- [ ] 支持按知识库配置不同的入库策略：
  - 是否启用 OCR
  - 是否启用视觉描述
  - 切分策略
  - 检索权重
- [ ] 支持上传后自动识别文件类型并走对应入库分支。
- [ ] 在 UI 中展示知识库构建状态：
  - 文件数
  - chunk 数
  - OCR / VLM 使用情况
  - 最后重建时间
- [ ] 在 UI 中增加检索解释信息：
  - 命中了哪些模态
  - 为什么排前面
  - 哪些证据被回答阶段使用
- [ ] 为不同知识库保留独立评测配置与回归结果。

#### 完成标准

- [ ] 不同知识库可采用不同多模态策略，而不是共享同一套固定开关。
- [ ] 用户在 UI 中可以看出知识库是否启用了 OCR、VLM 和多模态检索。
- [ ] 关键检索与回答决策有基本可解释性展示。

#### 验证方式

- [ ] 创建两个知识库，分别配置不同的多模态参数并验证生效。
- [ ] 检查页面是否能展示构建状态和检索解释信息。
- [ ] 检查配置修改后是否不需要改动核心业务代码。

#### 阻塞后回退方案

- [ ] 若知识库级配置改造较大，先保留全局配置，同时为单个知识库增加可选覆盖字段。
- [ ] 若 UI 工作量过大，先在 API 返回中暴露必要状态，再逐步补页面展示。

### 多模态增强阶段测试清单

#### Phase H 结构化入库测试

- [ ] 图片知识入库后，`metadata.json` 中包含 `source_modality`。
- [ ] 图片知识入库后，OCR 与视觉描述字段可分别查看。
- [ ] `pdf/docx/epub` 样例中至少有一类结构化标题字段稳定返回。

#### Phase I 检索测试

- [ ] 文本问题主要命中文本正文通道。
- [ ] 图片问题主要命中 OCR 或视觉描述通道。
- [ ] 图文联合问题可同时返回不同模态证据。

#### Phase J 回答测试

- [ ] 回答中可区分“直接证据”和“推断结论”。
- [ ] 引用展示中能区分不同模态来源。
- [ ] 证据不足时，模型会明确提示不确定。

#### Phase K 评测与观测测试

- [x] 多模态评测脚本可重复执行。
- [x] 检索日志可定位失败样例的主要问题阶段。
- [x] 至少一项检索指标或证据指标可在优化后提升。

#### Phase L 产品化测试

- [ ] 不同知识库策略可独立生效。
- [ ] UI 或 API 可展示多模态构建状态与检索解释。
- [ ] 配置修改不会破坏现有主链路。
