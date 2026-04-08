# 基于LangChain的RAG项目介绍

## 1. 项目概述

`Mini-Agent-RAG2` 是一个基于 `LangChain` 构建、面向学习与实验的本地 RAG + Agent 项目。它以 `FastAPI` 作为后端接口层，以 `Streamlit` 作为可视化控制台，围绕“知识库构建、检索增强问答、工具调用式 Agent、多模态文件入库”四条主线组织代码。

这个项目不是大型平台化框架，而是一个偏教学、偏实验性质的可运行样例。它保留了完整系统最关键的闭环：

- 本地知识库上传、重建与查询
- 临时文件上传后即时问答
- RAG 问答与流式输出
- 多工具 Agent 闭环执行
- 图片 OCR / 视觉描述后再入库
- `epub` 在内的多格式文档接入
- 增量重建、混合检索、启发式 / 模型重排
- 多模态结构化元数据、按模态分路召回、证据感知回答

如果从学习角度看，这个项目适合回答两个问题：

- 一个最小但不失工程结构的 RAG 系统应该怎么拆模块？
- 在本地环境里，如何把知识库、向量检索、LLM 和 Agent 能力串起来？

## 2. 技术栈

项目当前核心依赖如下：

- 后端接口：`FastAPI`、`Uvicorn`
- 前端界面：`Streamlit`
- 大模型编排：`LangChain`
- 向量检索：`FAISS`
- 模型接入：`Ollama`、OpenAI-compatible API
- 文档解析：`PyPDF`、`python-docx`、`docx2txt`、`markdown`、`beautifulsoup4`、`ebooklib`
- OCR / 图片处理：`pytesseract`、`Pillow`、`PyMuPDF`，并已接入 `PaddleOCR` 双后端抽象
- 重排模型：`sentence-transformers`

从依赖可以看出，这个项目默认优先支持本地部署和本地数据处理，同时保留接入外部兼容接口的能力。

## 3. 项目定位

项目的能力边界比较清晰：

- 它是“本地知识库问答 + 轻量 Agent”的实验系统
- 它强调本地文件入库、索引构建和问答闭环
- 它支持一定程度的多模态扩展，但重点仍然是文档型 RAG
- 它不是通用工作流平台，也不是复杂自治智能体系统

与成熟框架相比，这个项目的特点是结构清楚、代码集中、便于阅读和二次改造。

## 4. 目录结构

项目主要目录如下：

```text
Mini-Agent-RAG2/
├─ app/
│  ├─ agents/          # Agent 执行与多步工具编排
│  ├─ api/             # FastAPI 路由与流式响应
│  ├─ chains/          # RAG prompt 与文本切分
│  ├─ loaders/         # 各类文件加载与解析
│  ├─ retrievers/      # 本地知识库 / 临时知识库检索
│  ├─ schemas/         # 请求响应模型定义
│  ├─ services/        # 配置、嵌入、重排、入库、VLM 等服务
│  ├─ storage/         # 向量库适配与元数据过滤
│  ├─ tools/           # Agent 工具注册表
│  └─ ui/              # Streamlit 页面
├─ configs/            # YAML 配置文件
├─ data/               # 数据目录、知识库、向量索引、模型、工具
├─ scripts/            # 启动、初始化、评估、重建脚本
├─ requirements.txt
├─ LangChain-RAG-Agent-学习与搭建文档.md
└─ RAG-Agent-分阶段TODO.md
```

其中 `app/` 是核心业务代码，`configs/` 负责运行参数，`data/` 保存实际知识库内容和向量索引。

## 5. 核心模块说明

### 5.1 API 层

后端入口在 `app/api/main.py`，当前注册了三组主要路由：

- `chat_router`
- `knowledge_base_router`
- `tools_router`

同时暴露了 `/health` 健康检查接口。

对应的业务划分是：

- `app/api/chat.py`
  负责 RAG 对话和 Agent 对话
- `app/api/knowledge_base.py`
  负责知识库上传、重建和列表查询
- `app/api/tools.py`
  负责工具列表和工具调用调试

项目还支持 SSE 流式输出，适合展示 token 级回复、引用片段、工具调用和中间步骤。

### 5.2 配置系统

配置集中在 `app/services/settings.py`，使用三个 YAML 文件拆分职责：

- `configs/basic_settings.yaml`
  服务地址、数据目录、知识库目录、向量库存储目录
- `configs/kb_settings.yaml`
  检索参数、切分参数、OCR、临时知识库 TTL、增量重建开关
- `configs/model_settings.yaml`
  LLM、Embedding、Reranker、图片视觉模型相关配置

这种拆分方式比较适合教学项目：

- 基础环境配置独立
- 检索参数独立
- 模型接入独立

同时，UI 已支持直接修改部分 OCR / VLM 配置并回写 YAML。当前除了基础 OCR 开关、Tesseract 路径和视觉模型参数外，还能配置：

- OCR 主后端
- 说明书页补充后端
- `PaddleOCR` 语言、方向分类、检测长边限制、最低识别分数

在当前默认配置下，图片视觉模型自动 caption 已关闭，优先保证图片入库速度；若某次重建确实需要补齐视觉描述，可以在重建时临时启用图片 VLM，而不必长期打开全局自动 caption。

### 5.3 知识库入库链路

知识库入库主逻辑位于 `app/services/kb_ingestion_service.py` 与 `app/services/embedding_assembler.py`。

完整流程如下：

1. 上传文件到本地知识库或临时知识库目录
2. 通过 `loaders` 将文件解析为 `Document`
3. 使用文本切分器切成 chunks
4. 为每个 chunk 补充 `chunk_id`、页码、标题、章节、模态来源等元数据
5. 调用 embedding 模型生成向量
6. 写入向量库并生成 `metadata.json`

当前支持的文件类型包括：

- `.txt`
- `.md`
- `.pdf`
- `.docx`
- `.epub`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.webp`

当前入库已经不再只是“把所有内容拼成一段文本”，而是会保留一批结构化字段，例如：

- `content_type`
- `source_modality`
- `original_file_type`
- `ocr_text`
- `ocr_language`
- `image_caption`
- `evidence_summary`
- `title / section_title / section_path / page`

这意味着项目已经把图片、OCR、视觉描述和正文内容区分开来处理，而不是把所有模态都压成同一种文本块。

### 5.4 检索链路

检索逻辑集中在 `app/retrievers/local_kb.py`。当前不是单一向量检索，而是做了较完整的检索增强：

- 查询改写
- 稠密向量检索
- 词法检索
- 按 `source_modality` 分组召回
- RRF 融合
- 启发式重排
- 可选模型重排
- small-to-big 上下文扩展
- 元数据过滤
- 文件类型 / 标题路径 / 模态类型偏置
- 图片类问题查询扩展

当前检索已经具备一套轻量的跨模态融合思路：

- 文本问题优先偏正文类 chunk
- 图片相关问题优先偏 `ocr / vision / image` 类 chunk
- 旧索引若没有 `source_modality`，会自动回退到原有全局检索逻辑
- 检索结果会保留 `source_modality`、`evidence_type`、`ocr_text`、`image_caption` 等字段，供回答链路和 UI 继续使用

也就是说，这个项目已经从“基础 FAISS 相似度搜索”进一步走向“混合检索 + 多阶段排序”。

这部分是项目比较有价值的工程点，因为它体现了真实 RAG 系统里常见的问题处理方式：

- 单纯向量检索容易漏召回
- 单纯关键词检索容易排序不稳
- 命中的 chunk 太碎会影响回答质量
- 不做重排时，Top-K 质量波动明显

### 5.5 RAG 生成链路

RAG 生成逻辑位于 `app/chains/rag.py`。

整体思路比较直接：

- 根据检索结果按证据类型组织上下文
- 结合历史消息构造 prompt
- 调用聊天模型生成答案
- 支持普通返回和流式返回

当前上下文不会简单平铺，而是会分成三类证据：

- 文本证据
- OCR 证据
- 视觉描述证据

系统 prompt 明确要求：

- 优先基于检索上下文回答
- 不要编造事实
- 上下文不足时明确说明无法确定
- 区分直接证据与推断
- 不要把 OCR 文本和视觉描述混为一类事实

此外，项目已经补充了图片类问题专用提示词模板。当问题和证据分布更偏向图片理解时，会自动切换到多模态问答 prompt，专门约束：

- OCR 错字不能被模型主观“修正”为事实
- 视觉描述只能作为画面特征证据
- OCR 与视觉描述冲突时必须显式指出
- 图片证据不足时不能拿普通正文强行替代

这部分体现出项目的目标是“可解释、可控的知识库问答”，而不是开放闲聊。

### 5.6 Agent 链路

Agent 主逻辑在 `app/agents/multistep.py`。

当前 Agent 采用的是轻量多步执行方案，不是复杂的 ReAct 自由规划，而是“规则判定 + 工具编排 + 结果汇总”的模式。已内置工具有：

- `search_local_knowledge`
- `calculate`
- `current_time`

Agent 能做的事情包括：

- 先检索知识库再回答
- 从知识库结果中抽取数字继续计算
- 获取当前时间
- 展示工具调用记录和中间步骤

这种设计有两个优点：

- 逻辑清晰，便于教学和调试
- 工具调用行为可控，不容易陷入无边界循环

同时它也意味着当前 Agent 更接近“多工具任务编排器”，而不是高度自治的通用智能体。

### 5.7 图片 OCR 与视觉描述

项目支持将图片作为知识库输入。相关逻辑分散在图片加载、OCR 和 `app/services/image_caption_service.py`。

当前策略已经不是“整图 OCR + 单句 caption”的早期做法，而是面向多模态 RAG 做了更细的拆分：

- OCR 已抽成双后端入口，保留 `Tesseract` 主链路，并支持按说明书页特征切换到 `PaddleOCR`
- 说明书 / 维修手册页会额外抽取步骤类证据，而不是只保留整图文字
- 根据配置决定是否调用视觉模型生成结构化 caption
- 将整图 caption 解析为摘要、场景、主体对象、文字线索、动作状态、不确定点
- 对图片上半区 / 中部区域 / 下半区生成区域级 caption
- 对说明书页中的配图再生成专门的视觉证据，必要时继续尝试箭头 / 局部图区域
- 过滤明显无效的拒答式、违规式视觉描述
- 对超大图在 OCR、区域裁剪、VLM 前统一做安全缩放，并压掉 `DecompressionBombWarning`
- 将 OCR、整图结构化描述、区域级描述、说明书页步骤、配图描述、融合摘要和文件信息分别写入结构化元数据与视觉 chunk

图片与说明书页 chunk 当前会区分几类来源与内容类型：

- `ocr`
- `vision`
- `ocr+vision`
- `image`
- `instruction_text_evidence`
- `instruction_figure_evidence`
- `instruction_arrow_evidence`

同时回答与引用阶段还能继续透传：

- `source_modality`
- `evidence_type`
- `used_for_answer`

当前 API 返回和流式 `done` 事件里还会补充 `reference_overview`，用于总结：

- 文本证据数量
- 图片侧证据数量
- 是否形成图文联合覆盖
- `source_modality / evidence_type` 分布

UI 中已经提供了 OCR 与 VLM 的独立配置项，例如：

- OCR 主后端 / 说明书页补充后端
- Tesseract 路径
- Tesseract 语言包
- `PaddleOCR` 语言、方向分类、检测长边上限、最低识别分数
- OCR 置信度阈值
- VLM Base URL / API Key / Model
- 是否在 OCR 足够丰富时跳过视觉模型

另外，当前图片 caption 链路已切换为更稳定的 OpenAI-compatible `chat_completions` 风格调用，避免部分视觉模型只产出 reasoning 而不返回正式正文，降低“有理解、无可解析输出”的情况。

这部分能力比较适合处理：

- 扫描件
- 截图类知识
- 含文字图片
- 需要简单视觉补充说明的图片材料
- 图文混合知识源的教学型多模态 RAG 实验

相比早期“一张图只生成一条自然语言 caption”的做法，当前实现更适合检索阶段复用，因为：

- 整图结构化描述更利于稳定提取关键词
- 区域级描述可以提升局部内容的命中率
- 图片不再只以单个视觉字段参与检索，而是具备“主视觉证据 + 区域视觉证据”的双层组织
- 说明书页不再只是普通图片，而会补出“步骤证据 + 配图证据”的专门 chunk
- `ocr_language` 会按实际使用后端写回元数据，便于排查 OCR 来源

### 5.8 多模态证据可观测性

为了让多模态 RAG 的优化不只停留在“感觉是否变好”，项目已经补了一层轻量可观测性，主要体现在三个位置：

- 检索阶段会写入 `retrieval_trace.jsonl`
- 回答阶段会写入 `answer_trace.jsonl`
- API 与 UI 会同步展示 `reference_overview`
- `scripts/eval_retrieval.py` 已支持多模态评测样例与失败案例回放

其中：

- `retrieval_trace.jsonl` 记录查询类型、候选数量、候选模态分布、最终引用模态分布
- `answer_trace.jsonl` 记录回答使用的 prompt 类型、证据类型分布，以及是否同时命中文本证据和图片侧证据
- `reference_overview` 会在 `/chat/rag`、`/chat/agent` 以及流式 `done` 事件里返回，便于前端直接展示“文本证据数 / 图片侧证据数 / 联合覆盖状态”
- `data/eval/` 中已补充多模态评测集和失败案例库，便于针对 `image_ocr / image_vision / multimodal_joint` 做回归对比

在 `Streamlit` 控制台里，引用区域现在不仅能展开每条引用，还会先显示一块“证据概览”，让用户快速判断：

- 当前回答是否真的用了图片侧证据
- 当前是否形成了图文联合覆盖
- 当前引用更偏 `text / ocr / vision / multimodal` 的哪一类

这部分能力的意义不是提高模型上限，而是降低调试成本，让开发者能更快判断问题究竟出在入库、检索、重排还是回答组织阶段。

## 6. 运行方式

项目通过脚本启动，入口比较明确。

### 6.1 启动 API

```powershell
.\.venv\Scripts\python.exe .\scripts\start_api.py
```

默认会：

- 读取 `configs/` 下配置
- 在启动前尝试清理过期临时知识库
- 启动 FastAPI 服务

如果当前终端已经手动激活了 `.venv`，也可以直接使用 `python .\scripts\start_api.py`。推荐显式使用 `.venv\Scripts\python.exe`，避免误用系统 Python 或 Anaconda 环境。

### 6.2 启动 UI

```powershell
.\.venv\Scripts\python.exe .\scripts\start_ui.py
```

默认会：

- 读取配置
- 启动前清理临时知识库
- 拉起 Streamlit 页面

### 6.3 基本使用顺序

推荐的使用流程是：

1. 启动 API
2. 启动 UI
3. 在知识库管理页上传本地文件
4. 重建知识库索引
5. 在 RAG 对话页进行问答
6. 在 Agent 对话页测试工具闭环

如果某个知识库里的图片确实需要补齐视觉描述，可以在命令行重建时临时打开图片 VLM，例如：

```powershell
.\.venv\Scripts\python.exe .\scripts\rebuild_kb.py --kb-name test3 --enable-image-vlm-for-build --force-full-rebuild
```

## 7. 数据组织方式

项目的数据目录设计比较直观：

- `data/knowledge_base/<kb_name>/content/`
  存放知识库原始文件
- `data/vector_store/<kb_name>/`
  存放知识库向量索引和元数据
- `data/temp/<knowledge_id>/`
  存放临时知识库文件和临时向量索引
- `data/logs/`
  存放 `retrieval_trace.jsonl`、`answer_trace.jsonl`、`image_caption_trace.jsonl` 等调试日志
- `data/eval/`
  存放多模态评测集、失败案例和回归测试样本
- `data/models/`
  存放本地模型文件，如 reranker
- `data/tools/`
  存放本地工具，例如 Tesseract

这种组织方式的优点是源码与运行时数据分离，便于实验、迁移和后续扩展。

## 8. 当前项目的主要亮点

从当前实现看，这个项目的亮点主要有以下几点：

- 结构完整，覆盖配置、入库、检索、生成、前端和 Agent
- 支持长期知识库与临时知识库两种问答模式
- 检索链路比入门 Demo 更完整，已经包含混合检索与重排
- 支持流式输出，可观察引用、工具调用和中间步骤
- 已加入图片 OCR / VLM 入库能力，具备一定多模态扩展性
- 已支持 `epub` 文档入库，并能按章节保留结构信息
- 已支持多模态结构化元数据、按模态分路召回和证据感知回答
- 已支持说明书页模式，能把步骤文字与配图证据拆开入库
- 已接入 OCR 双后端抽象，支持 `Tesseract + PaddleOCR` 的增量切换方案
- 已对超大图做 OCR / 区域裁剪 / VLM 前安全缩放
- 引用区可区分 `text / ocr / vision / multimodal` 等证据类型
- 已具备基础多模态评测与 trace 观测能力，可回看检索分布、回答证据和图片 caption 行为
- 配置与数据目录分离，便于本地实验

## 9. 当前局限与后续方向

虽然项目已经具备完整闭环，但目前仍有明显边界。

### 9.1 当前局限

- Agent 规划方式较轻，主要靠规则触发，不是通用自治智能体
- 工具集较少，当前更偏演示与验证
- 向量库默认以 FAISS 为主，分布式或大规模场景能力有限
- 多模态部分仍以 OCR 和图片描述增强为主，深层视觉理解仍有限
- 图片检索效果仍明显依赖 OCR 与 caption 质量，弱图像样本下证据可能不足
- `PaddleOCR` 双后端抽象已经接入，但本地若未安装 `paddleocr / paddlepaddle`，当前仍会回落到 `Tesseract`
- 说明书页模式已经能拆步骤和配图，但步骤质量仍受 OCR 结果影响，尚未接入更强的版面分析
- 多模态评测与 trace 日志已经落地，但评测样本规模、指标覆盖和自动化对比能力仍有限
- 项目中存在较多实验数据和阶段性样例，工程清理度一般
- 目录内包含大量 `__pycache__`、示例索引和模型文件，学习友好但仓库纯净度较弱

### 9.2 可继续扩展的方向

- 增加更多 Agent 工具，如网页搜索、代码执行、数据库查询
- 提升 Agent 规划能力，引入更稳定的推理与状态管理
- 扩展检索评测、自动化 benchmark 和可视化分析
- 优化图片识别链路，例如版面检测、标题区 / 步骤区 / 清单区分块 OCR、箭头局部定位、领域词典纠错
- 继续完善多模态评测、检索观测和知识库级策略配置
- 增加更细粒度的权限控制、日志记录和错误追踪
- 对外提供更统一的 OpenAI-compatible 接口层

## 10. 适合的使用场景

这个项目尤其适合以下场景：

- 学习 LangChain 风格的 RAG 项目结构
- 课程设计、毕业设计或实验项目原型
- 本地知识库问答系统的快速验证
- 多工具 Agent 的基础演示
- 文档与图片混合知识源的小规模实验

如果目标是快速上手 RAG + Agent 的最小工程闭环，这个项目已经足够使用；如果目标是面向生产环境的大规模平台化部署，则还需要继续补强稳定性、治理能力和工程规范。

## 11. 总结

`Mini-Agent-RAG2` 可以理解为一个“基于 LangChain 的教学版、可运行、可扩展”的本地 RAG 智能体项目。它已经不再是单脚本 Demo，而是具备清晰模块边界的完整原型系统。

从项目结构上看，它最值得学习的部分有三块：

- 知识库从文件到向量索引的完整构建链路
- 混合检索、重排和上下文组织的 RAG 检索设计
- Agent 与工具调用、流式输出、可视化调试之间的配合方式

如果后续要继续完善，这个项目很适合作为基础骨架，在其上逐步扩展更多工具、更强检索、更稳多模态能力和更规范的工程化配置。
