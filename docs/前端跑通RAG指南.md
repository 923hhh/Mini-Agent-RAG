# 前端跑通 RAG 指南

这份文档说明怎样通过项目自带的 Streamlit 页面，把 RAG 从 0 跑通。

1. 启动后端 API。
2. 启动 Streamlit 页面。
3. 在页面里上传文件或选择本地知识库。
4. 在页面里发起 RAG 问答并看到答案与引用。

## 1. 先理解当前项目的前后端关系

- 前端页面启动入口：`scripts/start_ui.py`
- 前端页面主文件：`app/ui/app.py`
- 后端启动入口：`scripts/start_api.py`
- API 主程序：`app/api/main.py`
- 知识库接口：`app/api/knowledge_base.py`
- RAG 问答接口：`app/api/chat.py`

当前默认地址：

- API：`http://127.0.0.1:8000`
- UI：`http://127.0.0.1:8501`

前端页面本身不做本地检索和生成，它只是调用后端 API。所以前端要跑通，API 必须先可用。

## 2. 环境准备

建议使用 Python 3.11。当前项目的默认 Python 版本写在 `configs/basic_settings.yaml`。

如果还没有虚拟环境，可以先执行：

```powershell
py -3.11 -m venv .venv
.\scripts\activate.ps1
pip install -r requirements.txt
python scripts/init.py
```

如果你已经有 `.venv`，直接执行：

```powershell
.\scripts\activate.ps1
pip install -r requirements.txt
python scripts/init.py
```

`python scripts/init.py` 会补齐缺失的配置文件和运行目录。

## 3. 配置模型与密钥

项目当前默认配置来自：

- `configs/model_settings.yaml`
- `configs/kb_settings.yaml`
- `configs/.env`

当前默认模型策略是：

- `LLM_PROVIDER: openai_compatible`
- `EMBEDDING_PROVIDER: ollama`
- `DEFAULT_LLM_MODEL: deepseek-chat`
- `DEFAULT_EMBEDDING_MODEL: bge-m3:latest`

这意味着你至少要准备两部分能力：

1. 一个 OpenAI-compatible 对话模型接口
2. 一个本地可用的 Ollama Embedding 服务

先准备环境变量文件。可以把 `configs/.env.example` 复制成 `configs/.env`，至少填入：

```env
OPENAI_COMPATIBLE_API_KEY=你的大模型Key
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1
IMAGE_VLM_API_KEY=你的图片模型Key
IMAGE_VLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

说明：

- 如果你暂时不处理图片文档，可以把 `configs/model_settings.yaml` 里的 `IMAGE_VLM_ENABLED` 改成 `false`。
- 如果你不想依赖 Ollama，也可以把 `EMBEDDING_PROVIDER` 改成 `openai_compatible`，同时把 `DEFAULT_EMBEDDING_MODEL` 改成你接口支持的 embedding 模型。

如果沿用默认 Embedding 配置，请确保本机 Ollama 已启动，并准备好 `bge-m3:latest`。常见做法是：

```powershell
ollama pull bge-m3:latest
ollama serve
```

## 4. 启动后端 API

前端页面必须依赖 API，所以先启动它：

```powershell
python scripts/start_api.py
```

默认监听：

```text
http://127.0.0.1:8000
```

你可以再开一个终端做健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

正常应返回：

```json
{"status":"ok"}
```

## 5. 启动前端页面

新开一个终端，执行：

```powershell
.\scripts\activate.ps1
python scripts/start_ui.py
```

然后浏览器打开：

```text
http://127.0.0.1:8501
```

页面主要有 3 个标签页：

- `知识库管理`
- `RAG 对话`
- `Agent 对话`

这次跑通 RAG 主要用前两个。

## 6. 前端最短跑通路径一：本地知识库

这是最适合长期使用的方式。

### 步骤 1：检查 API 连接

在页面左侧边栏：

1. 确认 `API Base URL` 是 `http://127.0.0.1:8000`
2. 点击 `检查 API 健康状态`

如果看到成功提示，说明前后端已经连通。

### 步骤 2：上传长期知识库文件

进入 `知识库管理` 标签页：

1. 点击 `刷新知识库列表`
2. 在 `上传长期知识库文件` 区域填写 `上传目标知识库`
3. 选择文件
4. 如有需要勾选 `上传后自动重建`
5. 点击 `上传长期知识库文件`

支持的文件类型默认来自 `configs/kb_settings.yaml`：

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

如果你不想用页面上传，也可以直接把文件放到：

```text
data/knowledge_base/<知识库名>/content/
```

### 步骤 3：重建知识库索引

如果上传时没有勾选 `上传后自动重建`，就继续在 `知识库管理` 页的 `重建知识库索引` 区域操作：

1. 填写 `知识库名称`
2. 保持或调整 `chunk_size` 和 `chunk_overlap`
3. 点击 `执行重建`

默认参数来自 `configs/kb_settings.yaml`：

- `CHUNK_SIZE: 400`
- `CHUNK_OVERLAP: 80`

页面会轮询 `/knowledge_base/rebuild/{task_id}`，等状态变成 `succeeded` 后显示结果。

### 步骤 4：进入 RAG 对话

进入 `RAG 对话` 标签页：

1. 把 `问答来源` 选成 `本地知识库`
2. `knowledge_base_name` 填你刚才的知识库名
3. 输入问题
4. 选择 `top_k` 和 `score_threshold`
5. 勾选或取消 `启用流式输出`
6. 点击 `发送 RAG 请求`

成功后页面会显示：

- 回答
- 引用片段
- 引用统计

## 7. 前端最短跑通路径二：临时文件问答

这是最适合快速演示的方式，不需要先建长期知识库。

### 步骤 1：上传临时文件

进入 `RAG 对话` 标签页：

1. 把 `问答来源` 选成 `临时文件`
2. 选择一个或多个文件
3. 点击 `上传并生成临时知识库`

成功后页面会返回一个 `knowledge_id`，并自动写入输入框。

### 步骤 2：直接提问

继续在同一页：

1. 保持 `knowledge_id` 不变
2. 输入问题
3. 点击 `发送 RAG 请求`

这条链路背后会先调用 `/knowledge_base/upload`，然后再调用 `/chat/rag`。

## 8. 页面里几个重要参数怎么理解

- `knowledge_base_name`
  只用于 `local_kb`。
- `knowledge_id`
  只用于 `temp_kb`。
- `top_k`
  返回给生成阶段的候选片段数量上限。
- `score_threshold`
  召回过滤阈值，越高越严格。
- `启用流式输出`
  选中后会通过 SSE 逐步显示生成内容。

## 9. 常见问题排查

### 9.1 页面能打开，但 API 连不上

检查：

- `python scripts/start_api.py` 是否真的在运行
- 端口是不是 `8000`
- 页面左侧 `API Base URL` 是否填对

先用：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

确认后端是否存活。

### 9.2 上传成功，但重建失败

常见原因：

- LLM 或 Embedding 配置不完整
- Ollama 没启动
- OCR / 图片 VLM 相关依赖不可用
- 上传了不在 `SUPPORTED_EXTENSIONS` 里的文件

建议优先先用 `.txt` 或 `.md` 文本文件验证闭环。

### 9.3 能提问，但回答为空或没有引用

检查：

- 是否已经重建成功
- `knowledge_base_name` 是否填错
- `score_threshold` 是否过高
- 文档内容是否真的包含答案

最稳妥的测试方法，是先上传一个只有几行定义文本的小文件，再问一个能直接命中的问题。

### 9.4 临时知识库突然不能用了

临时知识库有 TTL。当前默认值来自 `configs/kb_settings.yaml`：

```text
TEMP_KB_TTL_MINUTES: 120
```

过期后重新上传即可。
