# 后端跑通 RAG 指南

这份文档说明怎样不依赖前端页面，只通过脚本和 API，把项目后端的 RAG 跑通。

1. 脚本方式：直接用项目脚本启动、建库和测试
2. API 方式：手动调用 `/knowledge_base/*` 和 `/chat/rag`

## 1. 当前后端的真实入口

- API 启动入口：`scripts/start_api.py`
- API 应用入口：`app/api/main.py`
- 健康检查：`GET /health`
- 知识库上传接口：`POST /knowledge_base/upload`
- 知识库重建接口：`POST /knowledge_base/rebuild`
- 重建状态接口：`GET /knowledge_base/rebuild/{task_id}`
- 知识库列表接口：`GET /knowledge_base/list`
- RAG 问答接口：`POST /chat/rag`

默认 API 地址：

```text
http://127.0.0.1:8000
```

## 2. 环境准备

建议使用 Python 3.11。

首次准备可以执行：

```powershell
py -3.11 -m venv .venv
.\scripts\activate.ps1
pip install -r requirements.txt
python scripts/init.py
```

如果虚拟环境已经存在，则执行：

```powershell
.\scripts\activate.ps1
pip install -r requirements.txt
python scripts/init.py
```

## 3. 配置模型与密钥

项目当前默认配置是：

- `LLM_PROVIDER: openai_compatible`
- `EMBEDDING_PROVIDER: ollama`
- `DEFAULT_LLM_MODEL: deepseek-chat`
- `DEFAULT_EMBEDDING_MODEL: bge-m3:latest`

所以至少需要：

1. 一个可用的大模型接口
2. 一个可用的 embedding 服务

建议在 `configs/.env` 中配置：

```env
OPENAI_COMPATIBLE_API_KEY=你的大模型Key
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1
IMAGE_VLM_API_KEY=你的图片模型Key
IMAGE_VLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

如果继续使用默认 embedding 配置，请确保 Ollama 正在运行，并有 `bge-m3:latest` 模型：

```powershell
ollama pull bge-m3:latest
ollama serve
```

如果你暂时不处理图片，可以把 `configs/model_settings.yaml` 里的 `IMAGE_VLM_ENABLED` 改成 `false`，减少额外依赖。

## 4. 先把 API 启起来

执行：

```powershell
python scripts/start_api.py
```

再检查健康状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

成功返回：

```json
{"status":"ok"}
```

如果这里都没通，后面的上传、建库和问答都不需要继续。

## 5. 后端最短跑通方式：项目自带脚本

如果你的目标只是先验证后端闭环，最短方式是：

```powershell
python scripts/run_rag_quickstart.py
```

这个脚本会做几件事：

1. 如果目标知识库目录为空，自动写一个示例文本
2. 检查 API 是否已启动
3. 必要时自动启动 API
4. 调用 `scripts/rebuild_kb.py` 重建知识库
5. 调用 `/chat/rag` 发起问答
6. 打印答案和引用摘要

如果依赖还没装，也可以：

```powershell
python scripts/run_rag_quickstart.py --install-requirements
```

这是最快的“后端是否能跑通”验证方法。

## 6. 后端标准方式一：本地知识库 `local_kb`

这是最常用、最稳定的后端调用方式。

### 步骤 1：准备知识库文件

把文件放到：

```text
data/knowledge_base/<知识库名>/content/
```

例如：

```text
data/knowledge_base/demo/content/intro.txt
```

建议第一次先放一个最简单的 `.txt` 或 `.md` 文件。

### 步骤 2：重建知识库

最简单的方式是直接调用项目脚本：

```powershell
python scripts/rebuild_kb.py --kb-name demo
```

你也可以直接调 API：

```powershell
$body = @{
  knowledge_base_name = "demo"
  chunk_size = 400
  chunk_overlap = 80
  enable_image_vlm_for_build = $false
  force_full_rebuild = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/knowledge_base/rebuild" `
  -ContentType "application/json" `
  -Body $body
```

正常会返回一个 `task_id`。

### 步骤 3：查询重建状态

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/knowledge_base/rebuild/<task_id>"
```

看到 `status` 为 `succeeded`，再继续问答。

### 步骤 4：发起 RAG 请求

```powershell
$rag = @{
  query = "什么是 RAG？"
  source_type = "local_kb"
  knowledge_base_name = "demo"
  knowledge_id = ""
  top_k = 5
  score_threshold = 0.5
  history = @()
  stream = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/chat/rag" `
  -ContentType "application/json" `
  -Body $rag
```

返回结果里最重要的字段是：

- `answer`
- `references`
- `reference_overview`
- `used_context`

## 7. 后端标准方式二：临时知识库 `temp_kb`

这条链路最适合“上传文件后立即问答”。

### 步骤 1：上传临时文件

Windows 下最稳的是直接用 `curl.exe`：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/knowledge_base/upload" `
  -F "scope=temp" `
  -F "knowledge_base_name=" `
  -F "files=@data\\knowledge_base\\demo\\content\\intro.txt"
```

成功后会返回：

- `scope`
- `knowledge_id`
- `saved_files`
- `expires_at`

把返回的 `knowledge_id` 记下来。

### 步骤 2：基于 `knowledge_id` 问答

```powershell
$rag = @{
  query = "什么是 RAG？"
  source_type = "temp_kb"
  knowledge_base_name = ""
  knowledge_id = "temp-替换成你的ID"
  top_k = 5
  score_threshold = 0.5
  history = @()
  stream = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/chat/rag" `
  -ContentType "application/json" `
  -Body $rag
```

这条链路不需要手动执行 `/knowledge_base/rebuild`，因为临时知识库上传时会直接完成入库。

## 8. 后端标准方式三：通过接口上传长期知识库

如果你不想手动往 `data/knowledge_base/` 放文件，也可以直接上传到长期知识库。

```powershell
curl.exe -X POST "http://127.0.0.1:8000/knowledge_base/upload" `
  -F "scope=local" `
  -F "knowledge_base_name=demo" `
  -F "overwrite_existing=false" `
  -F "auto_rebuild=true" `
  -F "chunk_size=400" `
  -F "chunk_overlap=80" `
  -F "enable_image_vlm_for_build=false" `
  -F "force_full_rebuild=false" `
  -F "files=@data\\knowledge_base\\demo\\content\\intro.txt"
```

如果 `auto_rebuild=true`，返回里可能直接带 `rebuild_result`。如果你设成 `false`，则还需要手动调用一次 `/knowledge_base/rebuild`。

## 9. 是否要开流式输出

第一次跑通建议先用：

```json
"stream": false
```

因为这样返回的是普通 JSON，最好排查。

如果要流式输出，把 `stream` 设为 `true`。这时 `/chat/rag` 返回的是 `text/event-stream`，你需要用 SSE 客户端来接收 `reference`、`token` 和 `done` 事件。

## 10. 常见报错和排查方法

### 10.1 `/health` 不通

优先检查：

- `python scripts/start_api.py` 是否还在运行
- 端口是否被占用
- 你访问的是不是 `127.0.0.1:8000`

### 10.2 重建时报缺少 API Key

这是大模型配置没准备好。检查：

- `configs/.env`
- `OPENAI_COMPATIBLE_API_KEY`
- `OPENAI_COMPATIBLE_BASE_URL`

### 10.3 重建时报 embedding 失败

通常是 Ollama 没启动，或没准备好默认 embedding 模型。检查：

- `OLLAMA_BASE_URL`
- `EMBEDDING_PROVIDER`
- `DEFAULT_EMBEDDING_MODEL`

### 10.4 返回 404，提示知识库不存在

说明：

- `knowledge_base_name` 填错了
- 或知识库目录存在，但还没有构建索引

优先先跑一遍：

```powershell
python scripts/rebuild_kb.py --kb-name <知识库名>
```

### 10.5 返回 410，提示临时知识库过期

说明 `knowledge_id` 对应的临时知识库已经被自动清理。重新上传文件即可。

当前默认 TTL 是 `120` 分钟。

### 10.6 有答案但没有引用

通常说明：

- 召回不到内容
- `score_threshold` 太高
- 文档里没有能命中的片段

第一次验证建议：

- 用 `.txt` 文本
- 问一个几乎逐字命中的问题
- `top_k=5`
- `score_threshold=0.3` 或 `0.5`
