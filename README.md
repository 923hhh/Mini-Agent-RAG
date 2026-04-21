# Mini-Agent-RAG

一个基于 FastAPI 的本地知识库 RAG 项目，支持：

- 本地知识库与临时知识库问答
- 混合检索（向量检索 + BM25）
- 句级索引与重排
- 多模态图片文档 OCR / 视觉描述
- 增量重建知识库

## 目录结构

- `app/`：后端 API、检索链路、服务层与前端页面
- `configs/`：基础配置、模型配置、知识库配置
- `scripts/`：常用启动、初始化、重建与清理脚本
- `data/`：运行期数据目录

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

启动后端：

```bash
python scripts/start_api.py
```

可选启动页面：

```bash
python scripts/start_ui.py
```

## 常用流程

1. 上传文件到知识库，或将文件放入本地知识库目录。
2. 重建知识库索引。
3. 调用 `/chat/rag` 发起问答。

## 常用脚本

- `python scripts/init.py`
- `python scripts/start_api.py`
- `python scripts/start_ui.py`
- `python scripts/rebuild_kb.py --kb-name <知识库名>`
- `python scripts/run_rag_quickstart.py`
- `python scripts/cleanup_temp_kb.py`

## 说明

仓库根目录仅保留 README 作为项目入口说明
