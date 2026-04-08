from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import requests
import streamlit as st

import app.services.settings as settings_module


settings_module = importlib.reload(settings_module)
load_settings = settings_module.load_settings
save_config_values = settings_module.save_config_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings(PROJECT_ROOT)
DEFAULT_API_BASE_URL = f"http://{SETTINGS.basic.API_HOST}:{SETTINGS.basic.API_PORT}"
DEFAULT_TOOLS = ["search_local_knowledge", "calculate", "current_time"]


def api_request(
    method: str,
    base_url: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    form_data: dict[str, Any] | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    timeout: int = 120,
) -> tuple[bool, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.request(
        method=method,
        url=url,
        json=json_body,
        data=form_data,
        files=files,
        timeout=timeout,
    )
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = response.json()
    else:
        payload = response.text

    if response.ok:
        return True, payload

    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("detail") or "请求失败。"
        return False, f"{response.status_code} {message}"
    return False, f"{response.status_code} {payload}"


def stream_api_events(
    base_url: str,
    path: str,
    *,
    json_body: dict[str, Any],
    timeout: int = 180,
):
    url = f"{base_url.rstrip('/')}{path}"
    with requests.post(
        url,
        json=json_body,
        timeout=timeout,
        stream=True,
        headers={"Accept": "text/event-stream"},
    ) as response:
        if not response.ok:
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                payload = response.json()
                message = payload.get("message") or payload.get("detail") or "流式请求失败。"
                raise RuntimeError(f"{response.status_code} {message}")
            raise RuntimeError(f"{response.status_code} {response.text}")

        data_lines: list[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            line = (raw_line or "").strip()
            if not line:
                if data_lines:
                    yield json.loads("".join(data_lines))
                    data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if data_lines:
            yield json.loads("".join(data_lines))


def inject_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(13, 148, 136, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.16), transparent 24%),
                linear-gradient(180deg, #f4f7f2 0%, #f8fafc 100%);
            color: #15212d;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
        }
        .hero-card, .section-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(21, 33, 45, 0.08);
            border-radius: 20px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.4rem;
        }
        .hero-subtitle {
            color: #405567;
            font-size: 0.98rem;
            line-height: 1.7;
            margin-bottom: 0;
        }
        .meta-chip {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: #e0f2f1;
            color: #115e59;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .section-note {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.6;
            margin-bottom: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Mini Agent RAG 控制台</div>
            <p class="hero-subtitle">
                这个页面覆盖知识库管理、RAG 对话和 Agent 对话三条主链路。
                默认面向本地 FastAPI 服务，聊天模型后端既可走 Ollama，也可切到 OpenAI 兼容 API。
            </p>
            <div style="margin-top:0.8rem;">
                <span class="meta-chip">FastAPI</span>
                <span class="meta-chip">FAISS</span>
                <span class="meta-chip">Ollama / API</span>
                <span class="meta-chip">Streamlit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown("## 连接配置")
    default_api_base_url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
    api_base_url = st.sidebar.text_input("API Base URL", value=default_api_base_url)
    st.session_state["api_base_url"] = api_base_url

    if st.sidebar.button("检查 API 健康状态", use_container_width=True):
        try:
            ok, payload = api_request("GET", api_base_url, "/health", timeout=20)
        except requests.RequestException as exc:
            st.sidebar.error(f"连接失败: {exc}")
        else:
            if ok:
                st.sidebar.success(f"API 可用: {payload}")
            else:
                st.sidebar.error(str(payload))

    st.sidebar.markdown(
        """
        - 先启动 API：`python .\\scripts\\start_api.py`
        - 再启动页面：`python .\\scripts\\start_ui.py`
        - 本页默认不在加载时自动请求接口，避免无服务状态下直接报错
        """
    )
    return api_base_url


def refresh_knowledge_bases(api_base_url: str) -> tuple[bool, Any]:
    ok, payload = api_request("GET", api_base_url, "/knowledge_base/list", timeout=60)
    if ok:
        st.session_state["knowledge_bases"] = payload
    return ok, payload


def render_image_ingestion_panel() -> None:
    image_extensions = [
        ext
        for ext in SETTINGS.kb.SUPPORTED_EXTENSIONS
        if ext.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]

    st.markdown("### 图片 OCR / VLM 配置")
    save_notice = st.session_state.pop("image_ingestion_config_notice", "")
    if save_notice:
        st.success(save_notice)

    with st.form("image_ingestion_config_form"):
        col_ocr, col_vlm = st.columns(2)

        with col_ocr:
            image_ocr_enabled = st.checkbox(
                "启用图片 OCR",
                value=SETTINGS.kb.IMAGE_OCR_ENABLED,
                help="图片文件入库时，先尝试提取图片中的文字。",
            )
            image_ocr_backend = st.selectbox(
                "OCR 主后端",
                options=["tesseract", "paddle", "auto"],
                index=["tesseract", "paddle", "auto"].index(
                    SETTINGS.kb.IMAGE_OCR_BACKEND
                    if SETTINGS.kb.IMAGE_OCR_BACKEND in {"tesseract", "paddle", "auto"}
                    else "tesseract"
                ),
                help="默认保留 Tesseract；说明书页可再切换到 PaddleOCR。",
            )
            image_ocr_instruction_backend = st.selectbox(
                "说明书页补充后端",
                options=["paddle", "tesseract", "auto"],
                index=["paddle", "tesseract", "auto"].index(
                    SETTINGS.kb.IMAGE_OCR_INSTRUCTION_PAGE_BACKEND
                    if SETTINGS.kb.IMAGE_OCR_INSTRUCTION_PAGE_BACKEND in {"paddle", "tesseract", "auto"}
                    else "paddle"
                ),
                help="当识别到说明书/手册页特征时，会尝试用该后端补跑 OCR。",
            )
            image_ocr_language = st.text_input(
                "Tesseract 语言包",
                value=SETTINGS.kb.IMAGE_OCR_LANGUAGE,
                help="常见组合如 `chi_sim+eng`。",
            )
            ocr_tesseract_cmd = st.text_input(
                "Tesseract 路径",
                value=SETTINGS.kb.OCR_TESSERACT_CMD,
                help="可填写绝对路径或相对项目根目录的路径。",
            )
            ocr_min_confidence = st.number_input(
                "OCR 最低置信度",
                min_value=0.0,
                max_value=100.0,
                value=float(SETTINGS.kb.OCR_MIN_CONFIDENCE),
                help="低于该值的 OCR 文本块会直接丢弃。",
            )
            ocr_min_text_length = st.number_input(
                "OCR 最短有效文本长度",
                min_value=0,
                max_value=200,
                value=int(SETTINGS.kb.OCR_MIN_TEXT_LENGTH),
                help="过滤后的单行 OCR 文本长度低于该值时，会被视为噪声。",
            )
            ocr_min_meaningful_ratio = st.number_input(
                "OCR 最低有效字符占比",
                min_value=0.0,
                max_value=1.0,
                value=float(SETTINGS.kb.OCR_MIN_MEANINGFUL_RATIO),
                help="用于过滤符号噪声和乱码，建议保持在 0.5 到 0.8 之间。",
            )
            paddle_ocr_language = st.text_input(
                "PaddleOCR 语言",
                value=SETTINGS.kb.PADDLE_OCR_LANGUAGE,
                help="常用中文是 `ch`。",
            )
            paddle_ocr_use_angle_cls = st.checkbox(
                "PaddleOCR 启用方向分类",
                value=SETTINGS.kb.PADDLE_OCR_USE_ANGLE_CLS,
            )
            paddle_ocr_det_limit_side_len = st.number_input(
                "PaddleOCR 检测长边上限",
                min_value=0,
                max_value=10000,
                value=int(SETTINGS.kb.PADDLE_OCR_DET_LIMIT_SIDE_LEN),
            )
            paddle_ocr_min_score = st.number_input(
                "PaddleOCR 最低识别分数",
                min_value=0.0,
                max_value=1.0,
                value=float(SETTINGS.kb.PADDLE_OCR_MIN_SCORE),
                help="低于该分数的 PaddleOCR 文本块会被丢弃。",
            )
            preview_tesseract_path = (
                SETTINGS.resolve_path(ocr_tesseract_cmd.strip())
                if ocr_tesseract_cmd.strip()
                else None
            )
            preview_tesseract_exists = bool(
                preview_tesseract_path and preview_tesseract_path.exists()
            )
            if ocr_tesseract_cmd.strip():
                st.caption(f"当前预览路径: `{preview_tesseract_path}`")
                if preview_tesseract_exists:
                    st.success("已检测到本地 Tesseract 可执行文件。")
                else:
                    st.warning("当前填写的 Tesseract 路径不存在，保存后 OCR 会失败。")
            else:
                st.info("未配置 Tesseract 路径时，启用 OCR 也无法真正识别图片文字。")

        with col_vlm:
            image_vlm_enabled = st.checkbox(
                "启用图片视觉描述",
                value=SETTINGS.model.IMAGE_VLM_ENABLED,
                help="在 OCR 不足时调用视觉模型生成图片内容描述后再入库。",
            )
            image_vlm_provider = st.selectbox(
                "VLM Provider",
                options=["openai_compatible"],
                index=0,
            )
            image_vlm_api_style = st.selectbox(
                "VLM API Style",
                options=["auto", "responses", "chat_completions"],
                index=["auto", "responses", "chat_completions"].index(
                    SETTINGS.model.IMAGE_VLM_API_STYLE
                    if SETTINGS.model.IMAGE_VLM_API_STYLE in {"auto", "responses", "chat_completions"}
                    else "auto"
                ),
                help="火山方舟视觉接入点通常使用 `responses`；旧式兼容接口可选 `chat_completions`。",
            )
            image_vlm_base_url = st.text_input(
                "VLM Base URL",
                value=SETTINGS.model.IMAGE_VLM_BASE_URL,
                help="例如火山引擎兼容端点 `https://ark.cn-beijing.volces.com/api/v3`。",
            )
            image_vlm_api_key = st.text_input(
                "VLM API Key",
                value=SETTINGS.model.IMAGE_VLM_API_KEY,
                type="password",
            )
            image_vlm_model = st.text_input(
                "VLM Model",
                value=SETTINGS.model.IMAGE_VLM_MODEL,
            )
            image_vlm_use_ocr_context = st.checkbox(
                "VLM 结合 OCR 文本理解图片",
                value=SETTINGS.model.IMAGE_VLM_USE_OCR_CONTEXT,
                help="启用后，会把 OCR 文本一起发给视觉模型，而不是只让模型看图。",
            )
            image_vlm_auto_trigger_by_ocr = st.checkbox(
                "按 OCR 丰富度自动决定是否调用视觉模型",
                value=SETTINGS.model.IMAGE_VLM_AUTO_TRIGGER_BY_OCR,
                help="OCR 文本很丰富时跳过视觉模型，OCR 很少时再调用。",
            )
            image_vlm_skip_if_ocr_chars_at_least = st.number_input(
                "OCR 达到多少字符后跳过视觉模型",
                min_value=0,
                max_value=5000,
                value=SETTINGS.model.IMAGE_VLM_SKIP_IF_OCR_CHARS_AT_LEAST,
                help="仅在启用“按 OCR 丰富度自动决定”时生效。",
            )
            image_vlm_timeout_seconds = st.number_input(
                "VLM Timeout (秒)",
                min_value=1,
                max_value=600,
                value=SETTINGS.model.IMAGE_VLM_TIMEOUT_SECONDS,
            )
            image_vlm_max_tokens = st.number_input(
                "VLM Max Tokens",
                min_value=16,
                max_value=2048,
                value=SETTINGS.model.IMAGE_VLM_MAX_TOKENS,
            )
            image_vlm_only_when_ocr_empty = st.checkbox(
                "仅 OCR 为空时才调用视觉模型",
                value=SETTINGS.model.IMAGE_VLM_ONLY_WHEN_OCR_EMPTY,
            )
            image_vlm_prompt = st.text_area(
                "VLM Prompt 覆盖",
                value=SETTINGS.model.IMAGE_VLM_PROMPT,
                height=120,
                help="留空时使用系统内置的图片描述提示词。",
            )

        save_image_ingestion_config = st.form_submit_button(
            "保存图片 OCR / VLM 配置",
            use_container_width=True,
        )

    if save_image_ingestion_config:
        try:
            save_config_values(
                PROJECT_ROOT,
                "kb_settings.yaml",
                {
                    "IMAGE_OCR_ENABLED": bool(image_ocr_enabled),
                    "IMAGE_OCR_BACKEND": image_ocr_backend,
                    "IMAGE_OCR_INSTRUCTION_PAGE_BACKEND": image_ocr_instruction_backend,
                    "IMAGE_OCR_LANGUAGE": image_ocr_language.strip(),
                    "OCR_TESSERACT_CMD": ocr_tesseract_cmd.strip(),
                    "PADDLE_OCR_LANGUAGE": paddle_ocr_language.strip(),
                    "PADDLE_OCR_USE_ANGLE_CLS": bool(paddle_ocr_use_angle_cls),
                    "PADDLE_OCR_DET_LIMIT_SIDE_LEN": int(paddle_ocr_det_limit_side_len),
                    "PADDLE_OCR_MIN_SCORE": float(paddle_ocr_min_score),
                    "OCR_MIN_CONFIDENCE": float(ocr_min_confidence),
                    "OCR_MIN_TEXT_LENGTH": int(ocr_min_text_length),
                    "OCR_MIN_MEANINGFUL_RATIO": float(ocr_min_meaningful_ratio),
                },
            )
            save_config_values(
                PROJECT_ROOT,
                "model_settings.yaml",
                {
                    "IMAGE_VLM_ENABLED": bool(image_vlm_enabled),
                    "IMAGE_VLM_PROVIDER": image_vlm_provider,
                    "IMAGE_VLM_API_STYLE": image_vlm_api_style,
                    "IMAGE_VLM_BASE_URL": image_vlm_base_url.strip(),
                    "IMAGE_VLM_API_KEY": image_vlm_api_key.strip(),
                    "IMAGE_VLM_MODEL": image_vlm_model.strip(),
                    "IMAGE_VLM_USE_OCR_CONTEXT": bool(image_vlm_use_ocr_context),
                    "IMAGE_VLM_AUTO_TRIGGER_BY_OCR": bool(image_vlm_auto_trigger_by_ocr),
                    "IMAGE_VLM_SKIP_IF_OCR_CHARS_AT_LEAST": int(image_vlm_skip_if_ocr_chars_at_least),
                    "IMAGE_VLM_TIMEOUT_SECONDS": int(image_vlm_timeout_seconds),
                    "IMAGE_VLM_MAX_TOKENS": int(image_vlm_max_tokens),
                    "IMAGE_VLM_PROMPT": image_vlm_prompt.strip(),
                    "IMAGE_VLM_ONLY_WHEN_OCR_EMPTY": bool(image_vlm_only_when_ocr_empty),
                },
            )
        except (OSError, ValueError) as exc:
            st.error(f"保存图片 OCR / VLM 配置失败: {exc}")
        else:
            st.session_state["image_ingestion_config_notice"] = (
                "图片 OCR / VLM 配置已保存到 `kb_settings.yaml` 和 `model_settings.yaml`。"
            )
            st.rerun()

    with st.expander("查看图片入库配置提示", expanded=False):
        st.markdown(
            f"""
            - 当前可直接上传并重建的图片格式：`{", ".join(image_extensions) or "未启用图片格式"}`
            - 图片入库主链路：`图片文件 -> OCR 提取文字 -> 可选 VLM 生成图片描述 -> 写入知识库`
            - `IMAGE_OCR_ENABLED`
              控制是否启用图片文字识别。
            - `OCR_TESSERACT_CMD`
              指向本机 `tesseract.exe`，改完后需要重启 API / UI。
            - `OCR_MIN_CONFIDENCE / OCR_MIN_TEXT_LENGTH / OCR_MIN_MEANINGFUL_RATIO`
              控制 OCR 置信度过滤和乱码清洗，适合用来抑制无文字图片的假阳性。
            - `IMAGE_VLM_ENABLED`
              控制是否启用图片视觉描述。
            - `IMAGE_VLM_API_STYLE`
              控制视觉接口调用风格。火山方舟视觉接入点建议用 `responses`。
            - `IMAGE_VLM_BASE_URL / IMAGE_VLM_API_KEY / IMAGE_VLM_MODEL`
              配置视觉模型的 OpenAI-compatible 接口。
            - `IMAGE_VLM_USE_OCR_CONTEXT`
              启用后，会把 OCR 文本一起发给视觉模型，让模型结合文字和图像做判断。
            - `IMAGE_VLM_AUTO_TRIGGER_BY_OCR`
              启用后，根据 OCR 文字丰富度自动判断是否需要视觉模型。
            - `IMAGE_VLM_SKIP_IF_OCR_CHARS_AT_LEAST`
              OCR 去空白后的字符数达到这个阈值，就跳过视觉模型。
            - `IMAGE_VLM_ONLY_WHEN_OCR_EMPTY`
              仅在关闭自动判断后生效；为 `true` 时，只有 OCR 没识别到有效文字才调用视觉模型。
            """
        )
        st.code(
            """# configs/kb_settings.yaml
IMAGE_OCR_ENABLED: true
IMAGE_OCR_LANGUAGE: chi_sim+eng
OCR_TESSERACT_CMD: C:/Users/ASUS/tesseract-local/tesseract.exe
OCR_MIN_CONFIDENCE: 60.0
OCR_MIN_TEXT_LENGTH: 6
OCR_MIN_MEANINGFUL_RATIO: 0.6

# configs/model_settings.yaml
IMAGE_VLM_ENABLED: true
IMAGE_VLM_PROVIDER: openai_compatible
IMAGE_VLM_API_STYLE: responses
IMAGE_VLM_BASE_URL: https://ark.cn-beijing.volces.com/api/v3
IMAGE_VLM_API_KEY: <YOUR_API_KEY>
IMAGE_VLM_MODEL: ep-20260406190608-m7w79
IMAGE_VLM_USE_OCR_CONTEXT: true
IMAGE_VLM_AUTO_TRIGGER_BY_OCR: true
IMAGE_VLM_SKIP_IF_OCR_CHARS_AT_LEAST: 20
IMAGE_VLM_PROMPT: |
  以下是图片中的文字：
  {ocr_text}

  请结合这些文字，分析图片中的设备状态和是否存在故障
IMAGE_VLM_ONLY_WHEN_OCR_EMPTY: true""",
            language="yaml",
        )
        st.caption("这块现在支持在页面内直接保存。保存后 UI 会刷新；API 在下一次请求时会重新读取配置。")


def render_knowledge_base_tab(api_base_url: str) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">知识库管理</div>
            <p class="section-note">
                这个区域负责查看本地知识库列表、上传长期知识库文件并触发索引重建。
                现在既支持目录放文件，也支持通过页面直接上传到 `data/knowledge_base/&lt;name&gt;/content/`。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_image_ingestion_panel()

    if "knowledge_bases" not in st.session_state:
        st.session_state["knowledge_bases"] = []

    col_refresh, col_spacer = st.columns([1, 2])
    with col_refresh:
        if st.button("刷新知识库列表", use_container_width=True):
            try:
                ok, payload = refresh_knowledge_bases(api_base_url)
            except requests.RequestException as exc:
                st.error(f"知识库列表请求失败: {exc}")
            else:
                if ok:
                    st.session_state["knowledge_bases"] = payload
                    st.success(f"已加载 {len(payload)} 个知识库。")
                else:
                    st.error(str(payload))

    knowledge_bases = st.session_state.get("knowledge_bases", [])
    if knowledge_bases:
        kb_names = [item["knowledge_base_name"] for item in knowledge_bases]
        selected_name = st.selectbox("选择知识库", options=kb_names)
        selected = next(item for item in knowledge_bases if item["knowledge_base_name"] == selected_name)
        st.json(
            {
                "knowledge_base_name": selected["knowledge_base_name"],
                "content_dir": selected["content_dir"],
                "vector_store_dir": selected["vector_store_dir"],
                "file_count": selected["file_count"],
                "index_exists": selected["index_exists"],
                "metadata_exists": selected["metadata_exists"],
            }
        )
        if selected["files"]:
            with st.expander("查看知识库文件列表", expanded=False):
                st.write("\n".join(f"- {name}" for name in selected["files"]))
    else:
        st.info("还没有加载知识库列表。点击“刷新知识库列表”后会显示当前目录下的知识库。")

    with st.form("upload_local_kb_form"):
        st.markdown("### 上传长期知识库文件")
        upload_knowledge_base_name = st.text_input("上传目标知识库", value="phase2_demo")
        local_uploads = st.file_uploader(
            "选择长期知识库文件",
            type=[ext.lstrip(".") for ext in SETTINGS.kb.SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        overwrite_existing = st.checkbox("覆盖同名文件", value=False)
        auto_rebuild = st.checkbox("上传后自动重建", value=False)
        upload_enable_image_vlm = st.checkbox(
            "自动重建时启用图片 VLM（仅本次）",
            value=False,
            help="只对这次自动重建生效，并会关闭增量复用，适合指定知识库手动补图片 caption。",
        )
        upload_force_full_rebuild = st.checkbox(
            "自动重建时强制全量重建",
            value=False,
            help="忽略增量缓存，确保本次重建重新处理全部文件。",
        )
        upload_chunk_size = st.number_input("上传重建 chunk_size", min_value=1, value=SETTINGS.kb.CHUNK_SIZE)
        upload_chunk_overlap = st.number_input(
            "上传重建 chunk_overlap",
            min_value=0,
            value=SETTINGS.kb.CHUNK_OVERLAP,
        )
        submit_upload = st.form_submit_button("上传长期知识库文件", use_container_width=True)

    if submit_upload:
        if not local_uploads:
            st.warning("请先选择至少一个长期知识库文件。")
        else:
            request_files = [
                (
                    "files",
                    (
                        item.name,
                        item.getvalue(),
                        item.type or "application/octet-stream",
                    ),
                )
                for item in local_uploads
            ]
            try:
                ok, payload = api_request(
                    "POST",
                    api_base_url,
                    "/knowledge_base/upload",
                    form_data={
                        "scope": "local",
                        "knowledge_base_name": upload_knowledge_base_name,
                        "overwrite_existing": overwrite_existing,
                        "auto_rebuild": auto_rebuild,
                        "enable_image_vlm_for_build": upload_enable_image_vlm,
                        "force_full_rebuild": upload_force_full_rebuild,
                        "chunk_size": int(upload_chunk_size),
                        "chunk_overlap": int(upload_chunk_overlap),
                    },
                    files=request_files,
                    timeout=240,
                )
            except requests.RequestException as exc:
                st.error(f"长期知识库上传失败: {exc}")
            else:
                if ok:
                    st.success("长期知识库文件上传完成。")
                    st.json(payload)
                    try:
                        refresh_ok, refresh_payload = refresh_knowledge_bases(api_base_url)
                    except requests.RequestException as exc:
                        st.warning(f"上传成功，但刷新知识库列表失败: {exc}")
                    else:
                        if refresh_ok:
                            st.info(f"知识库列表已刷新，当前共 {len(refresh_payload)} 个知识库。")
                        else:
                            st.warning(str(refresh_payload))
                else:
                    st.error(str(payload))

    with st.form("rebuild_kb_form"):
        st.markdown("### 重建知识库索引")
        knowledge_base_name = st.text_input("知识库名称", value="phase2_demo")
        chunk_size = st.number_input("chunk_size", min_value=1, value=SETTINGS.kb.CHUNK_SIZE)
        chunk_overlap = st.number_input("chunk_overlap", min_value=0, value=SETTINGS.kb.CHUNK_OVERLAP)
        embedding_model = st.text_input("Embedding 模型覆盖", value="")
        rebuild_enable_image_vlm = st.checkbox(
            "本次重建启用图片 VLM（仅当前知识库）",
            value=False,
            help="只对这次重建生效，适合在默认关闭自动 caption 的前提下，单独为某个知识库补图片描述。",
        )
        rebuild_force_full = st.checkbox(
            "本次强制全量重建",
            value=False,
            help="忽略增量缓存，确保图片和文档都按当前策略重新处理。",
        )
        submit_rebuild = st.form_submit_button("执行重建", use_container_width=True)

    if submit_rebuild:
        request_body: dict[str, Any] = {
            "knowledge_base_name": knowledge_base_name,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "enable_image_vlm_for_build": rebuild_enable_image_vlm,
            "force_full_rebuild": rebuild_force_full,
        }
        if embedding_model.strip():
            request_body["embedding_model"] = embedding_model.strip()
        try:
            ok, payload = api_request(
                "POST",
                api_base_url,
                "/knowledge_base/rebuild",
                json_body=request_body,
                timeout=180,
            )
        except requests.RequestException as exc:
            st.error(f"知识库重建失败: {exc}")
        else:
            if ok:
                st.success("知识库重建完成。")
                st.json(payload)
            else:
                st.error(str(payload))


def render_references(
    references: list[dict[str, Any]],
    reference_overview: dict[str, Any] | None = None,
) -> None:
    if not references:
        st.info("没有返回引用。")
        return
    render_reference_summary(references, reference_overview=reference_overview)
    for index, ref in enumerate(references, start=1):
        title = f"[{index}] {ref.get('source', 'unknown')}"
        with st.expander(title, expanded=index == 1):
            st.write(f"chunk_id: {ref.get('chunk_id', '')}")
            st.write(f"relevance_score: {ref.get('relevance_score', '')}")
            st.write(f"evidence_type: {ref.get('evidence_type', '')}")
            st.write(f"used_for_answer: {ref.get('used_for_answer', '')}")
            st.write(f"source_modality: {ref.get('source_modality', '')}")
            st.write(f"source_path: {ref.get('source_path', '')}")
            st.write(ref.get("content_preview") or ref.get("content") or "")


def render_reference_summary(
    references: list[dict[str, Any]],
    *,
    reference_overview: dict[str, Any] | None = None,
) -> None:
    summary = normalize_reference_overview(reference_overview) or summarize_references(references)
    text_count = summary["text_count"]
    image_side_count = summary["image_side_count"]
    multimodal_count = summary["multimodal_count"]
    has_joint_coverage = summary["has_joint_text_image_coverage"]

    st.markdown("#### 证据概览")
    col_text, col_image, col_joint = st.columns(3)
    with col_text:
        st.metric("文本证据", int(text_count))
    with col_image:
        st.metric("图片侧证据", int(image_side_count))
    with col_joint:
        st.metric("联合覆盖", "是" if has_joint_coverage else "否")

    if has_joint_coverage:
        st.success("本次检索已同时命中文本证据和图片侧证据。")
    elif image_side_count > 0:
        st.info("当前引用包含图片侧证据，但未形成文本 + 图片的联合覆盖。")
    else:
        st.info("当前引用以文本证据为主，未包含图片侧证据。")

    modality_summary = format_reference_distribution(summary["source_modality_counts"])
    evidence_summary = format_reference_distribution(summary["evidence_type_counts"])
    if modality_summary:
        st.caption(f"source_modality 分布: {modality_summary}")
    if evidence_summary:
        st.caption(f"evidence_type 分布: {evidence_summary}")
    if multimodal_count > 0:
        st.caption(f"包含 {multimodal_count} 条 OCR + 视觉联合证据。")


def summarize_references(references: list[dict[str, Any]]) -> dict[str, Any]:
    source_modality_counts: dict[str, int] = {}
    evidence_type_counts: dict[str, int] = {}
    text_count = 0
    image_side_count = 0
    multimodal_count = 0

    for ref in references:
        source_modality = str(ref.get("source_modality", "") or "").strip() or "missing"
        evidence_type = str(ref.get("evidence_type", "") or "").strip() or "missing"
        source_modality_counts[source_modality] = source_modality_counts.get(source_modality, 0) + 1
        evidence_type_counts[evidence_type] = evidence_type_counts.get(evidence_type, 0) + 1

        if evidence_type == "text" or source_modality == "text":
            text_count += 1
        if evidence_type in {"ocr", "vision", "multimodal"} or source_modality in {"ocr", "vision", "ocr+vision", "image"}:
            image_side_count += 1
        if evidence_type == "multimodal" or source_modality == "ocr+vision":
            multimodal_count += 1

    return {
        "text_count": text_count,
        "image_side_count": image_side_count,
        "multimodal_count": multimodal_count,
        "has_joint_text_image_coverage": text_count > 0 and image_side_count > 0,
        "source_modality_counts": source_modality_counts,
        "evidence_type_counts": evidence_type_counts,
    }


def normalize_reference_overview(
    overview: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(overview, dict):
        return None
    return {
        "text_count": int(overview.get("text_count", 0) or 0),
        "image_side_count": int(overview.get("image_side_count", 0) or 0),
        "multimodal_count": int(overview.get("multimodal_count", 0) or 0),
        "has_joint_text_image_coverage": bool(overview.get("has_joint_text_image_coverage", False)),
        "source_modality_counts": dict(overview.get("source_modality_counts", {}) or {}),
        "evidence_type_counts": dict(overview.get("evidence_type_counts", {}) or {}),
    }


def format_reference_distribution(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    parts = [f"{key}={value}" for key, value in sorted(counts.items(), key=lambda item: item[0])]
    return " / ".join(parts)


def render_tool_calls(tool_calls: list[dict[str, Any]]) -> None:
    if not tool_calls:
        st.info("没有触发工具调用。")
        return
    for index, record in enumerate(tool_calls, start=1):
        step_index = record.get("step_index", index)
        with st.expander(f"工具调用 {step_index}: {record.get('tool_name', '')}", expanded=True):
            st.json(record)


def render_agent_steps(steps: list[dict[str, Any]]) -> None:
    if not steps:
        st.info("没有返回中间步骤。")
        return
    for index, step in enumerate(steps, start=1):
        step_index = step.get("step_index", index)
        title = f"步骤 {step_index}: {step.get('kind', 'unknown')}"
        with st.expander(title, expanded=True):
            st.json(step)


def render_streaming_response(
    api_base_url: str,
    path: str,
    request_body: dict[str, Any],
    *,
    include_tools: bool,
    timeout: int = 180,
) -> None:
    status_placeholder = st.empty()
    st.markdown("### 回答")
    answer_placeholder = st.empty()
    references_placeholder = st.empty()
    tool_calls_placeholder = st.empty() if include_tools else None
    steps_placeholder = st.empty() if include_tools else None

    answer_text = ""
    references: list[dict[str, Any]] = []
    reference_overview: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    status_placeholder.info("流式输出进行中...")

    try:
        for event in stream_api_events(
            api_base_url,
            path,
            json_body=request_body,
            timeout=timeout,
        ):
            event_type = event.get("type")

            if event_type == "token":
                answer_text += str(event.get("delta", ""))
                answer_placeholder.markdown(answer_text)
                continue

            if event_type == "reference":
                reference = event.get("reference")
                if isinstance(reference, dict):
                    references.append(reference)
                    with references_placeholder.container():
                        st.markdown("### 引用")
                        render_references(references, reference_overview=reference_overview)
                continue

            if event_type == "tool_call":
                tool_call = event.get("tool_call")
                if include_tools and isinstance(tool_call, dict):
                    tool_calls.append(tool_call)
                    if tool_calls_placeholder is not None:
                        with tool_calls_placeholder.container():
                            st.markdown("### 工具调用记录")
                            render_tool_calls(tool_calls)
                continue

            if event_type == "step":
                step = event.get("step")
                if include_tools and isinstance(step, dict):
                    steps.append(step)
                    if steps_placeholder is not None:
                        with steps_placeholder.container():
                            st.markdown("### 中间步骤")
                            render_agent_steps(steps)
                continue

            if event_type == "error":
                status_placeholder.error(str(event.get("message", "流式请求失败。")))
                return

            if event_type == "done":
                final_answer = str(event.get("answer", ""))
                if final_answer and final_answer != answer_text:
                    answer_text = final_answer
                    answer_placeholder.markdown(answer_text)

                final_references = event.get("references")
                final_reference_overview = event.get("reference_overview")
                if isinstance(final_reference_overview, dict):
                    reference_overview = final_reference_overview

                if not references and isinstance(final_references, list):
                    references = [item for item in final_references if isinstance(item, dict)]
                if references:
                    with references_placeholder.container():
                        st.markdown("### 引用")
                        render_references(references, reference_overview=reference_overview)

                final_tool_calls = event.get("tool_calls")
                if include_tools and isinstance(final_tool_calls, list):
                    normalized_tool_calls = [item for item in final_tool_calls if isinstance(item, dict)]
                    if len(normalized_tool_calls) >= len(tool_calls):
                        tool_calls = normalized_tool_calls
                    if tool_calls_placeholder is not None:
                        with tool_calls_placeholder.container():
                            st.markdown("### 工具调用记录")
                            render_tool_calls(tool_calls)

                final_steps = event.get("steps")
                if include_tools and isinstance(final_steps, list):
                    normalized_steps = [item for item in final_steps if isinstance(item, dict)]
                    if len(normalized_steps) >= len(steps):
                        steps = normalized_steps
                    if steps_placeholder is not None:
                        with steps_placeholder.container():
                            st.markdown("### 中间步骤")
                            render_agent_steps(steps)

                status_placeholder.success("流式请求完成。")
                return

        status_placeholder.warning("流式连接结束，但未收到 done 事件。")
    except (requests.RequestException, RuntimeError, json.JSONDecodeError) as exc:
        status_placeholder.error(f"流式请求失败: {exc}")


def render_rag_tab(api_base_url: str) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">RAG 对话</div>
            <p class="section-note">
                支持本地知识库问答和临时文件问答。临时文件上传成功后，会返回 `knowledge_id`，后续查询直接复用该 ID。
                当前版本会按 TTL 自动清理临时知识库，上传响应里会给出过期时间。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_type = st.radio(
        "问答来源",
        options=["local_kb", "temp_kb"],
        horizontal=True,
        format_func=lambda value: "本地知识库" if value == "local_kb" else "临时文件",
    )

    if "temp_knowledge_id" not in st.session_state:
        st.session_state["temp_knowledge_id"] = ""

    if source_type == "temp_kb":
        uploads = st.file_uploader(
            "上传临时文件",
            type=[ext.lstrip(".") for ext in SETTINGS.kb.SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        if st.button("上传并生成临时知识库", use_container_width=True):
            if not uploads:
                st.warning("请先选择至少一个文件。")
            else:
                request_files = [
                    (
                        "files",
                        (
                            item.name,
                            item.getvalue(),
                            item.type or "application/octet-stream",
                        ),
                    )
                    for item in uploads
                ]
                try:
                    ok, payload = api_request(
                        "POST",
                        api_base_url,
                        "/knowledge_base/upload",
                        form_data={"scope": "temp", "knowledge_base_name": ""},
                        files=request_files,
                        timeout=180,
                    )
                except requests.RequestException as exc:
                    st.error(f"临时文件上传失败: {exc}")
                else:
                    if ok:
                        st.session_state["temp_knowledge_id"] = payload["knowledge_id"]
                        st.success(f"临时知识库已生成: {payload['knowledge_id']}")
                        st.json(payload)
                    else:
                        st.error(str(payload))

    with st.form("rag_chat_form"):
        knowledge_base_name = st.text_input(
            "knowledge_base_name",
            value="phase2_demo" if source_type == "local_kb" else "",
            disabled=source_type != "local_kb",
        )
        knowledge_id = st.text_input(
            "knowledge_id",
            value=st.session_state.get("temp_knowledge_id", ""),
            disabled=source_type != "temp_kb",
        )
        query = st.text_area("问题", value="什么是 RAG？", height=120)
        top_k = st.slider("top_k", min_value=1, max_value=10, value=SETTINGS.kb.VECTOR_SEARCH_TOP_K)
        score_threshold = st.slider("score_threshold", min_value=0.0, max_value=1.0, value=float(SETTINGS.kb.SCORE_THRESHOLD))
        stream_response = st.checkbox("启用流式输出", value=True)
        submit_rag = st.form_submit_button("发送 RAG 请求", use_container_width=True)

    if submit_rag:
        request_body = {
            "query": query,
            "source_type": source_type,
            "knowledge_base_name": knowledge_base_name,
            "knowledge_id": knowledge_id,
            "top_k": int(top_k),
            "score_threshold": float(score_threshold),
            "history": [],
            "stream": bool(stream_response),
        }
        if stream_response:
            render_streaming_response(
                api_base_url,
                "/chat/rag",
                request_body,
                include_tools=False,
                timeout=180,
            )
            return
        try:
            ok, payload = api_request(
                "POST",
                api_base_url,
                "/chat/rag",
                json_body=request_body,
                timeout=180,
            )
        except requests.RequestException as exc:
            st.error(f"RAG 请求失败: {exc}")
        else:
            if ok:
                st.success("RAG 请求成功。")
                st.markdown("### 回答")
                st.write(payload["answer"])
                st.markdown("### 引用")
                render_references(
                    payload.get("references", []),
                    reference_overview=payload.get("reference_overview"),
                )
            else:
                st.error(str(payload))


def render_agent_tab(api_base_url: str) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Agent 对话</div>
            <p class="section-note">
                这个区域用于验证工具调用闭环。当前版本已支持多工具连续编排，可展示按顺序执行的工具记录和中间步骤轨迹。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "tool_definitions" not in st.session_state:
        st.session_state["tool_definitions"] = []

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("刷新工具列表", use_container_width=True):
            try:
                ok, payload = api_request("GET", api_base_url, "/tools", timeout=60)
            except requests.RequestException as exc:
                st.error(f"工具列表请求失败: {exc}")
            else:
                if ok:
                    st.session_state["tool_definitions"] = payload
                    st.success(f"已加载 {len(payload)} 个工具。")
                else:
                    st.error(str(payload))

    tool_definitions = st.session_state.get("tool_definitions", [])
    if tool_definitions:
        with st.expander("查看工具定义", expanded=False):
            st.json(tool_definitions)

    available_tools = [item["name"] for item in tool_definitions] or DEFAULT_TOOLS
    with st.form("agent_chat_form"):
        knowledge_base_name = st.text_input("knowledge_base_name（可选）", value="phase2_demo")
        query = st.text_area("Agent 问题", value="什么是 RAG？", height=120)
        allowed_tools = st.multiselect("allowed_tools", options=available_tools, default=available_tools)
        top_k = st.slider("Agent top_k", min_value=1, max_value=10, value=SETTINGS.kb.VECTOR_SEARCH_TOP_K)
        score_threshold = st.slider(
            "Agent score_threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(SETTINGS.kb.SCORE_THRESHOLD),
        )
        max_steps = st.slider("Agent max_steps", min_value=1, max_value=8, value=4)
        stream_response = st.checkbox("Agent 启用流式输出", value=True)
        submit_agent = st.form_submit_button("发送 Agent 请求", use_container_width=True)

    if submit_agent:
        request_body = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": int(top_k),
            "score_threshold": float(score_threshold),
            "history": [],
            "allowed_tools": allowed_tools,
            "max_steps": int(max_steps),
            "stream": bool(stream_response),
        }
        if stream_response:
            render_streaming_response(
                api_base_url,
                "/chat/agent",
                request_body,
                include_tools=True,
                timeout=180,
            )
            return
        try:
            ok, payload = api_request(
                "POST",
                api_base_url,
                "/chat/agent",
                json_body=request_body,
                timeout=180,
            )
        except requests.RequestException as exc:
            st.error(f"Agent 请求失败: {exc}")
        else:
            if ok:
                st.success("Agent 请求成功。")
                st.markdown("### 回答")
                st.write(payload["answer"])
                st.markdown("### 工具调用记录")
                render_tool_calls(payload.get("tool_calls", []))
                st.markdown("### 中间步骤")
                render_agent_steps(payload.get("steps", []))
                st.markdown("### 引用")
                render_references(
                    payload.get("references", []),
                    reference_overview=payload.get("reference_overview"),
                )
            else:
                st.error(str(payload))


def main() -> None:
    st.set_page_config(
        page_title="Mini Agent RAG 控制台",
        layout="wide",
    )
    inject_page_style()
    render_header()
    api_base_url = render_sidebar()

    tab_kb, tab_rag, tab_agent = st.tabs(["知识库管理", "RAG 对话", "Agent 对话"])
    with tab_kb:
        render_knowledge_base_tab(api_base_url)
    with tab_rag:
        render_rag_tab(api_base_url)
    with tab_agent:
        render_agent_tab(api_base_url)


if __name__ == "__main__":
    main()
