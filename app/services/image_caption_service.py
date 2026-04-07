from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from app.services.network import build_httpx_client
from app.services.settings import AppSettings


DEFAULT_IMAGE_CAPTION_PROMPT = (
    "以下是图片中的文字：\n"
    "{ocr_text}\n\n"
    "请结合这些文字，用中文概括图片里可以直接确认的内容。"
    "优先描述：主体对象、场景类型、可见文字、界面/标识、明显动作或状态。"
    "如果 OCR 文字和画面不一致，要明确指出。"
    "如果图片信息不足，只说明看得见的内容和限制，不要输出拒答套话。"
    "不要编造看不见的细节，输出控制在 60 到 140 字。"
)


def caption_image(
    settings: AppSettings,
    image_path: Path,
    *,
    ocr_text: str = "",
) -> str:
    if not settings.model.IMAGE_VLM_ENABLED:
        return ""

    provider = settings.model.IMAGE_VLM_PROVIDER.strip().lower().replace("-", "_")
    if provider != "openai_compatible":
        raise ValueError(f"不支持的 IMAGE_VLM_PROVIDER: {settings.model.IMAGE_VLM_PROVIDER}")

    base_url = resolve_image_vlm_base_url(settings)
    api_key = resolve_image_vlm_api_key(settings)
    model = settings.model.IMAGE_VLM_MODEL.strip()
    if not base_url or not api_key or not model:
        raise ValueError("图片视觉描述已启用，但 IMAGE_VLM_BASE_URL / IMAGE_VLM_API_KEY / IMAGE_VLM_MODEL 未完整配置。")

    api_style = resolve_image_vlm_api_style(settings, model=model, base_url=base_url)
    client = build_openai_client(settings, base_url=base_url, api_key=api_key)

    if api_style == "responses":
        response = client.responses.create(
            model=model,
            input=build_openai_responses_input(
                settings=settings,
                image_path=image_path,
                ocr_text=ocr_text,
            ),
            max_output_tokens=settings.model.IMAGE_VLM_MAX_TOKENS,
        )
        return extract_response_text(response).strip()

    if api_style == "chat_completions":
        response = client.chat.completions.create(
            model=model,
            messages=build_openai_chat_messages(
                settings=settings,
                image_path=image_path,
                ocr_text=ocr_text,
            ),
            max_tokens=settings.model.IMAGE_VLM_MAX_TOKENS,
            temperature=0.1,
        )
        return extract_chat_completion_text(response).strip()

    raise ValueError(f"不支持的 IMAGE_VLM_API_STYLE: {api_style}")


def build_openai_client(
    settings: AppSettings,
    *,
    base_url: str,
    api_key: str,
):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "图片视觉描述需要安装 `openai`。"
            "请在当前虚拟环境执行 `pip install openai` 或 `pip install -r requirements.txt`。"
        ) from exc

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=build_httpx_client(timeout=settings.model.IMAGE_VLM_TIMEOUT_SECONDS),
    )


def build_openai_responses_input(
    *,
    settings: AppSettings,
    image_path: Path,
    ocr_text: str = "",
) -> list[dict[str, Any]]:
    prompt = build_image_caption_prompt(settings, ocr_text=ocr_text)
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": build_image_data_url(image_path),
                },
            ],
        }
    ]


def build_openai_chat_messages(
    *,
    settings: AppSettings,
    image_path: Path,
    ocr_text: str = "",
) -> list[dict[str, Any]]:
    prompt = build_image_caption_prompt(settings, ocr_text=ocr_text)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": build_image_data_url(image_path),
                    },
                },
            ],
        }
    ]


def build_image_caption_prompt(
    settings: AppSettings,
    *,
    ocr_text: str,
) -> str:
    prompt_template = settings.model.IMAGE_VLM_PROMPT.strip()
    use_ocr_context = settings.model.IMAGE_VLM_USE_OCR_CONTEXT
    normalized_ocr_text = ocr_text.strip() or "（未识别到明显文字）"

    if prompt_template:
        if use_ocr_context and "{ocr_text}" in prompt_template:
            return prompt_template.format(ocr_text=normalized_ocr_text)
        if use_ocr_context and ocr_text.strip():
            return f"以下是图片中的文字：\n{normalized_ocr_text}\n\n{prompt_template}"
        return prompt_template

    if use_ocr_context:
        return DEFAULT_IMAGE_CAPTION_PROMPT.format(ocr_text=normalized_ocr_text)

    return (
        "请用中文概括这张图片里可以直接确认的内容。"
        "优先描述主体对象、场景、界面、可见文字、颜色或明显状态。"
        "如果存在不确定性，只说明依据不足，不要输出泛化拒答。"
        "不要编造看不见的细节，输出控制在 60 到 140 字。"
    )


def build_image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return extract_response_text_from_dict(dumped)
    return ""


def extract_response_text_from_dict(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    return "\n".join(texts)


def extract_chat_completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts)
    return ""


def resolve_image_vlm_api_style(
    settings: AppSettings,
    *,
    model: str,
    base_url: str,
) -> str:
    configured = settings.model.IMAGE_VLM_API_STYLE.strip().lower().replace("-", "_")
    if configured in {"responses", "chat_completions"}:
        return configured
    if configured not in {"", "auto"}:
        return configured

    if model.startswith("ep-") or "volces.com" in base_url.lower():
        return "responses"
    return "chat_completions"


def resolve_image_vlm_base_url(settings: AppSettings) -> str:
    configured = settings.model.IMAGE_VLM_BASE_URL.strip()
    if configured:
        return configured
    return settings.model.OPENAI_COMPATIBLE_BASE_URL.strip()


def resolve_image_vlm_api_key(settings: AppSettings) -> str:
    configured = settings.model.IMAGE_VLM_API_KEY.strip()
    if configured:
        return configured
    return settings.model.OPENAI_COMPATIBLE_API_KEY.strip()
