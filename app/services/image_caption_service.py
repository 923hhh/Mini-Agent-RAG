from __future__ import annotations

import base64
import json
import mimetypes
import re
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from app.services.network import build_httpx_client
from app.services.settings import AppSettings


DEFAULT_IMAGE_CAPTION_PROMPT = (
    "请根据图片和可见文字，只提取能直接确认的内容。"
    "优先关注主体、场景、关键文字和明显状态。"
    "不要解释过程，不要编造。"
)

STRUCTURED_CAPTION_JSON_PROMPT = (
    "只输出 JSON，不要输出解释。"
    "字段固定为 "
    "{\"summary\":\"\",\"scene_type\":\"\",\"primary_objects\":[],\"visible_text_cues\":[],\"actions_or_states\":[],\"uncertainties\":[]}。"
    "`summary` 用 20 到 60 字中文；无法确定就返回空字符串或空数组。"
)


@dataclass(frozen=True)
class StructuredImageCaption:
    summary: str = ""
    scene_type: str = ""
    primary_objects: tuple[str, ...] = ()
    visible_text_cues: tuple[str, ...] = ()
    actions_or_states: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    raw_text: str = ""
    region_label: str = ""


def caption_image(
    settings: AppSettings,
    image_path: Path,
    *,
    ocr_text: str = "",
    task_type: str = "general",
) -> str:
    caption = caption_image_structured(
        settings,
        image_path,
        ocr_text=ocr_text,
        task_type=task_type,
    )
    return format_structured_image_caption(caption)


def caption_image_structured(
    settings: AppSettings,
    image_path: Path,
    *,
    ocr_text: str = "",
    region_label: str = "",
    task_type: str = "general",
) -> StructuredImageCaption:
    raw_text = request_image_caption(
        settings,
        image_data_url=build_image_data_url(
            image_path,
            max_side=settings.model.IMAGE_VLM_MAX_SIDE,
        ),
        ocr_text=ocr_text,
        region_label=region_label,
        task_type=task_type,
    )
    return parse_structured_image_caption(raw_text, region_label=region_label)


def caption_image_bytes_structured(
    settings: AppSettings,
    *,
    image_bytes: bytes,
    mime_type: str,
    ocr_text: str = "",
    region_label: str = "",
    task_type: str = "general",
) -> StructuredImageCaption:
    raw_text = request_image_caption(
        settings,
        image_data_url=build_image_data_url_from_bytes(
            image_bytes,
            mime_type=mime_type,
            max_side=settings.model.IMAGE_VLM_MAX_SIDE,
        ),
        ocr_text=ocr_text,
        region_label=region_label,
        task_type=task_type,
    )
    return parse_structured_image_caption(raw_text, region_label=region_label)


def request_image_caption(
    settings: AppSettings,
    *,
    image_data_url: str,
    ocr_text: str = "",
    region_label: str = "",
    task_type: str = "general",
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
                image_data_url=image_data_url,
                ocr_text=ocr_text,
                region_label=region_label,
                task_type=task_type,
            ),
            max_output_tokens=settings.model.IMAGE_VLM_MAX_TOKENS,
        )
        return extract_response_text(response).strip()

    if api_style == "chat_completions":
        response = client.chat.completions.create(
            model=model,
            messages=build_openai_chat_messages(
                settings=settings,
                image_data_url=image_data_url,
                ocr_text=ocr_text,
                region_label=region_label,
                task_type=task_type,
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
    image_data_url: str,
    ocr_text: str = "",
    region_label: str = "",
    task_type: str = "general",
) -> list[dict[str, Any]]:
    prompt = build_image_caption_prompt(
        settings,
        ocr_text=ocr_text,
        region_label=region_label,
        task_type=task_type,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": image_data_url,
                },
            ],
        }
    ]


def build_openai_chat_messages(
    *,
    settings: AppSettings,
    image_data_url: str,
    ocr_text: str = "",
    region_label: str = "",
    task_type: str = "general",
) -> list[dict[str, Any]]:
    prompt = build_image_caption_prompt(
        settings,
        ocr_text=ocr_text,
        region_label=region_label,
        task_type=task_type,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                    },
                },
            ],
        }
    ]


def build_image_caption_prompt(
    settings: AppSettings,
    *,
    ocr_text: str,
    region_label: str = "",
    task_type: str = "general",
) -> str:
    prompt_template = settings.model.IMAGE_VLM_PROMPT.strip()
    use_ocr_context = settings.model.IMAGE_VLM_USE_OCR_CONTEXT
    normalized_ocr_text = ocr_text.strip() or "（未识别到明显文字）"
    region_hint = f"分析范围：{region_label or '全图'}。"

    if prompt_template:
        if use_ocr_context and "{ocr_text}" in prompt_template:
            base_prompt = prompt_template.format(ocr_text=normalized_ocr_text)
        elif use_ocr_context and ocr_text.strip():
            base_prompt = f"以下是图片中的文字：\n{normalized_ocr_text}\n\n{prompt_template}"
        else:
            base_prompt = prompt_template
    elif use_ocr_context:
        base_prompt = (
            "以下是图片中的文字：\n"
            f"{normalized_ocr_text}\n\n"
            f"{DEFAULT_IMAGE_CAPTION_PROMPT}"
        )
    else:
        base_prompt = DEFAULT_IMAGE_CAPTION_PROMPT

    task_instruction = resolve_caption_task_instruction(
        task_type=task_type,
        region_label=region_label,
    )

    return "\n\n".join(
        part
        for part in (
            region_hint,
            base_prompt,
            task_instruction,
            STRUCTURED_CAPTION_JSON_PROMPT,
        )
        if part
    )


def resolve_caption_task_instruction(
    *,
    task_type: str,
    region_label: str = "",
) -> str:
    normalized_task_type = (task_type or "general").strip().lower()
    if normalized_task_type == "instruction_figure":
        return (
            "这是一张说明书、维修手册或操作页中的配图。"
            "优先识别工具、零部件、被操作对象、装配关系和局部特征。"
            "如果画面里有箭头、标记、放大图，请说明它指向什么。"
            "不要复述整页无关文字。"
        )
    if normalized_task_type == "arrow_focus":
        return (
            "这是说明书配图中的箭头或局部放大区域。"
            "优先说明红色箭头、标记线或放大框指向的部件、孔位、接触点或操作对象。"
            f"如果分析范围是 {region_label or '局部区域'}，只描述该局部。"
            "不要扩展到整页其他区域。"
        )
    return ""


def build_image_data_url(
    image_path: Path,
    *,
    max_side: int = 0,
) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return build_image_data_url_from_bytes(
        image_path.read_bytes(),
        mime_type=mime_type,
        max_side=max_side,
    )


def build_image_data_url_from_bytes(
    image_bytes: bytes,
    *,
    mime_type: str,
    max_side: int = 0,
) -> str:
    prepared_bytes, prepared_mime_type = prepare_image_bytes_for_vlm(
        image_bytes,
        mime_type=mime_type,
        max_side=max_side,
    )
    encoded = base64.b64encode(prepared_bytes).decode("ascii")
    return f"data:{prepared_mime_type};base64,{encoded}"


def prepare_image_bytes_for_vlm(
    image_bytes: bytes,
    *,
    mime_type: str,
    max_side: int,
) -> tuple[bytes, str]:
    if max_side <= 0:
        return image_bytes, mime_type

    try:
        from PIL import Image
    except ImportError:
        return image_bytes, mime_type

    try:
        with warnings.catch_warnings():
            if hasattr(Image, "DecompressionBombWarning"):
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as original_image:
                prepared_image = resize_image_for_vlm(original_image, max_side=max_side)
                if prepared_image is original_image:
                    return image_bytes, mime_type

                output_format, output_mime_type = resolve_vlm_output_format(
                    original_image=original_image,
                    fallback_mime_type=mime_type,
                )
                buffer = BytesIO()
                save_kwargs: dict[str, Any] = {"format": output_format}
                if output_format in {"JPEG", "WEBP"}:
                    save_kwargs["quality"] = 90
                prepared_image.save(buffer, **save_kwargs)
                return buffer.getvalue(), output_mime_type
    except Exception:
        return image_bytes, mime_type


def resize_image_for_vlm(
    image,
    *,
    max_side: int,
):
    try:
        from PIL import Image
    except ImportError:
        return image

    if max_side <= 0:
        return image

    longest_side = max(image.width, image.height)
    if longest_side <= max_side:
        return image

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.LANCZOS

    prepared = image.convert("RGBA" if has_alpha_channel(image) else "RGB")
    scale = max_side / max(1, longest_side)
    return prepared.resize(
        (
            max(1, int(prepared.width * scale)),
            max(1, int(prepared.height * scale)),
        ),
        resample=resample,
    )


def has_alpha_channel(image) -> bool:
    bands = getattr(image, "getbands", lambda: ())()
    return "A" in bands


def resolve_vlm_output_format(
    *,
    original_image,
    fallback_mime_type: str,
) -> tuple[str, str]:
    normalized_mime_type = (fallback_mime_type or "").lower()
    if has_alpha_channel(original_image) or "png" in normalized_mime_type:
        return "PNG", "image/png"
    if "webp" in normalized_mime_type:
        return "WEBP", "image/webp"
    return "JPEG", "image/jpeg"


def parse_structured_image_caption(
    text: str,
    *,
    region_label: str = "",
) -> StructuredImageCaption:
    raw_text = (text or "").strip()
    payload = extract_structured_caption_payload(raw_text)

    summary = ""
    scene_type = ""
    primary_objects: tuple[str, ...] = ()
    visible_text_cues: tuple[str, ...] = ()
    actions_or_states: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()

    if payload is not None:
        summary = normalize_text_value(
            payload.get("summary")
            or payload.get("overview")
            or payload.get("description")
        )
        scene_type = normalize_text_value(
            payload.get("scene_type")
            or payload.get("scene")
            or payload.get("sceneType")
        )
        primary_objects = normalize_text_list(
            payload.get("primary_objects") or payload.get("objects")
        )
        visible_text_cues = normalize_text_list(
            payload.get("visible_text_cues")
            or payload.get("visible_text")
            or payload.get("text_cues")
        )
        actions_or_states = normalize_text_list(
            payload.get("actions_or_states")
            or payload.get("states")
            or payload.get("actions")
        )
        uncertainties = normalize_text_list(
            payload.get("uncertainties")
            or payload.get("limitations")
        )

    if not summary:
        summary = normalize_text_value(raw_text)
    if not summary:
        summary = build_fallback_structured_summary(
            scene_type=scene_type,
            primary_objects=primary_objects,
            visible_text_cues=visible_text_cues,
            actions_or_states=actions_or_states,
        )

    return StructuredImageCaption(
        summary=summary,
        scene_type=scene_type,
        primary_objects=primary_objects,
        visible_text_cues=visible_text_cues,
        actions_or_states=actions_or_states,
        uncertainties=uncertainties,
        raw_text=raw_text,
        region_label=region_label,
    )


def extract_structured_caption_payload(text: str) -> dict[str, Any] | None:
    normalized = strip_json_fences(text)
    if not normalized:
        return None

    candidate_texts: list[str] = [normalized]
    first_brace = normalized.find("{")
    last_brace = normalized.rfind("}")
    if 0 <= first_brace < last_brace:
        candidate_texts.append(normalized[first_brace : last_brace + 1])

    for candidate in candidate_texts:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def strip_json_fences(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def normalize_text_value(value: object) -> str:
    if isinstance(value, str):
        normalized = re.sub(r"\s+", " ", value).strip()
        return normalized
    return ""


def normalize_text_list(value: object) -> tuple[str, ...]:
    items: list[str] = []
    if isinstance(value, str):
        candidates = re.split(r"[、,，;；/\n]+", value)
        items.extend(normalize_text_value(item) for item in candidates)
    elif isinstance(value, (list, tuple)):
        items.extend(normalize_text_value(item) for item in value)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return tuple(deduped)


def build_fallback_structured_summary(
    *,
    scene_type: str,
    primary_objects: tuple[str, ...],
    visible_text_cues: tuple[str, ...],
    actions_or_states: tuple[str, ...],
) -> str:
    parts: list[str] = []
    if scene_type:
        parts.append(f"场景为{scene_type}")
    if primary_objects:
        parts.append(f"主体包括{'、'.join(primary_objects[:3])}")
    if visible_text_cues:
        parts.append(f"可见文字线索有{'、'.join(visible_text_cues[:3])}")
    if actions_or_states:
        parts.append(f"画面体现{'、'.join(actions_or_states[:2])}")
    return "；".join(parts)


def format_structured_image_caption(
    caption: StructuredImageCaption,
    *,
    include_region_label: bool = False,
) -> str:
    lines: list[str] = []
    if include_region_label and caption.region_label:
        lines.append(f"区域: {caption.region_label}")
    if caption.summary:
        lines.append(f"摘要: {caption.summary}")
    if caption.scene_type:
        lines.append(f"场景: {caption.scene_type}")
    if caption.primary_objects:
        lines.append(f"主体: {'、'.join(caption.primary_objects[:6])}")
    if caption.visible_text_cues:
        lines.append(f"文字线索: {'、'.join(caption.visible_text_cues[:6])}")
    if caption.actions_or_states:
        lines.append(f"动作/状态: {'、'.join(caption.actions_or_states[:6])}")
    if caption.uncertainties:
        lines.append(f"不确定点: {'、'.join(caption.uncertainties[:4])}")
    return "\n".join(lines).strip()


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
