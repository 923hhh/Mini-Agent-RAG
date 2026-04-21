"""提供工具调用与工具信息相关的 API 路由。"""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.chat import ToolDefinition
from app.tools.registry import list_tools


router = APIRouter(tags=["tools"])


@router.get("/tools", response_model=list[ToolDefinition])
def get_tools() -> list[ToolDefinition]:
    return list_tools()
