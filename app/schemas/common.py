"""定义多个模块共用的基础数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ApiErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: Any | None = None
