from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from app.services.settings import AppSettings


def get_settings(request: Request) -> AppSettings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise RuntimeError("应用设置尚未初始化。")
    return cast(AppSettings, settings)


SettingsDep = Annotated[AppSettings, Depends(get_settings)]
