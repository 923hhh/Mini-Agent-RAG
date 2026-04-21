"""定义 API 层统一使用的错误类型与处理。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def error_payload(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "details": details,
    }


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def handle_http_exception(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        del request
        if isinstance(exc.detail, dict):
            payload = error_payload(
                code=str(exc.detail.get("code", "http_error")),
                message=str(exc.detail.get("message", "请求失败")),
                details=exc.detail.get("details"),
            )
        else:
            payload = error_payload(
                code="http_error",
                message=str(exc.detail),
                details=None,
            )
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=422,
            content=error_payload(
                code="validation_error",
                message="请求参数校验失败。",
                details=exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=500,
            content=error_payload(
                code="internal_error",
                message="服务器内部错误。",
                details=str(exc),
            ),
        )
