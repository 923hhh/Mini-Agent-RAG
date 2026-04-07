from __future__ import annotations

from fastapi import FastAPI

from app.api.errors import install_exception_handlers
from app.api.chat import router as chat_router
from app.api.knowledge_base import router as knowledge_base_router
from app.api.tools import router as tools_router


app = FastAPI(title="Mini Agent RAG API", version="0.1.0")
install_exception_handlers(app)
app.include_router(chat_router)
app.include_router(knowledge_base_router)
app.include_router(tools_router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
