from __future__ import annotations

import os

import httpx
import requests


PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

BROKEN_PROXY_TARGETS = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
    "http://localhost:9",
    "https://localhost:9",
}


def should_ignore_invalid_env_proxies() -> bool:
    for env_name in PROXY_ENV_NAMES:
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            continue
        normalized = raw_value.rstrip("/").lower()
        if normalized in BROKEN_PROXY_TARGETS:
            return True
    return False


def build_requests_session() -> requests.Session:
    session = requests.Session()
    if should_ignore_invalid_env_proxies():
        session.trust_env = False
    return session


def build_httpx_client(*, timeout: float | None = None) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        trust_env=not should_ignore_invalid_env_proxies(),
    )
