from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_KB_NAME = "quickstart_demo"
DEFAULT_QUESTION = "什么是 RAG？"
DEFAULT_SAMPLE_FILE = "intro.txt"
DEFAULT_SAMPLE_TEXT = (
    "RAG（检索增强生成）是一种结合检索与大模型生成的问答方法。"
    "它会先从知识库中检索与问题相关的内容，再把这些证据作为上下文交给大模型生成答案。"
    "这样可以降低模型凭空编造内容的概率，并提升回答的可解释性。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_rag_quickstart",
        description="从 0 到跑通一个最小可用的本地知识库 RAG 示例。",
    )
    parser.add_argument("--kb-name", default=DEFAULT_KB_NAME, help="知识库名称。")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="发送给 /chat/rag 的问题。")
    parser.add_argument(
        "--sample-file-name",
        default=DEFAULT_SAMPLE_FILE,
        help="当知识库为空时自动写入的示例文件名。",
    )
    parser.add_argument(
        "--sample-text",
        default=DEFAULT_SAMPLE_TEXT,
        help="当知识库为空时自动写入的示例文本内容。",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="API 监听地址，同时用于拼装请求地址。",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="API 监听端口，同时用于拼装请求地址。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="RAG 查询 top_k。",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="RAG 查询 score_threshold。",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="等待 API 启动成功的超时时间（秒）。",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="接口请求超时时间（秒）。",
    )
    parser.add_argument(
        "--install-requirements",
        action="store_true",
        help="运行前先执行 pip install -r requirements.txt。",
    )
    parser.add_argument(
        "--keep-api-running",
        action="store_true",
        help="如果脚本帮你启动了 API，完成后不自动关闭。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    python_executable = Path(sys.executable)

    if args.install_requirements:
        install_requirements(python_executable)

    sample_path = ensure_sample_knowledge_base(
        kb_name=args.kb_name,
        sample_file_name=args.sample_file_name,
        sample_text=args.sample_text,
    )
    print(f"[quickstart] 知识库目录已就绪: {sample_path.parent}")
    if sample_path.exists():
        print(f"[quickstart] 示例文件: {sample_path}")

    api_process: subprocess.Popen[str] | None = None
    api_was_running = api_is_healthy(base_url, timeout=args.request_timeout)
    if api_was_running:
        print(f"[quickstart] 检测到 API 已运行: {base_url}")
    else:
        api_process = start_api(
            python_executable=python_executable,
            host=args.host,
            port=args.port,
        )
        print(f"[quickstart] 已启动 API 进程，PID={api_process.pid}")
        wait_for_api(base_url, timeout_seconds=args.startup_timeout, request_timeout=args.request_timeout)
        print(f"[quickstart] API 健康检查通过: {base_url}/health")

    try:
        rebuild_knowledge_base(python_executable=python_executable, kb_name=args.kb_name)
        response = query_rag(
            base_url=base_url,
            kb_name=args.kb_name,
            question=args.question,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            request_timeout=args.request_timeout,
        )
    finally:
        if api_process is not None and not args.keep_api_running:
            stop_process(api_process)

    render_result(response, base_url=base_url, kb_name=args.kb_name, question=args.question)
    return 0


def install_requirements(python_executable: Path) -> None:
    command = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "-r",
        str(PROJECT_ROOT / "requirements.txt"),
    ]
    print("[quickstart] 安装依赖中...")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_sample_knowledge_base(
    *,
    kb_name: str,
    sample_file_name: str,
    sample_text: str,
) -> Path:
    content_dir = PROJECT_ROOT / "data" / "knowledge_base" / kb_name / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    sample_path = content_dir / sample_file_name
    supported_files = [path for path in content_dir.rglob("*") if path.is_file()]
    if supported_files:
        return sample_path

    sample_path.write_text(sample_text.strip() + "\n", encoding="utf-8")
    return sample_path


def start_api(
    *,
    python_executable: Path,
    host: str,
    port: int,
) -> subprocess.Popen[str]:
    logs_dir = PROJECT_ROOT / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "quickstart_api.stdout.log"
    stderr_path = logs_dir / "quickstart_api.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    command = [
        str(python_executable),
        str(PROJECT_ROOT / "scripts" / "start_api.py"),
        "--host",
        host,
        "--port",
        str(port),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        creationflags=creationflags,
    )


def wait_for_api(
    base_url: str,
    *,
    timeout_seconds: float,
    request_timeout: float,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        if api_is_healthy(base_url, timeout=request_timeout):
            return
        time.sleep(1.0)
    raise RuntimeError(
        f"等待 API 启动超时: {base_url}。"
        "你可以查看 data/logs/quickstart_api.stdout.log 和 quickstart_api.stderr.log。"
        + (f" 最近一次错误: {last_error}" if last_error else "")
    )


def api_is_healthy(base_url: str, *, timeout: float) -> bool:
    try:
        payload = request_json(
            "GET",
            f"{base_url.rstrip('/')}/health",
            timeout=timeout,
        )
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def rebuild_knowledge_base(*, python_executable: Path, kb_name: str) -> None:
    command = [
        str(python_executable),
        str(PROJECT_ROOT / "scripts" / "rebuild_kb.py"),
        "--kb-name",
        kb_name,
    ]
    print(f"[quickstart] 开始构建知识库: {kb_name}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def query_rag(
    *,
    base_url: str,
    kb_name: str,
    question: str,
    top_k: int,
    score_threshold: float,
    request_timeout: float,
) -> dict[str, Any]:
    payload = {
        "query": question,
        "source_type": "local_kb",
        "knowledge_base_name": kb_name,
        "knowledge_id": "",
        "top_k": top_k,
        "score_threshold": score_threshold,
        "history": [],
        "stream": False,
    }
    print(f"[quickstart] 开始查询 /chat/rag: {question}")
    response = request_json(
        "POST",
        f"{base_url.rstrip('/')}/chat/rag",
        json_body=payload,
        timeout=request_timeout,
    )
    if not isinstance(response, dict):
        raise RuntimeError(f"/chat/rag 返回了非 JSON 对象: {response!r}")
    return response


def request_json(
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float,
) -> Any:
    headers = {"Accept": "application/json"}
    data: bytes | None = None
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
    request = Request(url=url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} 请求失败: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接到 {url}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"响应不是合法 JSON: {raw}") from exc


def render_result(
    response: dict[str, Any],
    *,
    base_url: str,
    kb_name: str,
    question: str,
) -> None:
    references = response.get("references", [])
    answer = str(response.get("answer", "")).strip()
    print()
    print("========== RAG 跑通结果 ==========")
    print(f"API: {base_url}")
    print(f"知识库: {kb_name}")
    print(f"问题: {question}")
    print()
    print("回答:")
    print(answer or "(空回答)")
    print()
    print(f"引用数量: {len(references) if isinstance(references, list) else 0}")
    if isinstance(references, list):
        for index, item in enumerate(references[:5], start=1):
            if not isinstance(item, dict):
                continue
            source = item.get("source", "")
            score = item.get("relevance_score", "")
            preview = str(item.get("content_preview", "")).replace("\n", " ").strip()
            print(f"[{index}] source={source} relevance={score} preview={preview[:120]}")
    print("=================================")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=5)
        else:
            process.terminate()
            process.wait(timeout=5)
    except Exception:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
