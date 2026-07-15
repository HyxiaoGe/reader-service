"""
reader-service — 网页内容读取微服务
第一阶段：代理 Jina Reader，返回结构化网页内容
"""

import asyncio
import logging
import os
import time
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="Reader Service")
logger = logging.getLogger("reader_service")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

JINA_BASE_URL = "https://r.jina.ai"
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_CONNECT_TIMEOUT = float(os.getenv("JINA_CONNECT_TIMEOUT", "3"))
JINA_MIN_READ_TIMEOUT = 10.0
JINA_READ_TIMEOUT = max(
    JINA_MIN_READ_TIMEOUT,
    float(os.getenv("JINA_READ_TIMEOUT", os.getenv("JINA_TIMEOUT", "16"))),
)
JINA_WRITE_TIMEOUT = float(os.getenv("JINA_WRITE_TIMEOUT", "3"))
JINA_POOL_TIMEOUT = float(os.getenv("JINA_POOL_TIMEOUT", "2"))
JINA_TOTAL_TIMEOUT = float(os.getenv("JINA_TOTAL_TIMEOUT", "18"))
JINA_RETRY_BACKOFF = float(os.getenv("JINA_RETRY_BACKOFF", "0.2"))
JINA_MIN_RETRY_WINDOW = max(
    JINA_MIN_READ_TIMEOUT,
    float(os.getenv("JINA_MIN_RETRY_WINDOW", "10")),
)
JINA_MAX_ATTEMPTS = 2
GOOGLE_FAVICON_API = "https://www.google.com/s2/favicons?sz=32&domain="


class ReadResponse(BaseModel):
    url: str
    title: str | None = None
    content: str
    favicon: str | None = None
    content_length: int
    fetch_ms: int
    attempts: int = 1


def _monotonic() -> float:
    return time.monotonic()


def _timeout_for(remaining_seconds: float) -> httpx.Timeout:
    """让每个阶段的超时都受本次请求剩余总预算约束。"""
    return httpx.Timeout(
        connect=min(JINA_CONNECT_TIMEOUT, remaining_seconds),
        read=min(JINA_READ_TIMEOUT, remaining_seconds),
        write=min(JINA_WRITE_TIMEOUT, remaining_seconds),
        pool=min(JINA_POOL_TIMEOUT, remaining_seconds),
    )


async def _fetch_jina(
    jina_url: str, headers: dict[str, str], timeout: httpx.Timeout
) -> httpx.Response:
    """执行单次读取；由调用方的总 deadline 包住客户端创建、请求与关闭。"""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(jina_url, headers=headers)
        response.raise_for_status()
        return response


def _error_detail(
    *,
    kind: str,
    message: str,
    retryable: bool,
    upstream_status: int | None,
    attempts: int,
    started_at: float,
) -> dict:
    return {
        "kind": kind,
        "message": message,
        "retryable": retryable,
        "upstream_status": upstream_status,
        "attempts": attempts,
        "duration_ms": max(0, int((_monotonic() - started_at) * 1000)),
    }


def _log_failure(domain: str, status_code: int, detail: dict) -> None:
    logger.warning(
        "reader_read_failed domain=%s kind=%s status=%s upstream_status=%s "
        "attempts=%s duration_ms=%s",
        domain,
        detail["kind"],
        status_code,
        detail["upstream_status"],
        detail["attempts"],
        detail["duration_ms"],
    )


def _raise_reader_error(*, status_code: int, domain: str, detail: dict) -> None:
    _log_failure(domain, status_code, detail)
    raise HTTPException(status_code=status_code, detail=detail) from None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/read", response_model=ReadResponse)
async def read_url(url: str = Query(..., description="要读取的网页 URL")):
    # 校验 URL 格式
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        raise HTTPException(status_code=422, detail="URL 格式无效") from None
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not hostname:
        raise HTTPException(status_code=422, detail="URL 格式无效")

    # 调用 Jina Reader
    jina_url = f"{JINA_BASE_URL}/{url}"
    headers = {"Accept": "text/markdown"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"

    start = _monotonic()
    deadline = start + JINA_TOTAL_TIMEOUT
    domain = hostname
    attempts = 0
    error_status = 504
    error_kind = "timeout"
    error_message = "Jina Reader 请求超时"
    error_retryable = True
    upstream_status = None

    while attempts < JINA_MAX_ATTEMPTS:
        remaining = JINA_TOTAL_TIMEOUT if attempts == 0 else deadline - _monotonic()
        if remaining <= 0:
            detail = _error_detail(
                kind=error_kind,
                message=error_message,
                retryable=error_retryable,
                upstream_status=upstream_status,
                attempts=attempts,
                started_at=start,
            )
            _raise_reader_error(status_code=error_status, domain=domain, detail=detail)

        attempts += 1

        try:
            timeout = _timeout_for(remaining)
            resp = await asyncio.wait_for(
                _fetch_jina(jina_url, headers, timeout), timeout=remaining
            )
            if _monotonic() > deadline:
                raise asyncio.TimeoutError
            break
        except (httpx.TimeoutException, asyncio.TimeoutError):
            error_status = 504
            error_kind = "timeout"
            error_message = "Jina Reader 请求超时"
            error_retryable = True
            upstream_status = None
        except httpx.HTTPStatusError as exc:
            upstream_status = exc.response.status_code
            error_status = 502
            if upstream_status in (401, 403):
                error_kind = "upstream_auth"
                error_message = "Jina Reader 上游鉴权失败"
                error_retryable = False
            elif upstream_status == 429:
                error_status = 503
                error_kind = "rate_limited"
                error_message = "Jina Reader 上游限流"
                error_retryable = True
            else:
                error_kind = "upstream_error"
                error_message = f"Jina Reader 上游返回 HTTP {upstream_status}"
                error_retryable = upstream_status >= 500
        except httpx.RequestError:
            error_status = 502
            error_kind = "request_error"
            error_message = "Jina Reader 请求失败"
            error_retryable = True
            upstream_status = None
        except Exception:
            error_status = 502
            error_kind = "request_error"
            error_message = "Jina Reader 请求失败"
            error_retryable = False
            upstream_status = None

        remaining = deadline - _monotonic()
        can_retry = (
            error_retryable
            and attempts < JINA_MAX_ATTEMPTS
            and remaining - JINA_RETRY_BACKOFF >= JINA_MIN_RETRY_WINDOW
        )
        if can_retry:
            await asyncio.sleep(min(JINA_RETRY_BACKOFF, remaining))
            continue

        detail = _error_detail(
            kind=error_kind,
            message=error_message,
            retryable=error_retryable,
            upstream_status=upstream_status,
            attempts=attempts,
            started_at=start,
        )
        _raise_reader_error(status_code=error_status, domain=domain, detail=detail)

    fetch_ms = int((_monotonic() - start) * 1000)
    content = resp.text
    logger.info(
        "reader_read_succeeded domain=%s status=200 attempts=%s duration_ms=%s",
        domain,
        attempts,
        fetch_ms,
    )

    # 提取标题：优先 Jina X-Title 头 → "Title: xxx" 行 → 首个 # 标题
    title = resp.headers.get("x-title")
    if not title:
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Title: "):
                title = stripped[7:].strip()
                break
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

    # 提取 favicon
    domain = parsed.netloc
    favicon = f"{GOOGLE_FAVICON_API}{domain}"

    return ReadResponse(
        url=url,
        title=title,
        content=content,
        favicon=favicon,
        content_length=len(content),
        fetch_ms=fetch_ms,
        attempts=attempts,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8091, access_log=False)
