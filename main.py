"""
reader-service — 网页内容读取微服务
优先 Tavily Extract 读取正文，失败或取不到正文时回退 Jina Reader
"""

import asyncio
import logging
import os
import time
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
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
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
# 实测 Tavily p90 约 3s；超过上限就把剩余总预算留给 Jina 回退
TAVILY_TIMEOUT = float(os.getenv("TAVILY_TIMEOUT", "8"))
# Tavily 账户级失败：密钥无效 / 套餐额度用完 / 按量额度用完
TAVILY_ACCOUNT_FAILURE_STATUSES = (401, 432, 433)
GOOGLE_FAVICON_API = "https://www.google.com/s2/favicons?sz=32&domain="
# Jina 账户级失败（密钥失效/余额不足）：换哪个网页都读不了，需要人处理
ACCOUNT_FAILURE_STATUSES = (401, 402, 403)

# 最近一次读取暴露的账户级故障；/health 据此报 503 让哨兵告警，下次读取成功即清除。
# 只在真实读取时更新，健康检查本身不调用 Jina、不消耗额度。
_account_failure: dict | None = None
_tavily_account_failure: dict | None = None


class ReadResponse(BaseModel):
    url: str
    title: str | None = None
    content: str
    favicon: str | None = None
    content_length: int
    fetch_ms: int
    attempts: int = 1
    provider: str = "jina"


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


async def _fetch_tavily(url: str, timeout: float) -> dict:
    """单次 Tavily Extract；返回该 URL 的结果，失败或无结果时返回空 dict。"""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            TAVILY_EXTRACT_URL,
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            json={"urls": [url], "extract_depth": "basic", "format": "markdown"},
        )
        response.raise_for_status()
        payload = response.json()
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return {}
    return results[0]


async def _read_with_tavily(url: str, domain: str) -> tuple[str, str | None] | None:
    """Tavily 读取成功返回 (正文, 标题)；任何失败都返回 None，交给 Jina 回退。"""
    global _tavily_account_failure
    started_at = _monotonic()
    reason = "empty_content"
    upstream_status = None
    try:
        result = await asyncio.wait_for(
            _fetch_tavily(url, TAVILY_TIMEOUT), timeout=TAVILY_TIMEOUT
        )
        content = result.get("raw_content")
        if isinstance(content, str) and content.strip():
            _tavily_account_failure = None
            title = result.get("title")
            return content, title if isinstance(title, str) and title.strip() else None
    except (httpx.TimeoutException, asyncio.TimeoutError):
        reason = "timeout"
    except httpx.HTTPStatusError as exc:
        upstream_status = exc.response.status_code
        reason = "upstream_error"
        if upstream_status in TAVILY_ACCOUNT_FAILURE_STATUSES:
            reason = "upstream_auth"
            _tavily_account_failure = {
                "upstream_status": upstream_status,
                "message": "Tavily 密钥无效或额度已用完",
                "since": _tavily_account_failure["since"]
                if _tavily_account_failure
                else time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
    except Exception:
        reason = "request_error"
    logger.warning(
        "reader_tavily_fallback domain=%s kind=%s upstream_status=%s duration_ms=%s",
        domain,
        reason,
        upstream_status,
        max(0, int((_monotonic() - started_at) * 1000)),
    )
    return None


def _extract_title(content: str, header_title: str | None) -> str | None:
    """优先上游给的标题 → "Title: xxx" 行 → 首个 # 标题。"""
    if header_title:
        return header_title
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Title: "):
            return stripped[7:].strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


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


def _set_account_failure(upstream_status: int | None, message: str) -> None:
    global _account_failure
    if upstream_status in ACCOUNT_FAILURE_STATUSES:
        _account_failure = {
            "upstream_status": upstream_status,
            "message": message,
            "since": _account_failure["since"]
            if _account_failure
            else time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }


def _clear_account_failure() -> None:
    global _account_failure
    _account_failure = None


@app.get("/health")
async def health():
    if _account_failure or _tavily_account_failure:
        content: dict = {"status": "degraded"}
        if _account_failure:
            content["jina_account"] = _account_failure
        if _tavily_account_failure:
            content["tavily_account"] = _tavily_account_failure
        return JSONResponse(status_code=503, content=content)
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

    start = _monotonic()
    domain = hostname
    favicon = f"{GOOGLE_FAVICON_API}{parsed.netloc}"

    # Tavily 尝试计入 attempts；失败后 Jina 只用剩余的总预算
    tavily_attempts = 0
    first_budget = JINA_TOTAL_TIMEOUT
    if TAVILY_API_KEY:
        tavily_attempts = 1
        extracted = await _read_with_tavily(url, domain)
        if extracted is not None:
            content, title = extracted
            fetch_ms = int((_monotonic() - start) * 1000)
            logger.info(
                "reader_read_succeeded domain=%s provider=tavily status=200 "
                "attempts=1 duration_ms=%s",
                domain,
                fetch_ms,
            )
            return ReadResponse(
                url=url,
                title=_extract_title(content, title),
                content=content,
                favicon=favicon,
                content_length=len(content),
                fetch_ms=fetch_ms,
                attempts=1,
                provider="tavily",
            )
        first_budget = JINA_TOTAL_TIMEOUT - (_monotonic() - start)

    # 调用 Jina Reader
    jina_url = f"{JINA_BASE_URL}/{url}"
    headers = {"Accept": "text/markdown"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"

    deadline = start + JINA_TOTAL_TIMEOUT
    attempts = 0
    error_status = 504
    error_kind = "timeout"
    error_message = "Jina Reader 请求超时"
    error_retryable = True
    upstream_status = None

    while attempts < JINA_MAX_ATTEMPTS:
        remaining = first_budget if attempts == 0 else deadline - _monotonic()
        if remaining <= 0:
            detail = _error_detail(
                kind=error_kind,
                message=error_message,
                retryable=error_retryable,
                upstream_status=upstream_status,
                attempts=attempts + tavily_attempts,
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
            elif upstream_status == 402:
                error_kind = "upstream_auth"
                error_message = "Jina Reader 账户余额不足"
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

        _set_account_failure(upstream_status, error_message)
        detail = _error_detail(
            kind=error_kind,
            message=error_message,
            retryable=error_retryable,
            upstream_status=upstream_status,
            attempts=attempts + tavily_attempts,
            started_at=start,
        )
        _raise_reader_error(status_code=error_status, domain=domain, detail=detail)

    _clear_account_failure()
    fetch_ms = int((_monotonic() - start) * 1000)
    content = resp.text
    logger.info(
        "reader_read_succeeded domain=%s provider=jina status=200 attempts=%s "
        "duration_ms=%s",
        domain,
        attempts + tavily_attempts,
        fetch_ms,
    )

    return ReadResponse(
        url=url,
        title=_extract_title(content, resp.headers.get("x-title")),
        content=content,
        favicon=favicon,
        content_length=len(content),
        fetch_ms=fetch_ms,
        attempts=attempts + tavily_attempts,
        provider="jina",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8091, access_log=False)
