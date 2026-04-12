"""
reader-service — 网页内容读取微服务
第一阶段：代理 Jina Reader，返回结构化网页内容
"""

import os
import time
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="Reader Service")

JINA_BASE_URL = "https://r.jina.ai"
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_TIMEOUT = float(os.getenv("JINA_TIMEOUT", "10"))
GOOGLE_FAVICON_API = "https://www.google.com/s2/favicons?sz=32&domain="


class ReadResponse(BaseModel):
    url: str
    title: str | None = None
    content: str
    favicon: str | None = None
    content_length: int
    fetch_ms: int


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/read", response_model=ReadResponse)
async def read_url(url: str = Query(..., description="要读取的网页 URL")):
    # 校验 URL 格式
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=422, detail=f"URL 格式无效: {url}")

    # 调用 Jina Reader
    jina_url = f"{JINA_BASE_URL}/{url}"
    headers = {"Accept": "text/markdown"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=JINA_TIMEOUT) as client:
            resp = await client.get(jina_url, headers=headers)
            resp.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=502, detail="Jina Reader 超时")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Jina Reader 返回 {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Jina Reader 请求失败: {e}")

    fetch_ms = int((time.monotonic() - start) * 1000)
    content = resp.text

    # 提取标题：优先 Jina 返回的 X-Title 头，其次 Markdown 首行 #
    title = resp.headers.get("x-title")
    if not title:
        for line in content.split("\n"):
            stripped = line.strip()
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
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8091)
