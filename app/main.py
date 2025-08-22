import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import (
    LOG_ROTATE_KEEP,
    LOG_ROTATE_MAX_BYTES,
    TOKEN_LOG_PATH,
    UPSTREAM_BASE_URL,
    UPSTREAM_TIMEOUT_SECS,
)
from .token_logger import AsyncJSONLLogger, TokenEventLogger


app = FastAPI()


@app.on_event("startup")
async def _on_startup() -> None:
    jsonl_logger = AsyncJSONLLogger(
        file_path=TOKEN_LOG_PATH,
        rotate_max_bytes=LOG_ROTATE_MAX_BYTES,
        rotate_keep=LOG_ROTATE_KEEP,
    )
    await jsonl_logger.start()
    app.state.jsonl_logger = jsonl_logger
    app.state.token_logger = TokenEventLogger(jsonl_logger)


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    logger: AsyncJSONLLogger = app.state.jsonl_logger
    await logger.stop()


def _join_upstream_url(path: str) -> str:
    base = UPSTREAM_BASE_URL.rstrip("/")
    if path.startswith("/"):
        return base + path
    return base + "/" + path


def _extract_token_text(chunk: Dict[str, Any]) -> str:
    try:
        choices = chunk.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        if "delta" in choice:
            delta = choice.get("delta", {})
            return delta.get("content", "") or ""
        if "text" in choice:
            return choice.get("text", "") or ""
    except Exception:
        return ""
    return ""


def _extract_finish_reason(chunk: Dict[str, Any]) -> Optional[str]:
    try:
        choices = chunk.get("choices", [])
        if not choices:
            return None
        return choices[0].get("finish_reason")
    except Exception:
        return None


async def _stream_and_log(
    client: httpx.AsyncClient,
    upstream_url: str,
    payload: Dict[str, Any],
    req_headers: Dict[str, str],
    token_logger: TokenEventLogger,
) -> AsyncGenerator[bytes, None]:
    request_start = time.perf_counter()
    token_idx = 0
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    request_id_fallback = f"req_{uuid.uuid4()}"
    batch_id_from_headers: Optional[str] = None

    async with client.stream("POST", upstream_url, json=payload, headers=req_headers, timeout=UPSTREAM_TIMEOUT_SECS) as resp:
        # Try to capture potential batch id from headers if present
        for key, value in resp.headers.items():
            lk = key.lower()
            if lk in ("x-vllm-batch-id", "x-batch-id", "x-scheduler-batch-id"):
                batch_id_from_headers = value
                break

        async for raw_line in resp.aiter_lines():
            # Forward exactly as received
            if raw_line is None:
                continue
            line = raw_line.rstrip("\r")
            yield (line + "\n").encode("utf-8")

            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            request_id = (
                chunk.get("id")
                or request_id_fallback
            )

            # Also check for batch id in chunk
            batch_id = (
                chunk.get("batch_id")
                or chunk.get("vllm_batch_id")
                or batch_id_from_headers
            )

            token_text = _extract_token_text(chunk)
            now = time.perf_counter()

            if token_text:
                if first_token_time is None:
                    first_token_time = now
                    prefill_ms = (first_token_time - request_start) * 1000.0
                    await token_logger.log_prefill(
                        request_id=request_id,
                        batch_id=batch_id,
                        prefill_ms=prefill_ms,
                    )
                    last_token_time = first_token_time
                # Compute token generation time as delta from previous token (or first token)
                token_idx += 1
                gen_ms = (now - (last_token_time or now)) * 1000.0
                await token_logger.log_token(
                    request_id=request_id,
                    batch_id=batch_id,
                    token_idx=token_idx,
                    gen_ms=gen_ms,
                )
                last_token_time = now

            # Stop when finish_reason present
            finish_reason = _extract_finish_reason(chunk)
            if finish_reason is not None:
                # nothing to do; client will see the final chunk
                pass


async def _proxy_json(
    client: httpx.AsyncClient,
    upstream_url: str,
    payload: Dict[str, Any],
    req_headers: Dict[str, str],
) -> Response:
    resp = await client.post(upstream_url, json=payload, headers=req_headers, timeout=UPSTREAM_TIMEOUT_SECS)
    return JSONResponse(status_code=resp.status_code, content=resp.json())


def _collect_forward_headers(request: Request, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for k, v in request.headers.items():
        # Forward most headers including Authorization and OpenAI-* ones
        headers[k] = v
    if extra:
        headers.update(extra)
    return headers


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    stream = bool(payload.get("stream", False))
    upstream_url = _join_upstream_url(request.url.path)
    token_logger: TokenEventLogger = app.state.token_logger

    async with httpx.AsyncClient(http2=True) as client:
        if stream:
            generator = _stream_and_log(
                client=client,
                upstream_url=upstream_url,
                payload=payload,
                req_headers=_collect_forward_headers(request),
                token_logger=token_logger,
            )
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            return await _proxy_json(
                client=client,
                upstream_url=upstream_url,
                payload=payload,
                req_headers=_collect_forward_headers(request),
            )


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    payload = await request.json()
    stream = bool(payload.get("stream", False))
    upstream_url = _join_upstream_url(request.url.path)
    token_logger: TokenEventLogger = app.state.token_logger

    async with httpx.AsyncClient(http2=True) as client:
        if stream:
            generator = _stream_and_log(
                client=client,
                upstream_url=upstream_url,
                payload=payload,
                req_headers=_collect_forward_headers(request),
                token_logger=token_logger,
            )
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            return await _proxy_json(
                client=client,
                upstream_url=upstream_url,
                payload=payload,
                req_headers=_collect_forward_headers(request),
            )


# Generic passthrough for other endpoints if needed (no logging)
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def passthrough(request: Request, path: str) -> Response:
    upstream_url = _join_upstream_url("/v1/" + path)
    method = request.method.upper()

    async with httpx.AsyncClient(http2=True) as client:
        if method == "GET":
            resp = await client.get(upstream_url, params=dict(request.query_params), headers=_collect_forward_headers(request), timeout=UPSTREAM_TIMEOUT_SECS)
        elif method == "POST":
            # Try JSON, fallback to body
            try:
                payload = await request.json()
                resp = await client.post(upstream_url, json=payload, headers=_collect_forward_headers(request), timeout=UPSTREAM_TIMEOUT_SECS)
            except Exception:
                body = await request.body()
                resp = await client.post(upstream_url, content=body, headers=_collect_forward_headers(request), timeout=UPSTREAM_TIMEOUT_SECS)
        elif method == "PUT":
            body = await request.body()
            resp = await client.put(upstream_url, content=body, headers=_collect_forward_headers(request), timeout=UPSTREAM_TIMEOUT_SECS)
        elif method == "PATCH":
            body = await request.body()
            resp = await client.patch(upstream_url, content=body, headers=_collect_forward_headers(request), timeout=UPSTREAM_TIMEOUT_SECS)
        elif method == "DELETE":
            resp = await client.delete(upstream_url, headers=_collect_forward_headers(request), timeout=UPSTREAM_TIMEOUT_SECS)
        else:
            return JSONResponse(status_code=405, content={"error": "method not allowed"})

    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))