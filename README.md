# vLLM Token Timing Proxy

A lightweight FastAPI reverse proxy that forwards OpenAI-compatible vLLM requests and logs per-token timing records to JSONL.

## Features

- Transparent proxy for `/v1/chat/completions` and `/v1/completions`
- Streaming (`stream=true`) SSE parsing with per-token logging
- Records include:
  - `is_prefill`: whether this record is the prefill (True for a synthetic pre-first-token record, False for generated tokens)
  - `batch_id`: batch identifier if exposed by upstream headers or chunk payload; otherwise `null`
  - `request_id`: upstream response `id` (or a generated fallback)
  - `token_idx`: 0-based index of emitted token chunk within the request; `-1` for the prefill record
  - `unit`: always `"ms"`
  - `generation_token`: elapsed milliseconds
    - Prefill record: time from request send to first token chunk
    - Token record: time since previous token chunk
- JSONL with simple size-based rotation

## Requirements

- Python 3.10+

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Set the upstream vLLM OpenAI-compatible base URL (default `http://127.0.0.1:8000/v1`) and start the proxy:

```bash
export UPSTREAM_BASE_URL="http://127.0.0.1:8000/v1"
export TOKEN_LOG_PATH="/workspace/token_logs.jsonl"  # optional
# If you're in the project root:
uvicorn app.main:app --host 0.0.0.0 --port 9000
# Or from any directory (add project to import path):
uvicorn app.main:app --host 0.0.0.0 --port 9000 --app-dir /workspace
```

Point your client to `http://localhost:9000/v1/...` instead of the vLLM server.

## Notes

- Logging is most accurate with `stream=true` requests. For non-stream requests, the proxy forwards responses without per-token logging.
- `batch_id` is logged only if upstream provides a header like `X-VLLM-Batch-ID` or includes it in the chunk JSON; otherwise it is `null`.
- The prefill record is synthetic and marks the time until the first output token.

## Capture true vLLM batch ids (optional)

To enrich `batch_id` even if vLLM doesn't expose it in responses, preload a small monkey-patch in the vLLM process:

```bash
# 1) Make the hook visible to vLLM Python
export PYTHONPATH="/workspace/vllm_batch_hook:$PYTHONPATH"
# 2) Choose a directory for hook logs (default /tmp/vllm_hook)
export VLLM_HOOK_LOG_DIR="/tmp/vllm_hook"
# 3) Start vLLM as usual
python -m vllm.entrypoints.openai.api_server --model /path/to/model --port 8000
```

What it does:
- Logs engine_request_id <-> openai_request_id mapping to `${VLLM_HOOK_LOG_DIR}/request_map.jsonl`
- Logs scheduler batch formation events with a generated `batch_id` and the list of engine request ids to `${VLLM_HOOK_LOG_DIR}/batch_events.jsonl`

The proxy will read those files and map each streaming `request_id` to a `batch_id` as soon as they are available, so `batch_id` in your JSONL logs will be populated even if the upstream response omits it.