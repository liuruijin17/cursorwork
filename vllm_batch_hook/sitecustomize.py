# This module is imported automatically if present in PYTHONPATH as sitecustomize
# It attempts best-effort monkey patches to vLLM to log:
# 1) Mapping from internal engine request id -> outward OpenAI request id
# 2) Batch formation events with a generated batch_id and the list of engine request ids

import os
import json
import time
import uuid
from typing import Any, Dict, List

LOG_DIR = os.environ.get("VLLM_HOOK_LOG_DIR", "/tmp/vllm_hook")
REQ_MAP_FILE = os.path.join(LOG_DIR, "request_map.jsonl")
BATCH_FILE = os.path.join(LOG_DIR, "batch_events.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# Patch 1: capture mapping at OpenAI entrypoint layer
try:
    # vLLM 0.5+: openai protocol server
    from vllm.entrypoints.openai.server import OpenAIServing

    orig_create_req = getattr(OpenAIServing, "_create_request", None)
    if orig_create_req is not None:
        def _wrapped_create_request(self, *args, **kwargs):
            req = orig_create_req(self, *args, **kwargs)
            try:
                engine_id = req.request_id  # internal engine request id
                openai_id = getattr(req, "openai_request_id", None) or getattr(req, "request_id_str", None)
                # Fallback: synthesize an outward id if not present yet
                if not openai_id:
                    openai_id = f"chatcmpl-{uuid.uuid4().hex}"
                    setattr(req, "openai_request_id", openai_id)
                _append_jsonl(REQ_MAP_FILE, {
                    "type": "request_map",
                    "engine_request_id": engine_id,
                    "openai_request_id": openai_id,
                    "ts": time.time(),
                })
            except Exception:
                pass
            return req
        OpenAIServing._create_request = _wrapped_create_request  # type: ignore[attr-defined]
except Exception:
    pass


# Patch 2: capture batch formation at scheduler layer
try:
    from vllm.engine.scheduler import Scheduler

    # Newer versions may use method name 'schedule'; older may use '_schedule'
    schedule_attr = None
    if hasattr(Scheduler, "schedule"):
        schedule_attr = "schedule"
    elif hasattr(Scheduler, "_schedule"):
        schedule_attr = "_schedule"

    if schedule_attr is not None:
        orig_schedule = getattr(Scheduler, schedule_attr)
        def _wrapped_schedule(self, *args, **kwargs):
            result = orig_schedule(self, *args, **kwargs)
            try:
                # result may contain a list of scheduled requests or a structure including it
                engine_ids: List[str] = []
                batch_id = f"vllm_batch_{uuid.uuid4().hex}"
                if isinstance(result, (list, tuple)):
                    for item in result:
                        eid = getattr(item, "request_id", None) or getattr(item, "id", None)
                        if eid:
                            engine_ids.append(str(eid))
                else:
                    # Try common container shapes
                    maybe_reqs = getattr(result, "scheduled_requests", None) or getattr(result, "requests", None)
                    if isinstance(maybe_reqs, (list, tuple)):
                        for item in maybe_reqs:
                            eid = getattr(item, "request_id", None) or getattr(item, "id", None)
                            if eid:
                                engine_ids.append(str(eid))
                if engine_ids:
                    _append_jsonl(BATCH_FILE, {
                        "type": "batch_formed",
                        "batch_id": batch_id,
                        "engine_request_ids": engine_ids,
                        "ts": time.time(),
                    })
            except Exception:
                pass
            return result
        setattr(Scheduler, schedule_attr, _wrapped_schedule)
except Exception:
    pass