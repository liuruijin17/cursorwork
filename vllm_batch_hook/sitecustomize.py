# This module is imported automatically if present in PYTHONPATH as sitecustomize
# It attempts best-effort monkey patches to vLLM to log:
# 1) Mapping from internal engine request id -> outward OpenAI request id
# 2) Batch formation events with a generated batch_id and the list of engine request ids

import os
import json
import time
import uuid
import importlib
from typing import Any, Dict, List

LOG_DIR = os.environ.get("VLLM_HOOK_LOG_DIR", "/tmp/vllm_hook")
REQ_MAP_FILE = os.path.join(LOG_DIR, "request_map.jsonl")
BATCH_FILE = os.path.join(LOG_DIR, "batch_events.jsonl")
DEBUG_FILE = os.path.join(LOG_DIR, "hook_debug.log")

os.makedirs(LOG_DIR, exist_ok=True)


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _debug(msg: str) -> None:
    try:
        with open(DEBUG_FILE, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except Exception:
        pass


# Patch 1: capture mapping at OpenAI entrypoint layer
patched_openai_entry = False
for mod_path, cls_name, method_names in [
    ("vllm.entrypoints.openai.server", "OpenAIServing", ["_create_request", "_create_chat_completion_request"]),
    ("vllm.entrypoints.openai.api_server", "OpenAIServing", ["_create_request", "_create_chat_completion_request"]),
    ("vllm.entrypoints.openai.serving", "OpenAIServing", ["_create_request", "_create_chat_completion_request"]),
    ("vllm.entrypoints.openai.serving_chat", "OpenAIServingChat", ["_create_request", "_create_chat_completion_request"]),
]:
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        target_attr = None
        for name in method_names:
            if hasattr(cls, name):
                target_attr = name
                break
        if target_attr is None:
            continue
        orig = getattr(cls, target_attr)

        def _wrapped_create_request(self, *args, **kwargs):
            req = orig(self, *args, **kwargs)
            try:
                engine_id = getattr(req, "request_id", None) or getattr(req, "id", None)
                openai_id = getattr(req, "openai_request_id", None) or getattr(req, "request_id_str", None)
                if not openai_id:
                    # Some server paths store outward id on request "id" or build later; synthesize if absent
                    openai_id = getattr(req, "id", None) or f"chatcmpl-{uuid.uuid4().hex}"
                    setattr(req, "openai_request_id", openai_id)
                if engine_id and openai_id:
                    _append_jsonl(REQ_MAP_FILE, {
                        "type": "request_map",
                        "engine_request_id": str(engine_id),
                        "openai_request_id": str(openai_id),
                        "ts": time.time(),
                    })
            except Exception as e:
                _debug(f"openai map error: {e}")
            return req

        setattr(cls, target_attr, _wrapped_create_request)
        patched_openai_entry = True
        _debug(f"patched openai entry: {mod_path}.{cls_name}.{target_attr}")
        break
    except Exception as e:
        _debug(f"probe failed: {mod_path}.{cls_name}: {e}")

# Fallback: engine-level hook captures engine request ids and attempts to infer outward id from args
if not patched_openai_entry:
    try:
        mod = importlib.import_module("vllm.engine.async_llm_engine")
        eng_cls = getattr(mod, "AsyncLLMEngine", None)
        if eng_cls is not None and hasattr(eng_cls, "add_request"):
            orig_add = getattr(eng_cls, "add_request")

            def _wrapped_add_request(self, *args, **kwargs):
                # Try to extract engine request id and an outward id if provided by caller
                engine_id = None
                outward_id = None
                try:
                    if "request_id" in kwargs:
                        engine_id = kwargs.get("request_id")
                    elif len(args) >= 1:
                        engine_id = args[0]
                    outward_id = kwargs.get("openai_request_id") or kwargs.get("request_id_str")
                except Exception:
                    pass
                if engine_id:
                    _append_jsonl(REQ_MAP_FILE, {
                        "type": "request_map",
                        "engine_request_id": str(engine_id),
                        "openai_request_id": str(outward_id or engine_id),
                        "ts": time.time(),
                    })
                return orig_add(self, *args, **kwargs)

            setattr(eng_cls, "add_request", _wrapped_add_request)
            _debug("patched AsyncLLMEngine.add_request")
    except Exception as e:
        _debug(f"engine-level hook failed: {e}")


# Patch 2: capture batch formation at scheduler layer
try:
    from vllm.engine.scheduler import Scheduler

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
                engine_ids: List[str] = []
                batch_id = f"vllm_batch_{uuid.uuid4().hex}"
                if isinstance(result, (list, tuple)):
                    for item in result:
                        eid = getattr(item, "request_id", None) or getattr(item, "id", None)
                        if eid:
                            engine_ids.append(str(eid))
                else:
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
                    _debug(f"batch formed {batch_id} with {len(engine_ids)} reqs")
            except Exception as e:
                _debug(f"scheduler hook error: {e}")
            return result

        setattr(Scheduler, schedule_attr, _wrapped_schedule)
        _debug(f"patched Scheduler.{schedule_attr}")
except Exception as e:
    _debug(f"scheduler hook failed: {e}")