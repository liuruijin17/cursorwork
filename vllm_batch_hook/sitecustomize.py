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
            _debug(f"class not found: {mod_path}.{cls_name}")
            continue
        target_attr = None
        for name in method_names:
            if hasattr(cls, name):
                target_attr = name
                break
        if target_attr is None:
            _debug(f"no target method on {mod_path}.{cls_name} candidates={method_names}")
            continue
        orig = getattr(cls, target_attr)

        def _wrapped_create_request(self, *args, **kwargs):
            req = orig(self, *args, **kwargs)
            try:
                engine_id = getattr(req, "request_id", None) or getattr(req, "id", None)
                openai_id = getattr(req, "openai_request_id", None) or getattr(req, "request_id_str", None)
                if not openai_id:
                    openai_id = getattr(req, "id", None) or f"chatcmpl-{uuid.uuid4().hex}"
                    try:
                        setattr(req, "openai_request_id", openai_id)
                    except Exception:
                        pass
                if engine_id and openai_id:
                    _append_jsonl(REQ_MAP_FILE, {
                        "type": "request_map",
                        "engine_request_id": str(engine_id),
                        "openai_request_id": str(openai_id),
                        "ts": time.time(),
                    })
                    _debug(f"map wrote engine={engine_id} openai={openai_id}")
            except Exception as e:
                _debug(f"openai map error: {e}")
            return req

        setattr(cls, target_attr, _wrapped_create_request)
        patched_openai_entry = True
        _debug(f"patched openai entry: {mod_path}.{cls_name}.{target_attr}")
        break
    except Exception as e:
        _debug(f"probe failed: {mod_path}.{cls_name}: {e}")


# Patch 1b: engine-level add_request variants
def _try_patch_engine(path: str, cls_name: str, method_names: List[str]) -> bool:
    try:
        mod = importlib.import_module(path)
        cls = getattr(mod, cls_name, None)
        if cls is None:
            _debug(f"engine class not found: {path}.{cls_name}")
            return False
        target = None
        for m in method_names:
            if hasattr(cls, m):
                target = m
                break
        if target is None:
            _debug(f"no engine method on {path}.{cls_name} candidates={method_names}")
            return False
        orig = getattr(cls, target)

        def _wrapped_add_request(self, *args, **kwargs):
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
                _debug(f"engine map wrote engine={engine_id} openai={outward_id or engine_id}")
            return orig(self, *args, **kwargs)

        setattr(cls, target, _wrapped_add_request)
        _debug(f"patched {path}.{cls_name}.{target}")
        return True
    except Exception as e:
        _debug(f"engine patch failed: {path}.{cls_name}: {e}")
        return False


if not patched_openai_entry:
    patched_engine = False
    # Async engine
    patched_engine |= _try_patch_engine("vllm.engine.async_llm_engine", "AsyncLLMEngine", ["add_request", "_add_request"])  # type: ignore[assignment]
    # Sync engine
    patched_engine |= _try_patch_engine("vllm.engine.llm_engine", "LLMEngine", ["add_request", "_add_request"])  # type: ignore[assignment]
    if not patched_engine:
        _debug("no engine add_request patched")


# Patch 2: capture batch formation at scheduler layer (probe multiple paths)
for sched_mod in [
    "vllm.engine.scheduler",
    "vllm.core.scheduler",
    "vllm.core.scheduler_core",
]:
    try:
        mod = importlib.import_module(sched_mod)
        cls = getattr(mod, "Scheduler", None)
        if cls is None:
            _debug(f"Scheduler not found in {sched_mod}")
            continue
        schedule_attr = None
        for cand in ("schedule", "_schedule"):
            if hasattr(cls, cand):
                schedule_attr = cand
                break
        if schedule_attr is None:
            _debug(f"no schedule method on Scheduler in {sched_mod}")
            continue
        orig_schedule = getattr(cls, schedule_attr)

        def _wrapped_schedule(self, *args, **kwargs):
            result = orig_schedule(self, *args, **kwargs)
            try:
                engine_ids: List[str] = []
                batch_id = f"vllm_batch_{uuid.uuid4().hex}"
                # v0.9.2rc1 Scheduler.schedule returns (meta_list, outputs, allow_async)
                outputs = None
                if isinstance(result, tuple) and len(result) >= 2:
                    outputs = result[1]
                # Older/other variants might return a struct directly
                if outputs is None:
                    outputs = result
                # Prefer scheduled_seq_groups from outputs
                seq_groups_container = getattr(outputs, "scheduled_seq_groups", None)
                if seq_groups_container is not None:
                    for item in list(seq_groups_container):
                        try:
                            seq_group = getattr(item, "seq_group", None) or item
                            eid = getattr(seq_group, "request_id", None) or getattr(seq_group, "id", None)
                            if eid:
                                engine_ids.append(str(eid))
                        except Exception:
                            continue
                else:
                    # Fallback: previous heuristics
                    if isinstance(result, (list, tuple)):
                        for item in result:
                            eid = getattr(item, "request_id", None) or getattr(item, "id", None)
                            if eid:
                                engine_ids.append(str(eid))
                    else:
                        maybe_reqs = getattr(result, "scheduled_requests", None) or getattr(result, "requests", None) or getattr(result, "seq_groups", None)
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
                    _debug(f"batch formed {batch_id} with {len(engine_ids)} reqs via {sched_mod}.{schedule_attr}")
            except Exception as e:
                _debug(f"scheduler hook error: {e}")
            return result

        setattr(cls, schedule_attr, _wrapped_schedule)
        _debug(f"patched Scheduler in {sched_mod}.{schedule_attr}")
        break
    except Exception as e:
        _debug(f"scheduler hook failed in {sched_mod}: {e}")