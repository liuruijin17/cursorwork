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
                num_groups = None
                if seq_groups_container is not None:
                    try:
                        num_groups = len(seq_groups_container)  # may fail if not sized
                    except Exception:
                        try:
                            num_groups = sum(1 for _ in seq_groups_container)
                        except Exception:
                            num_groups = -1
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
                        num_groups = len(result)
                        for item in result:
                            eid = getattr(item, "request_id", None) or getattr(item, "id", None)
                            if eid:
                                engine_ids.append(str(eid))
                    else:
                        maybe_reqs = getattr(result, "scheduled_requests", None) or getattr(result, "requests", None) or getattr(result, "seq_groups", None)
                        try:
                            num_groups = len(maybe_reqs) if maybe_reqs is not None else 0
                        except Exception:
                            num_groups = -1
                        if isinstance(maybe_reqs, (list, tuple)):
                            for item in maybe_reqs:
                                eid = getattr(item, "request_id", None) or getattr(item, "id", None)
                                if eid:
                                    engine_ids.append(str(eid))
                # Always write a record to make presence observable
                _append_jsonl(BATCH_FILE, {
                    "type": "batch_formed",
                    "batch_id": batch_id,
                    "engine_request_ids": engine_ids,
                    "ts": time.time(),
                })
                _debug(f"batch observed groups={num_groups} engine_ids={len(engine_ids)} via {sched_mod}.{schedule_attr}")
            except Exception as e:
                _debug(f"scheduler hook error: {e}")
            return result

        setattr(cls, schedule_attr, _wrapped_schedule)
        _debug(f"patched Scheduler in {sched_mod}.{schedule_attr}")
        break
    except Exception as e:
        _debug(f"scheduler hook failed in {sched_mod}: {e}")

# Instance-level patch: wrap scheduler.schedule after engine constructs the scheduler
try:
    mod = importlib.import_module("vllm.engine.llm_engine")
    LLMEngine = getattr(mod, "LLMEngine", None)
    if LLMEngine is not None and hasattr(LLMEngine, "__init__"):
        _orig_llm_init = LLMEngine.__init__
        def _wrapped_llm_init(self, *args, **kwargs):
            _orig_llm_init(self, *args, **kwargs)
            try:
                sched = getattr(self, "scheduler", None)
                if sched is not None and hasattr(sched, "schedule"):
                    _orig_sched_schedule = sched.schedule
                    def _wrapped_schedule(*s_args, **s_kwargs):
                        result = _orig_sched_schedule(*s_args, **s_kwargs)
                        try:
                            engine_ids: List[str] = []
                            batch_id = f"vllm_batch_{uuid.uuid4().hex}"
                            outputs = None
                            if isinstance(result, tuple) and len(result) >= 2:
                                outputs = result[1]
                            if outputs is None:
                                outputs = result
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
                            _append_jsonl(BATCH_FILE, {
                                "type": "batch_formed",
                                "batch_id": batch_id,
                                "engine_request_ids": engine_ids,
                                "ts": time.time(),
                            })
                            _debug(f"instance hook batch engine_ids={len(engine_ids)}")
                        except Exception as e:
                            _debug(f"instance scheduler hook error: {e}")
                        return result
                    try:
                        setattr(sched, "schedule", _wrapped_schedule)
                        _debug("wrapped instance scheduler.schedule on LLMEngine")
                    except Exception as e:
                        _debug(f"failed wrapping instance scheduler on LLMEngine: {e}")
            except Exception as e:
                _debug(f"LLMEngine.__init__ post-wrap error: {e}")
        LLMEngine.__init__ = _wrapped_llm_init
        _debug("patched LLMEngine.__init__ to wrap scheduler")
except Exception as e:
    _debug(f"LLMEngine instance-level hook failed: {e}")

try:
    mod = importlib.import_module("vllm.engine.async_llm_engine")
    AsyncLLMEngine = getattr(mod, "AsyncLLMEngine", None)
    if AsyncLLMEngine is not None and hasattr(AsyncLLMEngine, "__init__"):
        _orig_async_init = AsyncLLMEngine.__init__
        def _wrapped_async_init(self, *args, **kwargs):
            _orig_async_init(self, *args, **kwargs)
            try:
                sched = getattr(self, "scheduler", None)
                if sched is not None and hasattr(sched, "schedule"):
                    _orig_sched_schedule = sched.schedule
                    def _wrapped_schedule(*s_args, **s_kwargs):
                        result = _orig_sched_schedule(*s_args, **s_kwargs)
                        try:
                            engine_ids: List[str] = []
                            batch_id = f"vllm_batch_{uuid.uuid4().hex}"
                            outputs = None
                            if isinstance(result, tuple) and len(result) >= 2:
                                outputs = result[1]
                            if outputs is None:
                                outputs = result
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
                            _append_jsonl(BATCH_FILE, {
                                "type": "batch_formed",
                                "batch_id": batch_id,
                                "engine_request_ids": engine_ids,
                                "ts": time.time(),
                            })
                            _debug(f"instance hook batch(engine-async) engine_ids={len(engine_ids)}")
                        except Exception as e:
                            _debug(f"instance async scheduler hook error: {e}")
                        return result
                    try:
                        setattr(sched, "schedule", _wrapped_schedule)
                        _debug("wrapped instance scheduler.schedule on AsyncLLMEngine")
                    except Exception as e:
                        _debug(f"failed wrapping instance scheduler on AsyncLLMEngine: {e}")
            except Exception as e:
                _debug(f"AsyncLLMEngine.__init__ post-wrap error: {e}")
        AsyncLLMEngine.__init__ = _wrapped_async_init
        _debug("patched AsyncLLMEngine.__init__ to wrap scheduler")
except Exception as e:
    _debug(f"AsyncLLMEngine instance-level hook failed: {e}")

# Intercept future engine.scheduler assignments to ensure schedule is wrapped
try:
    mod = importlib.import_module("vllm.engine.llm_engine")
    LLMEngine = getattr(mod, "LLMEngine", None)
    if LLMEngine is not None and hasattr(LLMEngine, "__setattr__"):
        _orig_llm_setattr = LLMEngine.__setattr__
        def _wrapped_llm_setattr(self, name, value):
            _orig_llm_setattr(self, name, value)
            try:
                if name == "scheduler" and value is not None:
                    sched = getattr(self, "scheduler", None)
                    if sched is not None and hasattr(sched, "schedule"):
                        try:
                            orig = sched.schedule
                            def _wrapped_schedule(*s_args, **s_kwargs):
                                result = orig(*s_args, **s_kwargs)
                                try:
                                    engine_ids: List[str] = []
                                    batch_id = f"vllm_batch_{uuid.uuid4().hex}"
                                    outputs = None
                                    if isinstance(result, tuple) and len(result) >= 2:
                                        outputs = result[1]
                                    if outputs is None:
                                        outputs = result
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
                                    _append_jsonl(BATCH_FILE, {
                                        "type": "batch_formed",
                                        "batch_id": batch_id,
                                        "engine_request_ids": engine_ids,
                                        "ts": time.time(),
                                    })
                                    _debug(f"__setattr__ hook batch engine_ids={len(engine_ids)}")
                                except Exception as e:
                                    _debug(f"__setattr__ scheduler hook error: {e}")
                                return result
                            setattr(sched, "schedule", _wrapped_schedule)
                            _debug("wrapped scheduler.schedule via LLMEngine.__setattr__")
                        except Exception as e:
                            _debug(f"failed to wrap scheduler via LLMEngine.__setattr__: {e}")
            except Exception as e:
                _debug(f"LLMEngine.__setattr__ post-wrap error: {e}")
        LLMEngine.__setattr__ = _wrapped_llm_setattr
        _debug("patched LLMEngine.__setattr__ to wrap scheduler on assignment")
except Exception as e:
    _debug(f"LLMEngine __setattr__ hook failed: {e}")

try:
    mod = importlib.import_module("vllm.engine.async_llm_engine")
    AsyncLLMEngine = getattr(mod, "AsyncLLMEngine", None)
    if AsyncLLMEngine is not None and hasattr(AsyncLLMEngine, "__setattr__"):
        _orig_async_setattr = AsyncLLMEngine.__setattr__
        def _wrapped_async_setattr(self, name, value):
            _orig_async_setattr(self, name, value)
            try:
                if name == "scheduler" and value is not None:
                    sched = getattr(self, "scheduler", None)
                    if sched is not None and hasattr(sched, "schedule"):
                        try:
                            orig = sched.schedule
                            def _wrapped_schedule(*s_args, **s_kwargs):
                                result = orig(*s_args, **s_kwargs)
                                try:
                                    engine_ids: List[str] = []
                                    batch_id = f"vllm_batch_{uuid.uuid4().hex}"
                                    outputs = None
                                    if isinstance(result, tuple) and len(result) >= 2:
                                        outputs = result[1]
                                    if outputs is None:
                                        outputs = result
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
                                    _append_jsonl(BATCH_FILE, {
                                        "type": "batch_formed",
                                        "batch_id": batch_id,
                                        "engine_request_ids": engine_ids,
                                        "ts": time.time(),
                                    })
                                    _debug(f"__setattr__ hook batch(engine-async) engine_ids={len(engine_ids)}")
                                except Exception as e:
                                    _debug(f"__setattr__ async scheduler hook error: {e}")
                                return result
                            setattr(sched, "schedule", _wrapped_schedule)
                            _debug("wrapped scheduler.schedule via AsyncLLMEngine.__setattr__")
                        except Exception as e:
                            _debug(f"failed to wrap scheduler via AsyncLLMEngine.__setattr__: {e}")
            except Exception as e:
                _debug(f"AsyncLLMEngine.__setattr__ post-wrap error: {e}")
        AsyncLLMEngine.__setattr__ = _wrapped_async_setattr
        _debug("patched AsyncLLMEngine.__setattr__ to wrap scheduler on assignment")
except Exception as e:
    _debug(f"AsyncLLMEngine __setattr__ hook failed: {e}")

# Class-level wrap of internal scheduling paths to be robust to overrides
try:
    mod = importlib.import_module("vllm.core.scheduler")
    SchedulerCls = getattr(mod, "Scheduler", None)
    def _emit_from_outputs(outputs, tag: str):
        try:
            engine_ids: List[str] = []
            batch_id = f"vllm_batch_{uuid.uuid4().hex}"
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
            _append_jsonl(BATCH_FILE, {
                "type": "batch_formed",
                "batch_id": batch_id,
                "engine_request_ids": engine_ids,
                "ts": time.time(),
            })
            _debug(f"{tag} emit engine_ids={len(engine_ids)}")
        except Exception as e:
            _debug(f"{tag} emit error: {e}")

    if SchedulerCls is not None:
        for meth in ["_schedule", "_schedule_default", "_schedule_chunked_prefill"]:
            if hasattr(SchedulerCls, meth):
                orig = getattr(SchedulerCls, meth)
                def _make_wrap(o, name):
                    def _wrapped(self, *args, **kwargs):
                        out = o(self, *args, **kwargs)
                        # _schedule returns outputs; default/chunked also return outputs
                        try:
                            _emit_from_outputs(out, f"{name}")
                        except Exception as e:
                            _debug(f"wrapper {name} error: {e}")
                        return out
                    return _wrapped
                try:
                    setattr(SchedulerCls, meth, _make_wrap(orig, meth))
                    _debug(f"wrapped Scheduler.{meth}")
                except Exception as e:
                    _debug(f"failed to wrap Scheduler.{meth}: {e}")
except Exception as e:
    _debug(f"class-level _schedule wrapping failed: {e}")