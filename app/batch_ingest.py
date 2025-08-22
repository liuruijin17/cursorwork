import asyncio
import json
import os
from typing import Dict, Optional


class BatchIdMapper:
    def __init__(self, log_dir: str) -> None:
        self._log_dir = log_dir
        self._engine_to_openai: Dict[str, str] = {}
        self._openai_to_batch: Dict[str, str] = {}
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None
        self._map_file = os.path.join(self._log_dir, "request_map.jsonl")
        self._batch_file = os.path.join(self._log_dir, "batch_events.jsonl")
        self._map_pos = 0
        self._batch_pos = 0

    async def start(self) -> None:
        os.makedirs(self._log_dir, exist_ok=True)
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task

    def get_batch_id(self, openai_request_id: str) -> Optional[str]:
        return self._openai_to_batch.get(openai_request_id)

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            await self._ingest_map()
            await self._ingest_batch()
            await asyncio.sleep(0.2)

    async def _ingest_map(self) -> None:
        try:
            with open(self._map_file, "r", encoding="utf-8") as f:
                f.seek(self._map_pos)
                for line in f:
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("type") != "request_map":
                        continue
                    engine_id = evt.get("engine_request_id")
                    openai_id = evt.get("openai_request_id")
                    if engine_id and openai_id:
                        self._engine_to_openai[engine_id] = openai_id
                self._map_pos = f.tell()
        except FileNotFoundError:
            return

    async def _ingest_batch(self) -> None:
        try:
            with open(self._batch_file, "r", encoding="utf-8") as f:
                f.seek(self._batch_pos)
                for line in f:
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("type") != "batch_formed":
                        continue
                    batch_id = evt.get("batch_id")
                    engine_ids = evt.get("engine_request_ids") or []
                    if not batch_id:
                        continue
                    for eid in engine_ids:
                        openai_id = self._engine_to_openai.get(eid) or eid
                        if openai_id:
                            self._openai_to_batch[openai_id] = batch_id
                self._batch_pos = f.tell()
        except FileNotFoundError:
            return