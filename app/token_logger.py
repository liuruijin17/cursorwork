import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import aiofiles


class AsyncJSONLLogger:
    def __init__(self, file_path: str, rotate_max_bytes: int, rotate_keep: int) -> None:
        self._file_path = file_path
        self._rotate_max_bytes = rotate_max_bytes
        self._rotate_keep = rotate_keep
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task[None]] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def stop(self) -> None:
        self._shutdown_event.set()
        if self._writer_task is not None:
            await self._writer_task

    async def log_dict(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        await self._queue.put(line)

    async def _writer_loop(self) -> None:
        while not self._shutdown_event.is_set() or not self._queue.empty():
            try:
                line = await asyncio.wait_for(self._queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            try:
                await self._maybe_rotate()
                async with aiofiles.open(self._file_path, "a", encoding="utf-8") as f:
                    await f.write(line)
            finally:
                self._queue.task_done()

    async def _maybe_rotate(self) -> None:
        try:
            current_size = os.path.getsize(self._file_path)
        except FileNotFoundError:
            return
        if current_size < self._rotate_max_bytes:
            return
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        rotated_path = f"{self._file_path}.{timestamp}"
        try:
            os.replace(self._file_path, rotated_path)
        except FileNotFoundError:
            return
        # Cleanup old files, keep most recent N
        prefix = os.path.basename(self._file_path)
        parent = os.path.dirname(self._file_path)
        rotated = sorted(
            [
                os.path.join(parent, name)
                for name in os.listdir(parent)
                if name.startswith(prefix + ".")
            ],
            reverse=True,
        )
        for idx, path in enumerate(rotated):
            if idx >= self._rotate_keep:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass


class TokenEventLogger:
    def __init__(self, jsonl_logger: AsyncJSONLLogger) -> None:
        self._jsonl_logger = jsonl_logger

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"

    async def log_prefill(self, *, request_id: str, batch_id: Optional[str], prefill_ms: float) -> None:
        record = {
            "ts": self._now_iso(),
            "is_prefill": True,
            "batch_id": batch_id,
            "request_id": request_id,
            "token_idx": 0,
            "unit": "ms",
            "generation_token": float(f"{prefill_ms:.3f}"),
        }
        await self._jsonl_logger.log_dict(record)

    async def log_token(self, *, request_id: str, batch_id: Optional[str], token_idx: int, gen_ms: float) -> None:
        record = {
            "ts": self._now_iso(),
            "is_prefill": False,
            "batch_id": batch_id,
            "request_id": request_id,
            "token_idx": token_idx,
            "unit": "ms",
            "generation_token": float(f"{gen_ms:.3f}"),
        }
        await self._jsonl_logger.log_dict(record)