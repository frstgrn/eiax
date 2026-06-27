"""Run async coroutines from sync call sites (including Jupyter notebooks)."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor


def run_sync[T](coro: Coroutine[object, object, T]) -> T:
    """ponytail: thread + fresh loop when a loop is already running (Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
