from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
from datetime import UTC, datetime

from .config import get_settings
from .pipeline import Pipeline
from .source import BudgetExhausted, CircuitOpen, Sentinel2StacClient
from .storage import Store


async def loop(once: bool = False) -> None:
    settings = get_settings()
    store = Store(settings)
    source = Sentinel2StacClient(settings, store)
    pipeline = Pipeline(settings, store, source)
    while True:
        store.prune_request_log(datetime.now(UTC))
        try:
            frame = await pipeline.ingest_once()
            logging.info(
                "Sentinel-2 ingestion completed",
                extra={"frame_id": None if frame is None else frame.id},
            )
        except (BudgetExhausted, CircuitOpen) as error:
            logging.warning("ingestion safely deferred: %s", error)
        except Exception:
            logging.exception("ingestion failed")
        if once:
            return
        jitter = secrets.SystemRandom().uniform(0.95, 1.05)
        await asyncio.sleep(settings.poll_interval_seconds * jitter)


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(loop(args.once))


if __name__ == "__main__":
    run()
