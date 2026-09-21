#!/usr/bin/env python3
"""
Snowball sampling di canali Telegram basato su messaggi inoltrati.

Flusso:
1. Legge i canali seed dal file config/config.json.
2. Per ogni canale legge al massimo X messaggi recenti (X da config JSON).
3. Cerca i messaggi inoltrati da altri canali.
4. Salva:
   - CSV canali (seed + canali scoperti)
   - CSV relazioni (channel_to, channel_from, metadati messaggio)

Esempio:
    python src/snowball_sampling.py --config config/config.json
"""

from __future__ import annotations

import asyncio
import csv
import logging
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import qrcode
import pandas as pd


try:
    from .config import DEFAULT_CONFIG_FILE, TelegramConfig, load_project_config, DATA_DIR
except ImportError:
    from config import DEFAULT_CONFIG_FILE, TelegramConfig, load_project_config, DATA_DIR

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import Channel, PeerChannel

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    messages_per_channel: int
    max_depth: int
    save_interval_seconds: int
    log_level: str


@dataclass
class RunConfig:
    telegram: TelegramConfig
    sampling: SamplingConfig
    seed_channels: List[Tuple[str, int]]  # List of (channel_ref, discovered_depth)
    previous_depth: int
    channels_output: str
    relations_output: str


@dataclass
class QueueItem:
    channel_ref: str
    depth: int


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Esegue snowball sampling tra canali Telegram"
#     )
#     parser.add_argument(
#         "--config",
#         default=str(DEFAULT_CONFIG_FILE),
#         help="File JSON unico di configurazione",
#     )
#     parser.add_argument(
#         "--channels-output",
#         default=None,
#         help="Percorso base CSV canali (verra aggiunto un suffisso timestamp)",
#     )
#     parser.add_argument(
#         "--relations-output",
#         default=None,
#         help="Percorso base CSV relazioni (verra aggiunto un suffisso timestamp)",
#     )
#     return parser.parse_args()


def normalize_channel_ref(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""

    if value.startswith("https://t.me/"):
        value = value.split("https://t.me/", 1)[1]

    if value.startswith("@"):
        value = value[1:]

    return value.strip()


def with_run_timestamp(output_file: str, run_stamp: str) -> str:
    output_path = Path(output_file)
    suffix = output_path.suffix or ".csv"
    filename = f"{output_path.stem}_{run_stamp}{suffix}"
    return str(output_path.with_name(filename))


def find_latest_csv_for_base(base_output_file: str) -> Optional[Path]:
    base_path = Path(base_output_file)
    if not base_path.parent.exists():
        return None

    pattern = f"{base_path.stem}*{base_path.suffix}"
    candidates = [item for item in base_path.parent.glob(pattern) if item.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda item: item.stat().st_mtime)


def load_seed_channels(channels_df: pd.DataFrame) -> Tuple[List[Tuple[str, int]], int]:
    df_channels = channels_df.loc[:, ["username", "discovered_depth"]]
    logger.debug("Loading seed channels from DataFrame with %d entries", len(df_channels))
    logger.debug("Removing duplicate seed channels and nan values")
    df_channels = df_channels.drop_duplicates(subset=["username"])
    df_channels = df_channels.dropna(subset=["username"])
    seed_channels = df_channels.to_records(index=False)
    previous_depth = df_channels["discovered_depth"].max()
    return seed_channels.tolist(), previous_depth


def load_run_config(
    config_file: Optional[str] = None,
    channels_output_override: Optional[str] = None,
    relations_output_override: Optional[str] = None,
) -> RunConfig:

    if config_file is None:
        config_file = str(DEFAULT_CONFIG_FILE)
    project_cfg = load_project_config(config_file)
    telegram_cfg = project_cfg.telegram
    snowball_cfg = project_cfg.snowball

    restart = snowball_cfg.restart
    channels_output = snowball_cfg.channels_output_csv
    relations_output = snowball_cfg.relations_output_csv

    if channels_output_override:
        channels_output = str(channels_output_override).strip()
    if relations_output_override:
        relations_output = str(relations_output_override).strip()

    previous_depth = 0
    seed_channels: List[Tuple[str, int]] = []

    if restart:
        # if restart use seed channels defined in the config, if not take seed channels from snowball_channels.csv
        logger.info("Snowball sampling will restart from the beginning.")
        channels = [
            normalize_channel_ref(item)
            for item in snowball_cfg.seed_channels
            if item.strip()
        ]
        channels = list(dict.fromkeys(channels))
        previous_depth = 0
        seed_channels = [(ch, 0) for ch in channels]
        if not seed_channels:
            raise ValueError("snowball.seed_channels non puo essere vuoto")
    else:
        print("Snowball sampling will continue from the existing channels.")
        # Load all previously discovered seed channels from all the previous runs
        # Read all the files snowball_channels_* and merge them in a DataFrame (or similar structure)

        if not any(Path(DATA_DIR).glob("snowball_channels_*.csv")):
            print("Nessun file snowball_channels_*.csv trovato. Nessun seed channel caricato.")
        else:
            df_channels = pd.concat(
                [pd.read_csv(f) for f in Path(DATA_DIR).glob("snowball_channels_*.csv")],
                ignore_index=True,
            )
            seed_channels, previous_depth = load_seed_channels(df_channels)

    return RunConfig(
        telegram=TelegramConfig(
            api_id=telegram_cfg.api_id,
            api_hash=telegram_cfg.api_hash,
            session_name=telegram_cfg.session_name,
        ),
        sampling=SamplingConfig(
            messages_per_channel=snowball_cfg.messages_per_channel,
            max_depth=snowball_cfg.max_depth,
            save_interval_seconds=snowball_cfg.save_interval_seconds,
            log_level=snowball_cfg.log_level,
        ),
        seed_channels=seed_channels,
        previous_depth=previous_depth,
        channels_output=channels_output,
        relations_output=relations_output,
    )


def safe_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def channel_key_from_id(channel_id: int) -> str:
    return f"channel:{channel_id}"


def serialize_date(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    return dt.astimezone(timezone.utc).isoformat()


def build_channel_record(entity: Channel, *, is_seed: bool, depth: int) -> Dict[str, Any]:
    return {
        "channel_id": int(entity.id),
        "username": entity.username or "",
        "title": entity.title or "",
        "is_seed": int(is_seed),
        "discovered_depth": int(depth),
        "participants_count": int(getattr(entity, "participants_count", 0) or 0),
        "broadcast": int(safe_bool(getattr(entity, "broadcast", False))),
        "megagroup": int(safe_bool(getattr(entity, "megagroup", False))),
        "verified": int(safe_bool(getattr(entity, "verified", False))),
        "scam": int(safe_bool(getattr(entity, "scam", False))),
        "fake": int(safe_bool(getattr(entity, "fake", False))),
        "date_collected_utc": serialize_date(datetime.now(timezone.utc)),
        "resolution_status": "ok",
    }


def build_unresolved_seed_record(channel_ref: str) -> Dict[str, Any]:
    return {
        "channel_id": "",
        "username": normalize_channel_ref(channel_ref),
        "title": "",
        "is_seed": 1,
        "discovered_depth": 0,
        "participants_count": 0,
        "broadcast": 0,
        "megagroup": 0,
        "verified": 0,
        "scam": 0,
        "fake": 0,
        "date_collected_utc": serialize_date(datetime.now(timezone.utc)),
        "resolution_status": "seed_unresolved",
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def configure_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class IncrementalCsvWriter:
    def __init__(
        self,
        output_file: str,
        fieldnames: List[str],
        flush_interval_seconds: int,
    ) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = output_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
        )
        self._writer.writeheader()

        self._pending_rows: List[Dict[str, Any]] = []
        self._flush_interval_seconds = flush_interval_seconds
        self._last_flush_ts = asyncio.get_running_loop().time()

    def add_row(self, row: Dict[str, Any]) -> None:
        self._pending_rows.append(row)

    def maybe_flush(self, *, force: bool = False) -> int:
        now = asyncio.get_running_loop().time()
        if not self._pending_rows:
            if force:
                self._file.flush()
            return 0

        elapsed = now - self._last_flush_ts
        if not force and elapsed < self._flush_interval_seconds:
            return 0

        self._writer.writerows(self._pending_rows)
        written_rows = len(self._pending_rows)
        self._pending_rows.clear()
        self._file.flush()
        self._last_flush_ts = now
        return written_rows

    def close(self) -> None:
        self.maybe_flush(force=True)
        with suppress(Exception):
            self._file.close()


async def resolve_channel_entity(
    client: TelegramClient,
    channel_ref: str,
) -> Optional[Channel]:
    
    try:
        logger.debug("Resolving channel entity for '%s'", channel_ref)
        entity = await client.get_entity(channel_ref)
        logger.debug("Resolved channel entity for '%s': %s", channel_ref, entity)
    except FloodWaitError as exc:
        logger.warning("Flood wait error while resolving channel '%s': %s", channel_ref, exc)
        if int(exc.seconds)>3600*3:
            logger.error("Flood wait too long for channel '%s': %s", channel_ref, exc)
            return None
        else:
            logger.info("Waiting for %s seconds due to flood wait for channel '%s'", int(exc.seconds), channel_ref)
            await asyncio.sleep(int(exc.seconds) + 1)
            entity = await client.get_entity(channel_ref)

    except Exception as exc:
        logger.warning("Impossibile risolvere il canale '%s': %s", channel_ref, exc)
        return None

    if isinstance(entity, Channel):
        return entity

    return None


async def resolve_source_channel(
    client: TelegramClient,
    from_peer: Optional[PeerChannel],
) -> Tuple[Optional[Channel], Optional[int]]:
    if not isinstance(from_peer, PeerChannel):
        return None, None

    source_id = int(from_peer.channel_id)

    try:
        source_entity = await client.get_entity(from_peer)
    except FloodWaitError as exc:
        if int(exc.seconds) > 3600 * 3:
            logger.error("Flood wait too long for source channel '%s': %s", source_id, exc)
            return None, source_id
        else:
            logger.warning("Flood wait error while resolving source channel '%s': %s", source_id, exc)
            await asyncio.sleep(int(exc.seconds) + 1)
            source_entity = await client.get_entity(from_peer)
    except Exception as exc:
        logger.warning("Impossibile risolvere il canale sorgente '%s': %s", source_id, exc)
        source_entity = None

    if isinstance(source_entity, Channel):
        return source_entity, source_id

    return None, source_id


async def run_snowball_sampling(
    client: TelegramClient,
    seed_channels: List[Tuple[str, int]],
    previous_depth: int,
    sampling_cfg: SamplingConfig,
    channels_output_file: str,
    relations_output_file: str,
) -> Tuple[int, int, int]:
    # Resume strategy: process channels at current depth, treat lower depths as already visited.
    queue: Deque[QueueItem] = deque(
        [QueueItem(ch, depth) for ch, depth in seed_channels if depth == previous_depth]
    )
    visited: Set[str] = {
        normalize_channel_ref(ch) for ch, depth in seed_channels if depth < previous_depth
    }

    discovered_channel_keys: Set[str] = set()
    relations_count = 0

    seed_set = {normalize_channel_ref(item[0]) for item in seed_channels if item[0].strip()}
    unresolved_seed_refs: Set[str] = set(seed_set)

    channels_writer = IncrementalCsvWriter(
        output_file=channels_output_file,
        fieldnames=[
            "channel_id",
            "username",
            "title",
            "is_seed",
            "discovered_depth",
            "participants_count",
            "broadcast",
            "megagroup",
            "verified",
            "scam",
            "fake",
            "date_collected_utc",
            "resolution_status",
        ],
        flush_interval_seconds=sampling_cfg.save_interval_seconds,
    )
    relations_writer = IncrementalCsvWriter(
        output_file=relations_output_file,
        fieldnames=[
            "channel_to_id",
            "channel_to_username",
            "channel_to_title",
            "channel_from_id",
            "channel_from_username",
            "channel_from_title",
            "from_name_fallback",
            "target_message_id",
            "target_message_date_utc",
            "sampling_depth",
        ],
        flush_interval_seconds=sampling_cfg.save_interval_seconds,
    )

    def maybe_flush_writers(force: bool = False) -> None:
        channels_written = channels_writer.maybe_flush(force=force)
        relations_written = relations_writer.maybe_flush(force=force)
        if channels_written > 0 or relations_written > 0:
            logger.info(
                "Checkpoint CSV salvato (%ss): +canali=%s, +relazioni=%s",
                sampling_cfg.save_interval_seconds,
                channels_written,
                relations_written,
            )

    try:
        while queue:
            current = queue.popleft()
            if current.depth > sampling_cfg.max_depth:
                continue
            logger.info(
                "Elaborazione canale=%s profondita=%s coda_rimanente=%s",
                current.channel_ref,
                current.depth,
                len(queue),
            )
            current_entity = await resolve_channel_entity(client, current.channel_ref)
            if current_entity is None:
                logger.warning("Canale non risolto: %s", current.channel_ref)
                continue

            logger.debug(
                "Risolto canale=%s",
                current.channel_ref
            )

            current_key = channel_key_from_id(int(current_entity.id))
            if current_key in visited:
                continue

            visited.add(current_key)

            is_seed = normalize_channel_ref(current.channel_ref) in seed_set
            if is_seed:
                unresolved_seed_refs.discard(normalize_channel_ref(current.channel_ref))

            if current_key not in discovered_channel_keys:
                channels_writer.add_row(
                    build_channel_record(
                        current_entity,
                        is_seed=is_seed,
                        depth=current.depth,
                    )
                )
                logger.debug(
                    "Aggiunto canale scoperto=%s a chiave=%s",
                    current.channel_ref,
                    current_key,
                )
                discovered_channel_keys.add(current_key)

            logger.debug(
                "Lettura messaggi canale=%s limit=%s",
                current.channel_ref,
                sampling_cfg.messages_per_channel,
            )
            try:
                iterator = client.iter_messages(
                    entity=current_entity,
                    limit=sampling_cfg.messages_per_channel,
                )
                i=1
                async for message in iterator:
                    logger.debug(
                        "Processing message id=%s from channel=%s (message count=%s)",
                        message.id,
                        current.channel_ref,
                        i,
                    )
                    i += 1
                    fwd_from = getattr(message, "fwd_from", None)
                    if not fwd_from:
                        continue

                    from_peer = getattr(fwd_from, "from_id", None)
                    source_entity, source_id = await resolve_source_channel(client, from_peer)

                    from_username = ""
                    from_title = ""
                    from_name_fallback = ""

                    if source_entity is not None:
                        source_key = channel_key_from_id(int(source_entity.id))
                        if source_key not in discovered_channel_keys:
                            channels_writer.add_row(
                                build_channel_record(
                                    source_entity,
                                    is_seed=False,
                                    depth=current.depth + 1,
                                )
                            )
                            discovered_channel_keys.add(source_key)

                        from_username = source_entity.username or ""
                        from_title = source_entity.title or ""

                        if current.depth + 1 <= sampling_cfg.max_depth:
                            if source_key not in visited:
                                next_ref = from_username or str(source_entity.id)
                                queue.append(QueueItem(next_ref, current.depth + 1))
                    else:
                        from_name_fallback = str(getattr(fwd_from, "from_name", "") or "")

                    relations_writer.add_row(
                        {
                            "channel_to_id": int(current_entity.id),
                            "channel_to_username": current_entity.username or "",
                            "channel_to_title": current_entity.title or "",
                            "channel_from_id": int(source_id) if source_id is not None else "",
                            "channel_from_username": from_username,
                            "channel_from_title": from_title,
                            "from_name_fallback": from_name_fallback,
                            "target_message_id": int(message.id),
                            "target_message_date_utc": serialize_date(message.date),
                            "sampling_depth": int(current.depth),
                        }
                    )
                    relations_count += 1
                    maybe_flush_writers()

            except FloodWaitError as exc:
                logger.warning(
                    "FloodWait durante lettura canale=%s: pausa %ss",
                    current.channel_ref,
                    exc.seconds,
                )
                await asyncio.sleep(int(exc.seconds) + 1)
                maybe_flush_writers()
            except Exception as exc:
                logger.exception(
                    "Errore durante la scansione del canale=%s: %s",
                    current.channel_ref,
                    exc,
                )
                continue

            maybe_flush_writers()

        for unresolved_seed in sorted(unresolved_seed_refs):
            channels_writer.add_row(build_unresolved_seed_record(unresolved_seed))

        maybe_flush_writers(force=True)
        # Pause 10 seconds between processing batches to avoid hitting rate limits
        await asyncio.sleep(10)
    finally:
        channels_writer.close()
        relations_writer.close()

    seeds_count = len(seed_set)
    total_channels_count = len(discovered_channel_keys) + len(unresolved_seed_refs)
    logger.info(
        "Sampling completato: canali_totali=%s canali_seed=%s relazioni=%s",
        total_channels_count,
        seeds_count,
        relations_count,
    )
    return total_channels_count, seeds_count, relations_count


def write_channels_csv(channels: List[Dict[str, Any]], output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "channel_id",
        "username",
        "title",
        "is_seed",
        "discovered_depth",
        "participants_count",
        "broadcast",
        "megagroup",
        "verified",
        "scam",
        "fake",
        "date_collected_utc",
        "resolution_status",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(channels)


def write_relations_csv(relations: List[Dict[str, Any]], output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "channel_to_id",
        "channel_to_username",
        "channel_to_title",
        "channel_from_id",
        "channel_from_username",
        "channel_from_title",
        "from_name_fallback",
        "target_message_id",
        "target_message_date_utc",
        "sampling_depth",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(relations)


def print_summary(total_channels: int, seeds: int, relations_count: int) -> None:
    discovered = total_channels - seeds

    logger.info("===== SNOWBALL SAMPLING SUMMARY =====")
    logger.info("Canali totali: %s", total_channels)
    logger.info("Canali seed: %s", seeds)
    logger.info("Canali scoperti: %s", discovered)
    logger.info("Relazioni raccolte: %s", relations_count)


def main() -> None:
    # args = parse_args()
    run_cfg = load_run_config()

    configure_logging(run_cfg.sampling.log_level)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    channels_output = with_run_timestamp(run_cfg.channels_output, run_stamp)
    relations_output = with_run_timestamp(run_cfg.relations_output, run_stamp)

    logger.info("Output base canali: %s", run_cfg.channels_output)
    logger.info("Output base relazioni: %s", run_cfg.relations_output)
    logger.info("Output run canali: %s", channels_output)
    logger.info("Output run relazioni: %s", relations_output)

    async def _async_main() -> None:
        client=TelegramClient(
            run_cfg.telegram.session_name,
            run_cfg.telegram.api_id,
            run_cfg.telegram.api_hash,
        )
        await client.connect()
        logger.info("Connessione Telegram avviata")

        if not await client.is_user_authorized():
            qr = await client.qr_login()
            logger.warning("Utente non autorizzato: richiesta autenticazione via QR")
            print("\n📱 QR-Code scannen:")
            print("Telegram → Einstellungen → Geräte → Gerät hinzufügen\n")

            # ASCII QR ohne console-Module
            qr_img = qrcode.QRCode(border=1)
            qr_img.add_data(qr.url)
            qr_img.make(fit=True)

            # ASCII-Ausgabe
            qr_matrix = qr_img.get_matrix()
            for row in qr_matrix:
                print("".join("██" if cell else "  " for cell in row))

            await qr.wait()
            logger.info("Autenticazione QR completata")
        me = await client.get_me()
        logger.info("Autenticato come: %s", me.first_name)

        logger.info("Avvio campionamento snowball")
        logger.info("Canali seed iniziali (%s):", len(run_cfg.seed_channels))
        for seed in run_cfg.seed_channels:
            logger.info(" - %s", seed)
        total_channels, seeds, relations_count = await run_snowball_sampling(
                client=client,
                seed_channels=run_cfg.seed_channels,
                previous_depth=run_cfg.previous_depth,
                sampling_cfg=run_cfg.sampling,
                channels_output_file=channels_output,
                relations_output_file=relations_output,
            )

        print_summary(total_channels, seeds, relations_count)
        logger.info("CSV canali salvato in: %s", channels_output)
        logger.info("CSV relazioni salvato in: %s", relations_output)

    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
