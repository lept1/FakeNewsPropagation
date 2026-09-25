#!/usr/bin/env python3
"""
Analisi della diffusione di notizie/fake news su Telegram tramite parole chiave.

Installazione dipendenze:
    pip install telethon pandas

Configurazione:
    config/config.json

Esempio di esecuzione:
    python src/message_collection.py
"""

from __future__ import annotations

import asyncio
import csv
import logging
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import PeerChannel, PeerChat, PeerUser

from pathlib import Path

try:
    from .config import load_project_config, validate_log_level
except ImportError:
    from config import load_project_config, validate_log_level

OUTPUT_COLUMNS = [
    "message_id",
    "channel_username",
    "date",
    "text",
    "views",
    "forwards",
    "is_forwarded",
    "forward_from_chat",
]
CHANNELS_TRACE_COLUMNS = [
    "channel_username",
    "messages_written",
]

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    api_id: int
    api_hash: str
    session_name: str
    channels_source_csv: Optional[str]
    channels_csv_column: str
    elaborated_channels_csv: str
    keywords: List[str]
    output_csv: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    limit: Optional[int]
    save_interval_seconds: int
    log_level: str

# def with_run_timestamp(output_file: str, run_stamp: str) -> str:
#     output_path = Path(output_file)
#     suffix = output_path.suffix or ".csv"
#     filename = f"{output_path.stem}_{run_stamp}{suffix}"
#     return str(output_path.with_name(filename))


def configure_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
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
        method: str = "w",
    ) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = output_path.open(method, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        self._writer.writeheader()

        self._pending_rows: List[Dict[str, Any]] = []
        self._flush_interval_seconds = flush_interval_seconds
        self._last_flush_ts = asyncio.get_running_loop().time()

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        if rows:
            self._pending_rows.extend(rows)

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


def load_app_config() -> AppConfig:
    logger.debug("Caricamento configurazione dall'ambiente di progetto...")
    project_cfg = load_project_config()
    telegram = project_cfg.telegram
    message_cfg = project_cfg.message_collection

    return AppConfig(
        api_id=telegram.api_id,
        api_hash=telegram.api_hash,
        session_name=telegram.session_name,
        channels_source_csv=message_cfg.channels_source_csv,
        channels_csv_column=message_cfg.channels_csv_column,
        elaborated_channels_csv=message_cfg.elaborated_channels_csv,
        keywords=message_cfg.keywords,
        output_csv=message_cfg.output_csv,
        start_date=message_cfg.start_date,
        end_date=message_cfg.end_date,
        limit=message_cfg.limit,
        save_interval_seconds=message_cfg.save_interval_seconds,
        log_level=message_cfg.log_level,
    )


def normalize_channel(channel: str) -> str:
    return channel.lstrip("@").strip()


def initialize_output_csv(output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()


def append_records_to_csv(records: List[Dict[str, Any]], output_path: str) -> None:
    if not records:
        return

    path = Path(output_path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writerows(records)


def load_channels_from_snowball_csv(csv_paths: List[str], channel_column: str) -> List[str]:
    df_list: List[pd.DataFrame] = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV canali snowball non trovato: {csv_path}")
        else:
            df_list.append(pd.read_csv(path))


    df = pd.concat(df_list, ignore_index=True)
    if channel_column not in df.columns:
        columns = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Colonna '{channel_column}' non trovata nei file CSV. Colonne disponibili: {columns}"
        )

    channels = [
        normalize_channel(str(item))
        for item in df[channel_column].fillna("").astype(str).tolist()
        if str(item).strip()
    ]

    unique_channels = list(dict.fromkeys(channels))
    if not unique_channels:
        raise ValueError("Nessun canale valido trovato nel CSV generato da snowball_sampling")

    return unique_channels


def message_matches_date_range(
    message_date: datetime,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> bool:
    msg_utc = message_date.astimezone(timezone.utc)
    if start_date and msg_utc < start_date:
        return False
    if end_date and msg_utc > end_date:
        return False
    return True


def peer_to_identifier(peer: Any) -> Optional[str]:
    if peer is None:
        return None

    if isinstance(peer, PeerChannel):
        return f"channel_id:{peer.channel_id}"
    if isinstance(peer, PeerChat):
        return f"chat_id:{peer.chat_id}"
    if isinstance(peer, PeerUser):
        return f"user_id:{peer.user_id}"

    for attr_name in ("channel_id", "chat_id", "user_id"):
        if hasattr(peer, attr_name):
            return f"{attr_name}:{getattr(peer, attr_name)}"

    return str(peer)


def extract_forward_source(message: Any) -> Optional[str]:
    fwd = getattr(message, "fwd_from", None)
    if not fwd:
        return None

    if getattr(fwd, "from_name", None):
        return str(fwd.from_name)

    source_peer = getattr(fwd, "from_id", None) or getattr(fwd, "saved_from_peer", None)
    source_id = peer_to_identifier(source_peer)
    if source_id:
        return source_id

    if getattr(fwd, "saved_from_msg_id", None):
        return f"saved_msg_id:{fwd.saved_from_msg_id}"

    return "unknown_forward_source"


async def iter_messages_with_flood_wait(client: TelegramClient, **kwargs: Any):
    while True:
        try:
            async for message in client.iter_messages(**kwargs):
                yield message
            return
        except FloodWaitError as exc:
            wait_seconds = int(exc.seconds) + 1
            logger.warning("FloodWait rilevato: attendo %ss prima di riprendere", wait_seconds)
            await asyncio.sleep(wait_seconds)


async def collect_messages(
    client: TelegramClient,
    channels: Sequence[str],
    keywords: Optional[Sequence[str]],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    limit: Optional[int],
    output_csv: str,
    elaborated_channels_csv: str,
    save_interval_seconds: int,
) -> int:
    total_written = 0
    seen_keys: set[Tuple[str, int]] = set()
    messages_writer = IncrementalCsvWriter(
        output_file=output_csv,
        fieldnames=OUTPUT_COLUMNS,
        flush_interval_seconds=save_interval_seconds,
    )
    channels_writer = IncrementalCsvWriter(
        output_file=elaborated_channels_csv,
        fieldnames=CHANNELS_TRACE_COLUMNS,
        flush_interval_seconds=save_interval_seconds,
        method="a",
    )

    def maybe_flush_writers(force: bool = False) -> None:
        messages_written = messages_writer.maybe_flush(force=force)
        channels_written = channels_writer.maybe_flush(force=force)
        if messages_written > 0 or channels_written > 0:
            logger.info(
                "Checkpoint CSV salvato (%ss): +messaggi=%s, +canali=%s",
                save_interval_seconds,
                messages_written,
                channels_written,
            )

    logger.info(
        "Avvio raccolta messaggi: canali=%s keyword=%s limit=%s",
        len(channels),
        len(keywords or []),
        limit,
    )
    logger.info(
        "Filtro date UTC: start=%s end=%s",
        start_date.isoformat() if start_date else "None",
        end_date.isoformat() if end_date else "None",
    )

    try:
        for raw_channel in channels:
            channel = normalize_channel(raw_channel)
            logger.info("Elaborazione canale: %s", channel)
            channel_results: Dict[Tuple[str, int], Dict[str, Any]] = {}
            for keyword in keywords:
                if keyword == "":
                    keyword = None
                logger.debug(
                    "Lettura messaggi canale=%s keyword=%s",
                    channel,
                    keyword if keyword is not None else "<tutte>",
                )

                scanned_count = 0
                matched_count = 0
                last_message_id: Optional[int] = None
                async for message in iter_messages_with_flood_wait(
                    client,
                    entity=channel,
                    search=keyword,
                    limit=limit
                ):
                    scanned_count += 1
                    if scanned_count % 25 == 0:
                        logger.debug(
                            "Progresso lettura canale=%s keyword=%s: messaggi_scansionati=%s ultimo_message_id=%s",
                            channel,
                            keyword if keyword is not None else "<tutte>",
                            scanned_count,
                            last_message_id,
                        )

                    text = message.message or ""
                    if not text:
                        last_message_id = int(message.id)
                        continue

                    if not message_matches_date_range(message.date, start_date, end_date):
                        last_message_id = int(message.id)
                        continue

                    key = (channel, int(message.id))
                    if key in seen_keys:
                        last_message_id = int(message.id)
                        continue

                    channel_results[key] = {
                        "message_id": int(message.id),
                        "channel_username": channel,
                        "date": message.date.astimezone(timezone.utc).isoformat(),
                        "text": text,
                        "views": int(message.views or 0),
                        "forwards": int(message.forwards or 0),
                        "is_forwarded": bool(getattr(message, "fwd_from", None)),
                        "forward_from_chat": extract_forward_source(message),
                    }
                    matched_count += 1
                    last_message_id = int(message.id)

                logger.info(
                    "Completata lettura canale=%s keyword=%s: scansionati=%s validi=%s",
                    channel,
                    keyword if keyword is not None else "<tutte>",
                    scanned_count,
                    matched_count,
                )

            channel_rows = list(channel_results.values())
            messages_writer.add_rows(channel_rows)
            for key in channel_results:
                seen_keys.add(key)
            if channel_rows:
                channels_writer.add_row(
                    {
                        "channel_username": channel,
                        "messages_written": len(channel_rows),
                    }
                )
            total_written += len(channel_rows)
            logger.info(
                "Canale completato=%s record_scritti=%s totale_progressivo=%s",
                channel,
                len(channel_rows),
                total_written,
            )
            maybe_flush_writers()

        maybe_flush_writers(force=True)
    finally:
        messages_writer.close()
        channels_writer.close()

    logger.info("Raccolta completata: record unici=%s", total_written)
    return total_written


# def build_channels_trace_output_path(messages_output_path: str, run_stamp: str) -> str:
#     output_path = Path(messages_output_path)
#     suffix = output_path.suffix or ".csv"
#     channels_filename = f"{output_path.stem}_channels_{run_stamp}{suffix}"
#     return str(output_path.with_name(channels_filename))


# def save_elaborated_channels_csv(channel_message_counts: Dict[str, int], output_path: str) -> None:
#     path = Path(output_path)
#     path.parent.mkdir(parents=True, exist_ok=True)

#     rows = [
#         {
#             "channel_username": channel,
#             "messages_written": count,
#         }
#         for channel, count in sorted(channel_message_counts.items())
#     ]

#     with path.open("a", encoding="utf-8", newline="") as handle:
#         writer = csv.DictWriter(
#             handle,
#             fieldnames=CHANNELS_TRACE_COLUMNS,
#             quoting=csv.QUOTE_ALL,
#             lineterminator="\n",
#         )
#         writer.writeheader()
#         writer.writerows(rows)


# def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
#     if not records:
#         logger.warning("Nessun record da convertire in DataFrame")
#         return pd.DataFrame(columns=OUTPUT_COLUMNS)

#     df = pd.DataFrame(records)
#     logger.info("DataFrame costruito: righe=%s", len(df))
#     return df[OUTPUT_COLUMNS].sort_values(by=["channel_username", "date", "message_id"]).reset_index(drop=True)


# def load_results_dataframe(csv_path: str) -> pd.DataFrame:
#     path = Path(csv_path)
#     if not path.exists():
#         logger.warning("CSV output non trovato, DataFrame vuoto: %s", csv_path)
#         return pd.DataFrame(columns=OUTPUT_COLUMNS)

#     df = pd.read_csv(path)
#     if df.empty:
#         return pd.DataFrame(columns=OUTPUT_COLUMNS)

#     for column in OUTPUT_COLUMNS:
#         if column not in df.columns:
#             df[column] = ""

#     return df[OUTPUT_COLUMNS].sort_values(by=["channel_username", "date", "message_id"]).reset_index(drop=True)


def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    logger.info("Salvataggio CSV in corso: %s", output_path)
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_ALL,
    )
    logger.info("Salvataggio CSV completato: %s", output_path)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Nessun messaggio trovato con i filtri specificati.")
        return

    messages_per_channel = defaultdict(int)
    for channel in df["channel_username"]:
        messages_per_channel[str(channel)] += 1

    total_views = int(df["views"].fillna(0).sum())

    forward_sources = [
        src
        for src in df["forward_from_chat"].fillna("").astype(str).tolist()
        if src.strip()
    ]
    top_sources = Counter(forward_sources).most_common(3)

    print("\n===== RIEPILOGO ANALISI TELEGRAM =====")
    print("Messaggi trovati per canale:")
    for channel, count in sorted(messages_per_channel.items()):
        print(f"  - {channel}: {count}")

    print(f"Visualizzazioni aggregate totali: {total_views}")

    if top_sources:
        print("Top 3 sorgenti forward:")
        for source, count in top_sources:
            print(f"  - {source}: {count}")
    else:
        print("Top 3 sorgenti forward: nessun forward trovato")


async def main() -> None:
    config = load_app_config()
    effective_log_level = validate_log_level(
        config.log_level,
        section_name="message_collection.log_level",
    )
    configure_logging(effective_log_level)
    logger.info("Logger configurato con livello: %s", effective_log_level)
    if len(config.channels_source_csv) == 0:
        logger.error("Nessun file CSV di canali specificato in channels_source_csv")
    else:
        logger.info("Caricamento canali da CSV: %s", config.channels_source_csv)
        channels = load_channels_from_snowball_csv(
            config.channels_source_csv,
            config.channels_csv_column,
        )

    elaborated_channels_csv = config.elaborated_channels_csv
    # read already elaborated channels if the file exists
    elaborated_channels = []
    if Path(elaborated_channels_csv).exists():
        elaborated_channels = load_channels_from_snowball_csv(
            [elaborated_channels_csv],
            config.channels_csv_column,
        )
    # remove already elaborated channels from the list of channels to be processed
    channels = [channel for channel in channels if channel not in elaborated_channels]
    logger.info("Canali caricati: %s", len(channels))

    keywords = config.keywords
    output_csv = config.output_csv
    elaborated_channels_csv = config.elaborated_channels_csv
    logger.info("Keyword configurate: %s", len(keywords))
    logger.info("Output CSV configurato: %s", output_csv)
    logger.info("Output CSV canali estratti: %s", elaborated_channels_csv)
    logger.info(
        "Inizializzazione scrittura incrementale (flush ogni %ss)",
        config.save_interval_seconds,
    )

    logger.info("Avvio client Telegram con sessione: %s", config.session_name)
    async with TelegramClient(config.session_name, config.api_id, config.api_hash) as client:
        logger.info("Client Telegram connesso")
        total_written = await collect_messages(
            client=client,
            channels=channels,
            keywords=keywords,
            start_date=config.start_date,
            end_date=config.end_date,
            limit=config.limit,
            output_csv=output_csv,
            elaborated_channels_csv=elaborated_channels_csv,
            save_interval_seconds=config.save_interval_seconds,
        )

    logger.info("Totale record scritti su CSV: %s", total_written)
    logger.info("CSV canali estratti salvato: %s", elaborated_channels_csv)
    print(f"CSV salvato in: {output_csv}")
    print(f"CSV canali estratti salvato in: {elaborated_channels_csv}")
    # print_summary(df)
    logger.info("Esecuzione completata")


if __name__ == "__main__":
    asyncio.run(main())
