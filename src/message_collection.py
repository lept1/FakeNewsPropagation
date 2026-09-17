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
import argparse
import csv
import logging
from collections import Counter, defaultdict
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

PARENT_DIR = Path(__file__).parent.parent

CONFIG_DIR = PARENT_DIR / "config"

DATA_DIR = PARENT_DIR / "data_collected"

DEFAULT_OUTPUT_CSV = DATA_DIR / "telegram_fakenews_analysis.csv"
DEFAULT_SNOWBALL_CHANNELS_CSV = DATA_DIR / "snowball_channels.csv"
INPUT_CONFIG = CONFIG_DIR / "config.json"
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

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    api_id: int
    api_hash: str
    session_name: str
    channels_source_csv: str
    channels_csv_column: str
    keywords: List[str]
    output_csv: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    limit: Optional[int]
    log_level: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raccoglie messaggi Telegram da canali e keyword"
    )
    parser.add_argument(
        "--config",
        default=str(INPUT_CONFIG),
        help="File JSON unico di configurazione",
    )
    parser.add_argument(
        "--log-level",
        default="",
        help="Override livello log (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_app_config(config_file: str) -> AppConfig:
    logger.debug("Caricamento configurazione da: %s", config_file)
    project_cfg = load_project_config(config_file)
    telegram = project_cfg.telegram
    message_cfg = project_cfg.message_collection

    return AppConfig(
        api_id=telegram.api_id,
        api_hash=telegram.api_hash,
        session_name=telegram.session_name,
        channels_source_csv=message_cfg.channels_source_csv,
        channels_csv_column=message_cfg.channels_csv_column,
        keywords=message_cfg.keywords,
        output_csv=message_cfg.output_csv,
        start_date=message_cfg.start_date,
        end_date=message_cfg.end_date,
        limit=message_cfg.limit,
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


def load_channels_from_snowball_csv(csv_path: str, channel_column: str) -> List[str]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV canali snowball non trovato: {csv_path}")

    df = pd.read_csv(path)
    if channel_column not in df.columns:
        columns = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Colonna '{channel_column}' non trovata in {csv_path}. Colonne disponibili: {columns}"
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
) -> int:
    total_written = 0
    seen_keys: set[Tuple[str, int]] = set()
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
        append_records_to_csv(channel_rows, output_csv)
        for key in channel_results:
            seen_keys.add(key)
        total_written += len(channel_rows)
        logger.info(
            "Canale completato=%s record_scritti=%s totale_progressivo=%s",
            channel,
            len(channel_rows),
            total_written,
        )

    logger.info("Raccolta completata: record unici=%s", total_written)
    return total_written


def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        logger.warning("Nessun record da convertire in DataFrame")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(records)
    logger.info("DataFrame costruito: righe=%s", len(df))
    return df[OUTPUT_COLUMNS].sort_values(by=["channel_username", "date", "message_id"]).reset_index(drop=True)


def load_results_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        logger.warning("CSV output non trovato, DataFrame vuoto: %s", csv_path)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[OUTPUT_COLUMNS].sort_values(by=["channel_username", "date", "message_id"]).reset_index(drop=True)


def save_to_csv(df: pd.DataFrame, output_path: str = str(DEFAULT_OUTPUT_CSV)) -> None:
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
    args = parse_args()
    config = load_app_config(str(args.config))
    effective_log_level = (
        validate_log_level(args.log_level, section_name="message_collection.log_level")
        if args.log_level
        else config.log_level
    )
    configure_logging(effective_log_level)
    logger.info("Logger configurato con livello: %s", effective_log_level)
    logger.info("Caricamento canali da CSV: %s", config.channels_source_csv)

    channels = load_channels_from_snowball_csv(
        config.channels_source_csv,
        config.channels_csv_column,
    )
    logger.info("Canali caricati: %s", len(channels))

    keywords = config.keywords
    output_csv = config.output_csv
    logger.info("Keyword configurate: %s", len(keywords))
    logger.info("Output CSV configurato: %s", output_csv)
    logger.info("Inizializzazione output CSV incrementale")
    initialize_output_csv(output_csv)

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
        )

    logger.info("Totale record scritti su CSV: %s", total_written)
    df = load_results_dataframe(output_csv)
    print(f"CSV salvato in: {output_csv}")
    print_summary(df)
    logger.info("Esecuzione completata")


if __name__ == "__main__":
    asyncio.run(main())
