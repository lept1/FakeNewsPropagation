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
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import PeerChannel, PeerChat, PeerUser

from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent

CONFIG_DIR = PARENT_DIR / "config"

DATA_DIR = PARENT_DIR / "data_collected"

DEFAULT_OUTPUT_CSV = DATA_DIR / "telegram_fakenews_analysis.csv"
DEFAULT_SNOWBALL_CHANNELS_CSV = DATA_DIR / "snowball_channels.csv"
INPUT_CONFIG = CONFIG_DIR / "config.json"


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


def load_json_config(config_file: str) -> Dict[str, Any]:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"File configurazione non trovato: {config_file}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Il file di configurazione deve contenere un oggetto JSON")

    return data


def parse_iso_datetime(raw_value: Optional[str], *, is_end: bool) -> Optional[datetime]:
    if not raw_value:
        return None

    value = raw_value.strip()
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        if "T" not in value:
            dt_time = time.max.replace(microsecond=0) if is_end else time.min
            dt = datetime.combine(dt.date(), dt_time)
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def load_app_config(config_file: str) -> AppConfig:
    cfg = load_json_config(config_file)

    telegram = cfg.get("telegram", {})
    message_cfg = cfg.get("message_collection", {})
    snowball_cfg = cfg.get("snowball", {})

    if not isinstance(telegram, dict) or not isinstance(message_cfg, dict):
        raise ValueError("Sezioni 'telegram' e 'message_collection' mancanti o non valide")
    if not isinstance(snowball_cfg, dict):
        raise ValueError("Sezione 'snowball' mancante o non valida")

    try:
        api_id = int(telegram["api_id"])
    except KeyError as exc:
        raise ValueError("Campo mancante: telegram.api_id") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("telegram.api_id deve essere un intero") from exc

    api_hash = str(telegram.get("api_hash", "")).strip()
    session_name = str(telegram.get("session_name", "")).strip()
    if not api_hash:
        raise ValueError("Campo mancante: telegram.api_hash")
    if not session_name:
        raise ValueError("Campo mancante: telegram.session_name")

    raw_keywords = message_cfg.get("keywords", [])
    if not isinstance(raw_keywords, list):
        raise ValueError("message_collection.keywords deve essere una lista")
    keywords = [str(item).strip() for item in raw_keywords if str(item).strip()]
    if not keywords:
        keywords = [""]

    output_csv = str(message_cfg.get("output_csv", DEFAULT_OUTPUT_CSV)).strip()
    if not output_csv:
        output_csv = str(DEFAULT_OUTPUT_CSV)

    channels_source_csv = str(
        message_cfg.get(
            "channels_source_csv",
            DEFAULT_SNOWBALL_CHANNELS_CSV,
        )
    ).strip()

    channels_csv_column = str(message_cfg.get("channels_csv_column", "username")).strip()
    if not channels_csv_column:
        channels_csv_column = "username"

    start_date = parse_iso_datetime(
        str(message_cfg.get("start_date")).strip(),
        is_end=False,
    ) if message_cfg.get("start_date") else None
    end_date = parse_iso_datetime(
        str(message_cfg.get("end_date")).strip(),
        is_end=True,
    ) if message_cfg.get("end_date") else None

    if start_date and end_date and start_date > end_date:
        raise ValueError("message_collection.start_date non puo essere successiva a end_date")

    return AppConfig(
        api_id=api_id,
        api_hash=api_hash,
        session_name=session_name,
        channels_source_csv=channels_source_csv,
        channels_csv_column=channels_csv_column,
        keywords=keywords,
        output_csv=output_csv,
        start_date=start_date,
        end_date=end_date,
    )


def normalize_channel(channel: str) -> str:
    return channel.lstrip("@").strip()


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
            print(f"[FloodWait] Attendo {wait_seconds}s prima di riprendere...")
            await asyncio.sleep(wait_seconds)


async def collect_messages(
    client: TelegramClient,
    channels: Sequence[str],
    keywords: Optional[Sequence[str]],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    results: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for raw_channel in channels:
        channel = normalize_channel(raw_channel)
        for keyword in keywords:
            if keyword == "":
                keyword = None
            async for message in iter_messages_with_flood_wait(
                client,
                entity=channel,
                search=keyword,
                limit=limit
            ):
                text = message.message or ""
                if not text:
                    continue

                if not message_matches_date_range(message.date, start_date, end_date):
                    continue

                key = (channel, int(message.id))
                results[key] = {
                    "message_id": int(message.id),
                    "channel_username": channel,
                    "date": message.date.astimezone(timezone.utc).isoformat(),
                    "text": text,
                    "views": int(message.views or 0),
                    "forwards": int(message.forwards or 0),
                    "is_forwarded": bool(getattr(message, "fwd_from", None)),
                    "forward_from_chat": extract_forward_source(message),
                }

    return list(results.values())


def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "message_id",
        "channel_username",
        "date",
        "text",
        "views",
        "forwards",
        "is_forwarded",
        "forward_from_chat",
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)
    return df[columns].sort_values(by=["channel_username", "date", "message_id"]).reset_index(drop=True)


def save_to_csv(df: pd.DataFrame, output_path: str = str(DEFAULT_OUTPUT_CSV)) -> None:
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_ALL,
    )


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
    config = load_app_config(str(INPUT_CONFIG))
    channels = load_channels_from_snowball_csv(
        config.channels_source_csv,
        config.channels_csv_column,
    )
    keywords = config.keywords
    output_csv = config.output_csv

    #client = TelegramClient(config.session_name, config.api_id, config.api_hash)

    async with TelegramClient(config.session_name, config.api_id, config.api_hash) as client:
        records = await collect_messages(
            client=client,
            channels=channels,
            keywords=keywords,
            start_date=config.start_date,
            end_date=config.end_date,
        )

    df = build_dataframe(records)
    save_to_csv(df, output_csv)
    print(f"CSV salvato in: {output_csv}")
    print_summary(df)


if __name__ == "__main__":
    asyncio.run(main())
