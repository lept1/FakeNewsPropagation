#!/usr/bin/env python3
"""
Analisi della diffusione di notizie/fake news su Telegram tramite parole chiave.

Installazione dipendenze:
    pip install telethon pandas python-dotenv

Esempio di esecuzione:
    python3 telegram_fake_news_datacollection.py.py \
        --channels-file channels.txt \
        --keywords "fake news" disinformazione \
        --start-date 2026-01-01 \
        --end-date 2026-12-31

Credenziali richieste (in .env o variabili d'ambiente):
    API_ID=123456
    API_HASH=abcdef1234567890abcdef1234567890
    TG_SESSION=telegram_analysis_session

Opzionale file JSON di configurazione:
    {
      "API_ID": 123456,
      "API_HASH": "abcdef1234567890abcdef1234567890",
      "TG_SESSION": "telegram_analysis_session"
    }
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import PeerChannel, PeerChat, PeerUser

from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent

VAR_DIR = PARENT_DIR / "var"

DATA_DIR = PARENT_DIR / "data_collected"

OUTPUT_CSV = DATA_DIR / "telegram_fakenews_analysis.csv"

INPUT_CHANNELS_FILE = VAR_DIR / "channels.txt"

INPUT_ENV = VAR_DIR / "config.env"


@dataclass
class AppConfig:
    api_id: int
    api_hash: str
    session_name: str = "telegram_analysis_session"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analizza la diffusione di fake news su canali Telegram pubblici"
    )
    parser.add_argument(
        "--channels-file",
        default=str(INPUT_CHANNELS_FILE),
        help="Path a file TXT con un canale Telegram pubblico per riga",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Una o piu parole chiave da cercare",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Data/ora inizio (ISO), es: 2026-01-01 o 2026-01-01T08:30:00+00:00",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Data/ora fine (ISO), es: 2026-12-31 o 2026-12-31T23:59:59+00:00",
    )
    parser.add_argument(
        "--config-file",
        default=str(INPUT_ENV),
        help="Percorso file .env con API_ID/API_HASH/TG_SESSION",
    )
    return parser.parse_args()


def load_channels_from_txt(channels_file: str) -> List[str]:
    if not os.path.exists(channels_file):
        raise FileNotFoundError(f"File canali non trovato: {channels_file}")

    channels: List[str] = []
    with open(channels_file, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            channels.append(normalize_channel(line))

    # Rimuove eventuali duplicati mantenendo l'ordine originale.
    unique_channels = list(dict.fromkeys(channels))
    if not unique_channels:
        raise ValueError(
            "Il file canali e vuoto: aggiungi almeno un canale Telegram pubblico per riga"
        )

    return unique_channels


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


# def load_json_config(config_file: Optional[str]) -> Dict[str, Any]:
#     if not config_file:
#         return {}

#     config_path = Path(config_file)
#     if not config_path.exists():
#         raise FileNotFoundError(f"File configurazione non trovato: {config_file}")

#     with config_path.open("r", encoding="utf-8") as fh:
#         data = json.load(fh)

#     if not isinstance(data, dict):
#         raise ValueError("Il file di configurazione deve contenere un oggetto JSON")
#     return data


def load_app_config(config_file: Optional[str]) -> AppConfig:
    load_dotenv(
        dotenv_path=config_file
    )

    api_id_raw = os.getenv("API_ID")
    if not api_id_raw:
        raise ValueError("API_ID mancante. Imposta API_ID in variabili d'ambiente/.env")
    api_hash = os.getenv("API_HASH")
    if not api_hash:
        raise ValueError("API_HASH mancante. Imposta API_HASH in variabili d'ambiente/.env")

    session_name = os.getenv("TG_SESSION")
    if not session_name:
        raise ValueError("TG_SESSION mancante. Imposta TG_SESSION in variabili d'ambiente/.env")

    try:
        api_id = int(api_id_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("API_ID deve essere un intero valido") from exc

    return AppConfig(api_id=api_id, api_hash=str(api_hash), session_name=str(session_name))


def normalize_channel(channel: str) -> str:
    return channel.lstrip("@").strip()


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
    keywords: Sequence[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> List[Dict[str, Any]]:
    results: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for raw_channel in channels:
        channel = normalize_channel(raw_channel)

        for keyword in keywords:
            async for message in iter_messages_with_flood_wait(
                client,
                entity=channel,
                search=keyword,
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


def save_to_csv(df: pd.DataFrame, output_path: str = OUTPUT_CSV) -> None:
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
    args = parse_args()
    config = load_app_config(args.config_file)
    channels = load_channels_from_txt(args.channels_file)

    start_date = parse_iso_datetime(args.start_date, is_end=False)
    end_date = parse_iso_datetime(args.end_date, is_end=True)

    if start_date and end_date and start_date > end_date:
        raise ValueError("start-date non puo essere successiva a end-date")

    #client = TelegramClient(config.session_name, config.api_id, config.api_hash)

    async with TelegramClient(config.session_name, config.api_id, config.api_hash) as client:
        records = await collect_messages(
            client=client,
            channels=channels,
            keywords=args.keywords,
            start_date=start_date,
            end_date=end_date,
        )

    df = build_dataframe(records)
    save_to_csv(df, OUTPUT_CSV)
    print(f"CSV salvato in: {OUTPUT_CSV}")
    print_summary(df)


if __name__ == "__main__":
    asyncio.run(main())
