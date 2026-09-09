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

import argparse
import asyncio
import csv
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import Channel, PeerChannel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data_collected"

DEFAULT_CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_CHANNELS_OUTPUT = DATA_DIR / "snowball_channels.csv"
DEFAULT_RELATIONS_OUTPUT = DATA_DIR / "snowball_relations.csv"


@dataclass
class TelegramConfig:
    api_id: int
    api_hash: str
    session_name: str


@dataclass
class SamplingConfig:
    messages_per_channel: int
    max_depth: int


@dataclass
class RunConfig:
    telegram: TelegramConfig
    sampling: SamplingConfig
    seed_channels: List[str]
    channels_output: str
    relations_output: str


@dataclass
class QueueItem:
    channel_ref: str
    depth: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegue snowball sampling tra canali Telegram"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_FILE),
        help="File JSON unico di configurazione",
    )
    parser.add_argument(
        "--channels-output",
        default=str(DEFAULT_CHANNELS_OUTPUT),
        help="CSV output dei canali scoperti",
    )
    parser.add_argument(
        "--relations-output",
        default=str(DEFAULT_RELATIONS_OUTPUT),
        help="CSV output delle relazioni channel_to/channel_from",
    )
    return parser.parse_args()


def normalize_channel_ref(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""

    if value.startswith("https://t.me/"):
        value = value.split("https://t.me/", 1)[1]

    if value.startswith("@"):
        value = value[1:]

    return value.strip()


def load_json_config(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File config non trovato: {file_path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_cfg = json.load(handle)

    if not isinstance(raw_cfg, dict):
        raise ValueError("Il file config deve contenere un oggetto JSON")

    return raw_cfg


def load_run_config(config_file: str) -> RunConfig:
    raw_cfg = load_json_config(config_file)

    telegram_cfg = raw_cfg.get("telegram", {})
    snowball_cfg = raw_cfg.get("snowball", {})

    if not isinstance(telegram_cfg, dict) or not isinstance(snowball_cfg, dict):
        raise ValueError("Sezioni 'telegram' e 'snowball' mancanti o non valide")

    try:
        api_id = int(telegram_cfg["api_id"])
    except KeyError as exc:
        raise ValueError("Campo mancante: telegram.api_id") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("telegram.api_id deve essere un intero") from exc

    api_hash = str(telegram_cfg.get("api_hash", "")).strip()
    session_name = str(telegram_cfg.get("session_name", "")).strip()
    if not api_hash:
        raise ValueError("Campo mancante: telegram.api_hash")
    if not session_name:
        raise ValueError("Campo mancante: telegram.session_name")

    raw_seeds = snowball_cfg.get("seed_channels", [])
    if not isinstance(raw_seeds, list):
        raise ValueError("snowball.seed_channels deve essere una lista")

    seed_channels = [
        normalize_channel_ref(str(item))
        for item in raw_seeds
        if str(item).strip()
    ]
    seed_channels = list(dict.fromkeys(seed_channels))
    if not seed_channels:
        raise ValueError("snowball.seed_channels non puo essere vuoto")

    try:
        messages_per_channel = int(snowball_cfg.get("messages_per_channel", 100))
        max_depth = int(snowball_cfg.get("max_depth", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Valori snowball numerici non validi") from exc

    if messages_per_channel <= 0:
        raise ValueError("snowball.messages_per_channel deve essere > 0")
    if max_depth < 0:
        raise ValueError("snowball.max_depth deve essere >= 0")

    channels_output = str(
        snowball_cfg.get("channels_output_csv", DEFAULT_CHANNELS_OUTPUT)
    ).strip() or str(DEFAULT_CHANNELS_OUTPUT)
    relations_output = str(
        snowball_cfg.get("relations_output_csv", DEFAULT_RELATIONS_OUTPUT)
    ).strip() or str(DEFAULT_RELATIONS_OUTPUT)

    return RunConfig(
        telegram=TelegramConfig(api_id=api_id, api_hash=api_hash, session_name=session_name),
        sampling=SamplingConfig(
            messages_per_channel=messages_per_channel,
            max_depth=max_depth,
        ),
        seed_channels=seed_channels,
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


async def resolve_channel_entity(
    client: TelegramClient,
    channel_ref: str,
) -> Optional[Channel]:
    try:
        entity = await client.get_entity(channel_ref)
    except FloodWaitError as exc:
        await asyncio.sleep(int(exc.seconds) + 1)
        entity = await client.get_entity(channel_ref)
    except Exception:
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
        await asyncio.sleep(int(exc.seconds) + 1)
        source_entity = await client.get_entity(from_peer)
    except Exception:
        source_entity = None

    if isinstance(source_entity, Channel):
        return source_entity, source_id

    return None, source_id


async def run_snowball_sampling(
    client: TelegramClient,
    seed_channels: List[str],
    sampling_cfg: SamplingConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    queue: Deque[QueueItem] = deque([QueueItem(ch, 0) for ch in seed_channels])
    visited: Set[str] = set()

    channels_by_key: Dict[str, Dict[str, Any]] = {}
    relations: List[Dict[str, Any]] = []

    seed_set = {normalize_channel_ref(item) for item in seed_channels if item.strip()}
    unresolved_seed_refs: Set[str] = set(seed_set)

    while queue:
        current = queue.popleft()
        if current.depth > sampling_cfg.max_depth:
            continue

        current_entity = await resolve_channel_entity(client, current.channel_ref)
        if current_entity is None:
            continue

        current_key = channel_key_from_id(int(current_entity.id))
        if current_key in visited:
            continue

        visited.add(current_key)

        is_seed = normalize_channel_ref(current.channel_ref) in seed_set
        if is_seed:
            unresolved_seed_refs.discard(normalize_channel_ref(current.channel_ref))

        channels_by_key[current_key] = build_channel_record(
            current_entity,
            is_seed=is_seed,
            depth=current.depth,
        )

        try:
            iterator = client.iter_messages(
                entity=current_entity,
                limit=sampling_cfg.messages_per_channel,
            )

            async for message in iterator:
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
                    if source_key not in channels_by_key:
                        channels_by_key[source_key] = build_channel_record(
                            source_entity,
                            is_seed=False,
                            depth=current.depth + 1,
                        )

                    from_username = source_entity.username or ""
                    from_title = source_entity.title or ""

                    if current.depth + 1 <= sampling_cfg.max_depth:
                        if source_key not in visited:
                            next_ref = from_username or str(source_entity.id)
                            queue.append(QueueItem(next_ref, current.depth + 1))
                else:
                    from_name_fallback = str(getattr(fwd_from, "from_name", "") or "")

                relations.append(
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

        except FloodWaitError as exc:
            await asyncio.sleep(int(exc.seconds) + 1)
        except Exception:
            continue

    for unresolved_seed in sorted(unresolved_seed_refs):
        placeholder_key = f"seed_ref:{unresolved_seed}"
        channels_by_key[placeholder_key] = build_unresolved_seed_record(unresolved_seed)

    channels = list(channels_by_key.values())
    channels.sort(
        key=lambda row: (
            _safe_int(row.get("discovered_depth", 0), 0),
            1 if str(row.get("channel_id", "")).strip() == "" else 0,
            _safe_int(row.get("channel_id", 0), 0),
            str(row.get("username", "")).lower(),
        )
    )

    relations.sort(
        key=lambda row: (
            int(row["sampling_depth"]),
            int(row["channel_to_id"]),
            int(row["target_message_id"]),
        )
    )

    return channels, relations


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


def print_summary(channels: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> None:
    seeds = sum(1 for item in channels if int(item.get("is_seed", 0)) == 1)
    discovered = len(channels) - seeds

    print("\n===== SNOWBALL SAMPLING SUMMARY =====")
    print(f"Canali totali: {len(channels)}")
    print(f"Canali seed: {seeds}")
    print(f"Canali scoperti: {discovered}")
    print(f"Relazioni raccolte: {len(relations)}")


def main() -> None:
    args = parse_args()
    run_cfg = load_run_config(args.config)

    channels_output = args.channels_output if args.channels_output else run_cfg.channels_output
    relations_output = (
        args.relations_output if args.relations_output else run_cfg.relations_output
    )

    async def _async_main() -> None:
        async with TelegramClient(
            run_cfg.telegram.session_name,
            run_cfg.telegram.api_id,
            run_cfg.telegram.api_hash,
        ) as client:
            channels, relations = await run_snowball_sampling(
                client=client,
                seed_channels=run_cfg.seed_channels,
                sampling_cfg=run_cfg.sampling,
            )

        write_channels_csv(channels, channels_output)
        write_relations_csv(relations, relations_output)

        print_summary(channels, relations)
        print(f"CSV canali salvato in: {channels_output}")
        print(f"CSV relazioni salvato in: {relations_output}")

    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
