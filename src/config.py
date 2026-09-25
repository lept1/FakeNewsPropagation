from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data_collected"

CONFIG_FILE = CONFIG_DIR / "config.json"
MESSAGE_OUTPUT_DIR = DATA_DIR /"message_collection"
SNOWBALL_CHANNELS_OUTPUT_DIR = DATA_DIR/ "snowball_channels"
SNOWBALL_RELATIONS_OUTPUT_DIR = DATA_DIR/ "snowball_relations"
RELATION_MATRIX_OUTPUT_DIR = DATA_DIR / "relation_matrix"
GRAPH_OUTPUT_DIR = DATA_DIR / "relation_graph"

# if config directory and config file do not exist raise an error
if not CONFIG_DIR.exists():
    raise FileNotFoundError(f"Config directory not found: {CONFIG_DIR}")
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")


@dataclass(frozen=True)
class TelegramConfig:
    api_id: int
    api_hash: str
    session_name: str


@dataclass(frozen=True)
class MessageCollectionConfig:
    channels_source_csv: List[str]
    channels_csv_column: str
    elaborated_channels_csv: str
    keywords: List[str]
    output_csv: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    limit: int
    save_interval_seconds: int
    log_level: str


@dataclass(frozen=True)
class SnowballConfig:
    restart: bool
    seed_channels: List[str]
    messages_per_channel: int
    max_depth: int
    save_interval_seconds: int
    log_level: str
    channels_output_csv: str
    relations_output_csv: str


@dataclass(frozen=True)
class GraphAnalysisConfig:
    relations_input_dir: str
    relation_matrix_output_csv: str
    graph_output_prefix: str
    min_weight: int
    include_unresolved_sources: bool


@dataclass(frozen=True)
class ProjectConfig:
    telegram: TelegramConfig
    message_collection: MessageCollectionConfig
    snowball: SnowballConfig
    graph_analysis: GraphAnalysisConfig


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


def validate_log_level(log_level: Any, *, section_name: str) -> str:
    normalized = str(log_level).strip().upper() or "INFO"
    if not isinstance(getattr(logging, normalized, None), int):
        raise ValueError(
            f"{section_name} deve essere uno tra DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    return normalized


def _load_json_config(config_file: str) -> Dict[str, Any]:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"File configurazione non trovato: {config_file}")

    with path.open("r", encoding="utf-8") as handle:
        raw_cfg = json.load(handle)

    if not isinstance(raw_cfg, dict):
        raise ValueError("Il file di configurazione deve contenere un oggetto JSON")

    return raw_cfg


def _require_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Sezione '{key}' mancante o non valida")
    return value


def _validate_telegram_config(telegram_cfg: Dict[str, Any]) -> TelegramConfig:
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

    return TelegramConfig(
        api_id=api_id,
        api_hash=api_hash,
        session_name=session_name,
    )


def _validate_snowball_config(snowball_cfg: Dict[str, Any]) -> SnowballConfig:
    restart = snowball_cfg.get("restart", False)
    if not isinstance(restart, bool):
        raise ValueError("snowball.restart deve essere un valore booleano")

    raw_seeds = snowball_cfg.get("seed_channels", [])
    if not isinstance(raw_seeds, list):
        raise ValueError("snowball.seed_channels deve essere una lista")
    seed_channels = [str(item).strip() for item in raw_seeds if str(item).strip()]

    filename_with_timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
    channels_output_dir =snowball_cfg.get("channels_output_dir")
    if not channels_output_dir:
        channels_output_dir = SNOWBALL_CHANNELS_OUTPUT_DIR
    channels_output_dir = Path(channels_output_dir)
    channels_output_csv = channels_output_dir / f"channels_{filename_with_timestamp}.csv"
    # check if the channels output directory exists and create it if it doesn't
    if not channels_output_dir.exists():
        channels_output_dir.mkdir(parents=True, exist_ok=True)

    relations_output_dir = snowball_cfg.get("relations_output_dir")
    if not relations_output_dir:
        relations_output_dir = SNOWBALL_RELATIONS_OUTPUT_DIR
    relations_output_dir = Path(relations_output_dir)
    if not relations_output_dir.exists():
        relations_output_dir.mkdir(parents=True, exist_ok=True)
    relations_output_csv = relations_output_dir / f"relations_{filename_with_timestamp}.csv"

    try:
        messages_per_channel = int(snowball_cfg.get("messages_per_channel", 100))
        max_depth = int(snowball_cfg.get("max_depth", 3))
        save_interval_seconds = int(snowball_cfg.get("save_interval_seconds", 30))
    except (TypeError, ValueError) as exc:
        raise ValueError("Valori snowball numerici non validi") from exc

    if messages_per_channel <= 0:
        raise ValueError("snowball.messages_per_channel deve essere > 0")
    if max_depth < 0:
        raise ValueError("snowball.max_depth deve essere >= 0")
    if save_interval_seconds <= 0:
        raise ValueError("snowball.save_interval_seconds deve essere > 0")

    return SnowballConfig(
        restart=restart,
        seed_channels=seed_channels,
        messages_per_channel=messages_per_channel,
        max_depth=max_depth,
        save_interval_seconds=save_interval_seconds,
        log_level=validate_log_level(
            snowball_cfg.get("log_level", "INFO"),
            section_name="snowball.log_level",
        ),
        channels_output_csv=channels_output_csv,
        relations_output_csv=relations_output_csv,
    )


def _validate_message_collection_config(
    message_cfg: Dict[str, Any],
    snowball_cfg: SnowballConfig,
) -> MessageCollectionConfig:
    raw_keywords = message_cfg.get("keywords", [])
    if not isinstance(raw_keywords, list):
        raise ValueError("message_collection.keywords deve essere una lista")
    keywords = [str(item).strip() for item in raw_keywords if str(item).strip()]
    if not keywords:
        # Empty keyword means "all messages" when passed to Telethon search.
        keywords = [""]

    filename_with_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    message_output_dir = message_cfg.get("output_dir")
    if not message_output_dir:
        message_output_dir = MESSAGE_OUTPUT_DIR
    message_output_dir = Path(message_output_dir)
    if not message_output_dir.exists():
        message_output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = str(message_output_dir / f"messages_{filename_with_timestamp}.csv").strip()

    channels_source_dir = message_cfg.get("channels_source_dir")
    if not channels_source_dir:
        channels_source_dir = SNOWBALL_CHANNELS_OUTPUT_DIR
    channels_source_dir = Path(channels_source_dir)
    if not channels_source_dir.exists():
        channels_source_dir.mkdir(parents=True, exist_ok=True)
    # in channels source csv a list of all csv files in channels_source_dir containing channel information will be stored
    channels_source_csv =[str(item) for item in channels_source_dir.glob("*.csv") if item.is_file()]

    channels_csv_column = str(message_cfg.get("channels_csv_column", "username")).strip()
    if not channels_csv_column:
        channels_csv_column = "username"
    elaborated_channels_csv = str(message_output_dir / f"elaborated_channels.csv").strip()

    start_date = (
        parse_iso_datetime(str(message_cfg.get("start_date")).strip(), is_end=False)
        if message_cfg.get("start_date")
        else None
    )
    end_date = (
        parse_iso_datetime(str(message_cfg.get("end_date")).strip(), is_end=True)
        if message_cfg.get("end_date")
        else None
    )
    if start_date and end_date and start_date > end_date:
        raise ValueError("message_collection.start_date non puo essere successiva a end_date")

    try:
        limit = int(message_cfg.get("limit", 100))
    except (TypeError, ValueError) as exc:
        raise ValueError("message_collection.limit deve essere un intero") from exc
    if limit <= 0:
        raise ValueError("message_collection.limit deve essere > 0")

    try:
        save_interval_seconds = int(message_cfg.get("save_interval_seconds", 30))
    except (TypeError, ValueError) as exc:
        raise ValueError("message_collection.save_interval_seconds deve essere un intero") from exc
    if save_interval_seconds <= 0:
        raise ValueError("message_collection.save_interval_seconds deve essere > 0")

    return MessageCollectionConfig(
        channels_source_csv=channels_source_csv,
        channels_csv_column=channels_csv_column,
        keywords=keywords,
        output_csv=output_csv,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        save_interval_seconds=save_interval_seconds,
        elaborated_channels_csv=elaborated_channels_csv,
        log_level=validate_log_level(
            message_cfg.get("log_level", "INFO"),
            section_name="message_collection.log_level",
        ),
    )


def _validate_graph_analysis_config(
    graph_cfg: Dict[str, Any],
    snowball_cfg: SnowballConfig,
) -> GraphAnalysisConfig:
    relations_input_dir = graph_cfg.get("relations_input_dir", SNOWBALL_RELATIONS_OUTPUT_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    relation_matrix_output_dir = graph_cfg.get("relation_matrix_output_dir")
    if not relation_matrix_output_dir:
        relation_matrix_output_dir = RELATION_MATRIX_OUTPUT_DIR
    relation_matrix_output_dir = Path(relation_matrix_output_dir)
    if not relation_matrix_output_dir.exists():
        relation_matrix_output_dir.mkdir(parents=True, exist_ok=True)
    relation_matrix_output_csv = str(relation_matrix_output_dir / f"relation_matrix_{timestamp}.csv")

    graph_output_dir = graph_cfg.get("graph_output_dir")
    if not graph_output_dir:
        graph_output_dir = GRAPH_OUTPUT_DIR
    graph_output_dir = Path(graph_output_dir)
    if not graph_output_dir.exists():
        graph_output_dir.mkdir(parents=True, exist_ok=True)
    graph_output_prefix = str(graph_output_dir / f"graph_{timestamp}")

    try:
        min_weight = int(graph_cfg.get("min_weight", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("graph_analysis.min_weight deve essere un intero") from exc

    include_unresolved_sources = graph_cfg.get("include_unresolved_sources", False)
    if not isinstance(include_unresolved_sources, bool):
        raise ValueError("graph_analysis.include_unresolved_sources deve essere booleano")

    return GraphAnalysisConfig(
        relations_input_dir=relations_input_dir,
        relation_matrix_output_csv=relation_matrix_output_csv,
        graph_output_prefix=graph_output_prefix,
        min_weight=max(1, min_weight),
        include_unresolved_sources=include_unresolved_sources,
    )


def load_project_config(config_file: str = str(CONFIG_FILE)) -> ProjectConfig:
    raw_cfg = _load_json_config(config_file)

    telegram_cfg = _require_dict(raw_cfg, "telegram")
    snowball_cfg_raw = _require_dict(raw_cfg, "snowball")
    message_cfg = _require_dict(raw_cfg, "message_collection")
    graph_cfg = _require_dict(raw_cfg, "graph_analysis")

    telegram = _validate_telegram_config(telegram_cfg)
    snowball = _validate_snowball_config(snowball_cfg_raw)
    message_collection = _validate_message_collection_config(message_cfg, snowball)
    graph_analysis = _validate_graph_analysis_config(graph_cfg, snowball)

    return ProjectConfig(
        telegram=telegram,
        message_collection=message_collection,
        snowball=snowball,
        graph_analysis=graph_analysis,
    )
