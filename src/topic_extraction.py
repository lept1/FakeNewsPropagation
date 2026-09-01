# Topic extraction using the Turftopic library
# Documentation: https://x-tabdeveloping.github.io/turftopic/

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

from turftopic import KeyNMF
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import string
import nltk
import re

from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent

DATA_DIR = PARENT_DIR / "data_collected"
VAR_DIR = PARENT_DIR / "var"
TOPIC_DIR = PARENT_DIR / "topic_analysis"

TOPIC_MODEL_FILE = TOPIC_DIR / "topic_model.pkl"
INPUT_DATA = DATA_DIR / "telegram_fakenews_analysis.csv"
INPUT_ENV = VAR_DIR / "config.env"

load_dotenv(INPUT_ENV)

df = pd.read_csv(INPUT_DATA)

df = df.dropna(subset=["text"])
df = df.drop_duplicates(subset=["text"])
df = df.dropna(subset=["date"])

corpus = df["text"].tolist()
# remove URLs
corpus = [doc for doc in corpus if isinstance(doc, str) and doc.strip() != ""]
corpus = [re.sub(r"http\S+|www\S+|https\S+", "", doc, flags=re.MULTILINE) for doc in corpus]

timestamps = pd.to_datetime(df["date"]).tolist()

nltk.download('stopwords')
stop_words = list(nltk.corpus.stopwords.words("italian"))
model = KeyNMF(5, top_n=5, random_state=42, vectorizer=CountVectorizer(stop_words=stop_words, min_df=5, ngram_range=(1,3)))
document_topic_matrix = model.fit_transform_dynamic(
    corpus, timestamps=timestamps, bins=10
)

model.print_topics()

model.print_topics_over_time()

print(model.top_documents)

print(model)