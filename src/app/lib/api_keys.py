"""API key generation, verification, and Firestore CRUD.

Key format: gea_<8-char key_id><48-char secret>
  - key_id  : 4 random bytes as hex (8 chars)
  - secret  : 24 random bytes as hex (48 chars)
  - total   : 60 chars including "gea_" prefix

Storage: only the salted SHA-256 hash is written to Firestore.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone

from google.cloud.firestore import AsyncClient  # type: ignore[import-untyped]
from google.cloud.firestore import Increment  # type: ignore[import-untyped]

from ..documents import ApiKeyDocument

logger = logging.getLogger(__name__)

KEY_PREFIX = "gea_"
KEY_ID_LEN = 8
SECRET_LEN = 48
FULL_KEY_LEN = len(KEY_PREFIX) + KEY_ID_LEN + SECRET_LEN  # 60

API_KEYS_COLLECTION = "api_keys"


# ---------------------------------------------------------------------------
# Pure functions (no I/O)
# ---------------------------------------------------------------------------


def generate_key() -> tuple[str, str, str, str]:
    """Return (key_id, full_key, salt, key_hash). Call once; discard full_key after display."""
    key_id = os.urandom(4).hex()
    secret = os.urandom(24).hex()
    full_key = f"{KEY_PREFIX}{key_id}{secret}"
    salt = os.urandom(16).hex()
    key_hash = _hash_key(salt, full_key)
    return key_id, full_key, salt, key_hash


def _hash_key(salt: str, full_key: str) -> str:
    return hashlib.sha256(bytes.fromhex(salt) + full_key.encode()).hexdigest()


def parse_key_id(full_key: str | None) -> str | None:
    """Extract key_id from a full API key string, or None if the format is invalid."""
    if not full_key:
        return None
    if not full_key.startswith(KEY_PREFIX):
        return None
    if len(full_key) != FULL_KEY_LEN:
        return None
    return full_key[len(KEY_PREFIX): len(KEY_PREFIX) + KEY_ID_LEN]
