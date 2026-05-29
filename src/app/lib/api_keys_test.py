"""Tests for api_keys module."""
from __future__ import annotations

import hmac

import pytest

from ..lib.api_keys import (
    FULL_KEY_LEN,
    KEY_PREFIX,
    _hash_key,
    generate_key,
    parse_key_id,
)


class TestGenerateKey:
    def test_returns_four_values(self):
        result = generate_key()
        assert len(result) == 4

    def test_full_key_format(self):
        key_id, full_key, salt, key_hash = generate_key()
        assert full_key.startswith(KEY_PREFIX)
        assert len(full_key) == FULL_KEY_LEN

    def test_key_id_is_8_hex_chars(self):
        key_id, full_key, salt, key_hash = generate_key()
        assert len(key_id) == 8
        int(key_id, 16)  # raises ValueError if not hex

    def test_salt_is_32_hex_chars(self):
        key_id, full_key, salt, key_hash = generate_key()
        assert len(salt) == 32
        int(salt, 16)  # raises ValueError if not hex

    def test_key_hash_is_hex(self):
        key_id, full_key, salt, key_hash = generate_key()
        assert len(key_hash) == 64
        int(key_hash, 16)

    def test_hash_verifies(self):
        key_id, full_key, salt, key_hash = generate_key()
        expected = _hash_key(salt, full_key)
        assert hmac.compare_digest(expected, key_hash)

    def test_keys_are_unique(self):
        keys = [generate_key()[1] for _ in range(10)]
        assert len(set(keys)) == 10


class TestHashKey:
    def test_deterministic(self):
        assert _hash_key("aabb", "gea_key") == _hash_key("aabb", "gea_key")

    def test_different_salts_produce_different_hashes(self):
        h1 = _hash_key("aabb", "gea_key")
        h2 = _hash_key("ccdd", "gea_key")
        assert h1 != h2

    def test_different_keys_produce_different_hashes(self):
        h1 = _hash_key("aabb", "gea_key1")
        h2 = _hash_key("aabb", "gea_key2")
        assert h1 != h2


class TestParseKeyId:
    def test_returns_key_id_from_valid_key(self):
        key_id, full_key, _, _ = generate_key()
        assert parse_key_id(full_key) == key_id

    def test_returns_none_for_wrong_prefix(self):
        assert parse_key_id("bad_" + "a" * 56) is None

    def test_returns_none_for_wrong_length(self):
        assert parse_key_id("gea_tooshort") is None

    def test_returns_none_for_none(self):
        assert parse_key_id(None) is None

    def test_returns_none_for_empty_string(self):
        assert parse_key_id("") is None
