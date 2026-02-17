# -*- coding: utf-8 -*-
"""Integration tests for mdict_indexer with real dictionary files.

要求：
- Python 3.8+
- pytest
- pytest-mock
"""

import os
import sqlite3
import json
from pathlib import Path

import pytest

from dict_vocab.indexer.mdict_indexer import IndexBuilder, ExtendedMDX, ExtendedMDD


RESOURCE_DIR = Path(__file__).parent.parent / "resource" / "cobuild2024"
MDX_FILE = RESOURCE_DIR / "cobuild2024.mdx"
MDD_FILE = RESOURCE_DIR / "cobuild2024.mdd"


@pytest.fixture
def mdx_file():
    """Return path to MDX file."""
    if not MDX_FILE.exists():
        pytest.skip(f"MDX file not found: {MDX_FILE}")
    return MDX_FILE


@pytest.fixture
def mdd_file():
    """Return path to MDD file."""
    if not MDD_FILE.exists():
        pytest.skip(f"MDD file not found: {MDD_FILE}")
    return MDD_FILE


@pytest.fixture
def index_builder(mdx_file, tmp_path):
    """Create IndexBuilder with force rebuild."""
    db_path = mdx_file.with_suffix(".mdx.db")
    if db_path.exists():
        db_path.unlink()
    
    builder = IndexBuilder(
        fname=str(mdx_file),
        encoding="",
        passcode=None,
        force_rebuild=True,
        sql_index=True,
        check=False,
    )
    yield builder
    
    if db_path.exists():
        db_path.unlink()


class TestIndexBuilderWithRealDict:
    """Test IndexBuilder with real dictionary files."""

    def test_build_mdx_index(self, mdx_file):
        """Test building MDX index with SQLite."""
        db_path = mdx_file.with_suffix(".mdx.db")
        
        if db_path.exists():
            db_path.unlink()
        
        builder = IndexBuilder(
            fname=str(mdx_file),
            force_rebuild=True,
            sql_index=True,
            check=False,
        )
        
        assert db_path.exists(), "SQLite database should be created"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "MDX_INDEX" in tables, "MDX_INDEX table should exist"
        assert "META" in tables, "META table should exist"
        
        cursor.execute("SELECT COUNT(*) FROM MDX_INDEX")
        count = cursor.fetchone()[0]
        assert count > 0, "Should have indexed entries"
        
        cursor.execute("SELECT key, value FROM META")
        meta = {row[0]: row[1] for row in cursor.fetchall()}
        assert "encoding" in meta
        assert "title" in meta
        
        conn.close()
        
        if db_path.exists():
            db_path.unlink()

    def test_mdx_lookup(self, index_builder):
        """Test looking up words in MDX."""
        keys = index_builder.get_mdx_keys()
        assert len(keys) > 0, "Should have keys"
        
        if keys:
            first_key = keys[0]
            results = index_builder.mdx_lookup(first_key)
            assert len(results) > 0, f"Should find definition for '{first_key}'"
            assert isinstance(results[0], str)

    def test_mdx_lookup_with_pattern(self, index_builder):
        """Test wildcard pattern matching."""
        keys = index_builder.get_mdx_keys(pattern="a*")
        assert len(keys) > 0, "Should match pattern"

    def test_get_all_keys(self, index_builder):
        """Test getting all keys."""
        keys = index_builder.get_mdx_keys()
        assert isinstance(keys, list)
        assert len(keys) > 0

    def test_mdd_index(self, mdd_file):
        """Test building MDD index."""
        db_path = mdd_file.with_suffix(".mdd.db")
        
        if db_path.exists():
            db_path.unlink()
        
        builder = IndexBuilder(
            fname=str(mdd_file.with_suffix(".mdx")),
            force_rebuild=True,
            sql_index=True,
            check=False,
        )
        
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM MDX_INDEX")
            count = cursor.fetchone()[0]
            conn.close()
            
            if db_path.exists():
                db_path.unlink()


class TestExtendedMDX:
    """Test ExtendedMDX class directly."""

    def test_get_index(self, mdx_file):
        """Test ExtendedMDX.get_index()."""
        mdx = ExtendedMDX(str(mdx_file))
        result = mdx.get_index(check_block=False)
        
        assert "index_dict_list" in result
        assert "meta" in result
        assert len(result["index_dict_list"]) > 0
        
        meta = result["meta"]
        assert "encoding" in meta
        assert "title" in meta


class TestSQLiteOptimization:
    """Test SQLite query optimization."""

    def test_index_exists(self, index_builder):
        """Test that SQL index is created."""
        db_path = Path(index_builder._mdx_db)
        assert db_path.exists()
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA index_list('MDX_INDEX')")
        indexes = cursor.fetchall()
        conn.close()
        
        assert len(indexes) > 0, "Should have indexes for optimization"

    def test_query_performance(self, index_builder):
        """Test query uses index."""
        db_path = index_builder._mdx_db
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM MDX_INDEX WHERE key_text = 'test'")
        plan = cursor.fetchall()
        
        conn.close()
        
        plan_text = str(plan)
        assert "INDEX" in plan_text or "key_index" in plan_text, "Query should use index"
