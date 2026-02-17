# -*- coding: utf-8 -*-
"""Unit tests for mdict_indexer.py

要求：
- Python 3.8+
- pytest
- pytest-mock
"""

import os
import sqlite3
import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

# 根据你的项目结构调整 import
# 假设 mdict_indexer 在 src/dict-vocab/ 下，且已经安装到环境中
# 可以在 pyproject.toml 里设置：
# [tool.pytest.ini_options]
# pythonpath = ["src"]
# 然后这样导入：
from dict_vocab.mdict_indexer import ExtendedMDX, ExtendedMDD, IndexBuilder


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def fake_mdx_file(tmp_path: Path) -> Path:
    """构造一个最小可用的 .mdx 文件（头部 + 一个记录块）。
    注意：这个格式非常简化，只满足 ExtendedMDX.get_index 的解析路径，
    并不是一个合法的 MDX 文件。
    """
    mdx_path = tmp_path / "test.mdx"
    # 写一个假的 MDX 文件
    with mdx_path.open("wb") as f:
        # 1. 写一个简单的文件头（让 MDX 解析时认为有一个词条）
        # 这里只是占位，具体字段含义不重要，只要 _key_list 有数据即可
        # 我们会在 mock 中覆盖真正的解析逻辑
        f.write(b"\x00" * 8)  # 假的版本号等
        f.write(b"\x00" * 8)  # 假的 record block offset

        # 2. 写一个记录块（压缩类型 + 压缩数据）
        # 块类型：无压缩（0x00000000）
        block_type = b"\x00\x00\x00\x00"
        # 块内容：随便写几个字节
        block_data = b"hello\x00world"
        compressed_size = len(block_type + block_data)
        decompressed_size = len(block_data)

        # 写记录块信息（假设版本 >= 3.0 的简单格式）
        # num_record_blocks = 1
        f.write(struct.pack(">I", 1))
        # total_size（随便写）
        f.write(struct.pack(">Q", 0))

        # compressed_size, decompressed_size
        f.write(struct.pack(">II", compressed_size, decompressed_size))

        # 写入块数据
        f.write(block_type)
        f.write(block_data)

    return mdx_path


@pytest.fixture
def fake_mdd_file(tmp_path: Path) -> Path:
    """构造一个最小化的 .mdd 文件（结构同上，仅后缀不同）。"""
    mdd_path = tmp_path / "test.mdd"
    with mdd_path.open("wb") as f:
        # 写一个简单块
        block_type = b"\x00\x00\x00\x00"
        block_data = b"dummy\x00data"
        compressed_size = len(block_type + block_data)
        decompressed_size = len(block_data)

        f.write(struct.pack(">I", 1))          # num_record_blocks
        f.write(struct.pack(">Q", 0))          # total_size
        f.write(struct.pack(">II", compressed_size, decompressed_size))
        f.write(block_type)
        f.write(block_data)

    return mdd_path


# ----------------------------------------------------------------------
# Tests for ExtendedMDX / ExtendedMDD
# ----------------------------------------------------------------------

def test_extended_mdx_get_index_basic(mocker):
    """ExtendedMDX.get_index 基本路径：返回正确结构的索引和元数据。"""
    # 模拟 MDX 父类
    MockMDX = mocker.MagicMock(name="MDX")
    MockMDX._version = 3.0
    MockMDX._encoding = "UTF-8"
    MockMDX._stylesheet = {"1": ("<b>", "</b>")}
    MockMDX.header = {
        b"Title": "TestDict".encode("utf-8"),
        b"Description": "A test dictionary".encode("utf-8"),
    }

    # 模拟 _key_list：两条记录
    MockMDX._key_list = [
        (0, "key1".encode("utf-8")),
        (10, "key2".encode("utf-8")),
    ]

    # 模拟 _decode_block：返回解压后的数据
    def fake_decode_block(block_compressed, decompressed_size):
        # block_compressed 前 4 字节是类型，后面是数据
        # 这里只验证解压后的长度是否匹配
        assert len(block_compressed) >= 4
        # 返回一个指定大小的字节串
        return b"\x00" * decompressed_size

    MockMDX._decode_block = fake_decode_block

    # 模拟文件对象
    fake_file = mocker.MagicMock(name="file")
    fake_file.tell.return_value = 123  # file_pos
    fake_file.read.side_effect = [
        # num_record_blocks
        struct.pack(">I", 1),
        # total_size
        struct.pack(">Q", 0),
        # compressed_size, decompressed_size
        struct.pack(">II", 10, 20),
        # block data (10 字节，含类型)
        b"\x00\x00\x00\x00data\x00data",
    ]
    mock_open = mocker.patch("builtins.open", return_value=fake_file)

    # 构造 ExtendedMDX 实例
    ext = ExtendedMDX.__new__(ExtendedMDX)
    ext._fname = "test.mdx"
    ext._record_block_offset = 0
    # 复制 mock 的父类属性
    for k, v in vars(MockMDX).items():
        if not k.startswith("__"):
            setattr(ext, k, v)

    # 调用 get_index
    result = ext.get_index(check_block=True)

    # 验证返回结构
    assert isinstance(result, dict)
    assert "index_dict_list" in result
    assert "meta" in result

    meta = result["meta"]
    assert meta["encoding"] == "UTF-8"
    assert meta["title"] == "TestDict"
    assert meta["description"] == "A test dictionary"
    assert meta["version"] == "1.0"

    # 验证 index_dict_list 的基本字段
    assert len(result["index_dict_list"]) == 2
    for item in result["index_dict_list"]:
        assert "key_text" in item
        assert "file_pos" in item
        assert "compressed_size" in item
        assert "decompressed_size" in item
        assert "record_block_type" in item
        assert "record_start" in item
        assert "record_end" in item
        assert "offset" in item


def test_extended_mdd_get_index_basic(mocker):
    """ExtendedMDD.get_index 基本路径：样式表应为空字符串 JSON。"""
    MockMDD = mocker.MagicMock(name="MDD")
    MockMDD._version = 3.0
    MockMDD._encoding = "UTF-8"
    MockMDD.header = {
        b"Title": "TestRes".encode("utf-8"),
        b"Description": "Test resources".encode("utf-8"),
    }
    MockMDD._key_list = [
        (0, "res1".encode("utf-8")),
    ]

    def fake_decode_block(block_compressed, decompressed_size):
        return b"\x00" * decompressed_size

    MockMDD._decode_block = fake_decode_block

    fake_file = mocker.MagicMock(name="file")
    fake_file.tell.return_value = 456
    fake_file.read.side_effect = [
        struct.pack(">I", 1),
        struct.pack(">Q", 0),
        struct.pack(">II", 10, 20),
        b"\x00\x00\x00\x00data\x00data",
    ]
    mocker.patch("builtins.open", return_value=fake_file)

    ext = ExtendedMDD.__new__(ExtendedMDD)
    ext._fname = "test.mdd"
    ext._record_block_offset = 0
    for k, v in vars(MockMDD).items():
        if not k.startswith("__"):
            setattr(ext, k, v)

    result = ext.get_index(check_block=True)

    assert isinstance(result, dict)
    assert "index_dict_list" in result
    assert "meta" in result

    meta = result["meta"]
    assert meta["encoding"] == "UTF-8"
    assert meta["stylesheet"] == "{}"
    assert meta["version"] == "1.0"


# ----------------------------------------------------------------------
# Tests for IndexBuilder
# ----------------------------------------------------------------------

def test_index_builder_build_mdx_index(tmp_path, fake_mdx_file, mocker):
    """IndexBuilder 能够为 MDX 创建 SQLite 索引并写入元数据。"""
    db_path = fake_mdx_file.with_suffix(".mdx.db")

    # 模拟 ExtendedMDX 和 get_index，避免依赖真实的 MDX 解析
    MockExtMDX = mocker.MagicMock(name="ExtendedMDX")
    MockExtMDX._version = 3.0
    MockExtMDX._encoding = "UTF-8"
    MockExtMDX._stylesheet = {"1": ("<b>", "</b>")}
    MockExtMDX.header = {
        b"Title": "TestDict".encode("utf-8"),
        b"Description": "A test dictionary".encode("utf-8"),
    }

    index_dict_list = [
        {
            "key_text": "key1",
            "file_pos": 0,
            "compressed_size": 10,
            "decompressed_size": 20,
            "record_block_type": 0,
            "record_start": 0,
            "record_end": 10,
            "offset": 0,
        },
        {
            "key_text": "key2",
            "file_pos": 0,
            "compressed_size": 10,
            "decompressed_size": 20,
            "record_block_type": 0,
            "record_start": 10,
            "record_end": 20,
            "offset": 0,
        },
    ]

    meta = {
        "encoding": "UTF-8",
        "stylesheet": json.dumps(MockExtMDX._stylesheet, ensure_ascii=False),
        "title": "TestDict",
        "description": "A test dictionary",
        "version": "1.0",
    }

    MockExtMDX.get_index.return_value = {
        "index_dict_list": index_dict_list,
        "meta": meta,
    }

    # 使用 patch 替换 ExtendedMDX
    mocker.patch("dict_vocab.mdict_indexer.ExtendedMDX", return_value=MockExtMDX)

    # 强制重建索引
    builder = IndexBuilder(
        fname=str(fake_mdx_file),
        encoding="",
        passcode=None,
        force_rebuild=True,
        sql_index=True,
        check=False,
    )

    # 验证数据库文件已创建
    assert db_path.exists()

    # 验证数据库结构
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 表存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='MDX_INDEX'")
    assert cursor.fetchone() is not None

    # META 表存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='META'")
    assert cursor.fetchone() is not None

    # 索引存在
    cursor.execute("PRAGMA index_list('MDX_INDEX')")
    indexes = [row[1] for row in cursor.fetchall()]
    assert any("key_index" in name for name in indexes)

    # 验证数据条数
    cursor.execute("SELECT COUNT(*) FROM MDX_INDEX")
    assert cursor.fetchone()[0] == len(index_dict_list)

    # 验证元数据
    cursor.execute("SELECT key, value FROM META")
    meta_from_db = {row[0]: row[1] for row in cursor.fetchall()}
    assert meta_from_db["encoding"] == meta["encoding"]
    assert meta_from_db["title"] == meta["title"]
    assert meta_from_db["description"] == meta["description"]
    assert meta_from_db["version"] == meta["version"]

    conn.close()


def test_index_builder_mdx_lookup(tmp_path, fake_mdx_file, mocker):
    """IndexBuilder.mdx_lookup 能够根据索引返回查询结果。"""
    db_path = fake_mdx_file.with_suffix(".mdx.db")

    # 预先写好一个索引数据库
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE MDX_INDEX ("
        "key_text TEXT NOT NULL,"
        "file_pos INTEGER,"
        "compressed_size INTEGER,"
        "decompressed_size INTEGER,"
        "record_block_type INTEGER,"
        "record_start INTEGER,"
        "record_end INTEGER,"
        "offset INTEGER"
        ")"
    )
    cursor.execute(
        "CREATE TABLE META (key TEXT, value TEXT)"
    )

    index_dict_list = [
        {
            "key_text": "key1",
            "file_pos": 0,
            "compressed_size": 10,
            "decompressed_size": 20,
            "record_block_type": 0,
            "record_start": 0,
            "record_end": 10,
            "offset": 0,
        },
    ]

    tuples = [
        (
            item["key_text"],
            item["file_pos"],
            item["compressed_size"],
            item["decompressed_size"],
            item["record_block_type"],
            item["record_start"],
            item["record_end"],
            item["offset"],
        )
        for item in index_dict_list
    ]
    cursor.executemany("INSERT INTO MDX_INDEX VALUES (?,?,?,?,?,?,?,?)", tuples)

    meta = {
        "encoding": "UTF-8",
        "stylesheet": "{}",
        "title": "TestDict",
        "description": "A test dictionary",
        "version": "1.0",
    }
    cursor.executemany(
        "INSERT INTO META VALUES (?,?)",
        [(k, v) for k, v in meta.items()],
    )
    conn.commit()
    conn.close()

    # 模拟 _extract_data，返回固定文本
    def fake_extract_data(mdict_obj, index):
        return "definition of key1".encode("utf-8")

    # 模拟 ExtendedMDX，避免真的打开文件
    MockExtMDX = mocker.MagicMock(name="ExtendedMDX")
    MockExtMDX._fname = str(fake_mdx_file)
    MockExtMDX._encoding = "UTF-8"

    builder = IndexBuilder.__new__(IndexBuilder)
    builder._mdx_file = str(fake_mdx_file)
    builder._mdx_db = str(db_path)
    builder._mdd_file = None
    builder._mdd_db = None
    builder._mdx_obj = MockExtMDX
    builder._mdd_obj = None
    builder._encoding = meta["encoding"]
    builder._stylesheet = json.loads(meta["stylesheet"])
    builder._title = meta["title"]
    builder._description = meta["description"]
    builder._passcode = None
    builder._sql_index = True
    builder._check = False

    # 替换 _extract_data
    builder._extract_data = fake_extract_data

    # 查询存在的 key
    results = builder.mdx_lookup("key1", ignorecase=False)
    assert len(results) == 1
    assert results[0] == "definition of key1"

    # 查询不存在的 key
    empty_results = builder.mdx_lookup("not_exist", ignorecase=False)
    assert empty_results == []


def test_index_builder_get_mdx_keys(tmp_path, fake_mdx_file, mocker):
    """IndexBuilder.get_mdx_keys 能够返回所有键，并支持通配符查询。"""
    db_path = fake_mdx_file.with_suffix(".mdx.db")

    # 预先写好一个索引数据库
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE MDX_INDEX (key_text TEXT NOT NULL)"
    )
    keys = ["key1", "key2", "other"]
    cursor.executemany("INSERT INTO MDX_INDEX VALUES (?)", [(k,) for k in keys])
    conn.commit()
    conn.close()

    builder = IndexBuilder.__new__(IndexBuilder)
    builder._mdx_file = str(fake_mdx_file)
    builder._mdx_db = str(db_path)
    builder._mdd_file = None
    builder._mdd_db = None
    builder._mdx_obj = None
    builder._mdd_obj = None
    builder._encoding = "UTF-8"
    builder._stylesheet = {}
    builder._title = "Test"
    builder._description = ""
    builder._passcode = None
    builder._sql_index = True
    builder._check = False

    # 获取所有键
    all_keys = builder.get_mdx_keys()
    assert set(all_keys) == set(keys)

    # 通配符查询
    some_keys = builder.get_mdx_keys(pattern="key*")
    assert set(some_keys) == {"key1", "key2"}


# ----------------------------------------------------------------------
# Integration-style test (optional)
# ----------------------------------------------------------------------

def test_index_builder_with_mdd(tmp_path, fake_mdx_file, fake_mdd_file, mocker):
    """当存在 .mdd 时，IndexBuilder 应同时构建 MDX 和 MDD 索引。"""
    mdx_db_path = fake_mdx_file.with_suffix(".mdx.db")
    mdd_db_path = fake_mdd_file.with_suffix(".mdd.db")

    # 模拟 ExtendedMDX
    MockExtMDX = mocker.MagicMock(name="ExtendedMDX")
    MockExtMDX._version = 3.0
    MockExtMDX._encoding = "UTF-8"
    MockExtMDX._stylesheet = {}
    MockExtMDX.header = {
        b"Title": "TestDict".encode("utf-8"),
        b"Description": "".encode("utf-8"),
    }
    MockExtMDX.get_index.return_value = {
        "index_dict_list": [],
        "meta": {
            "encoding": "UTF-8",
            "stylesheet": "{}",
            "title": "TestDict",
            "description": "",
            "version": "1.0",
        },
    }

    # 模拟 ExtendedMDD
    MockExtMDD = mocker.MagicMock(name="ExtendedMDD")
    MockExtMDD._version = 3.0
    MockExtMDD._encoding = "UTF-8"
    MockExtMDD.header = {
        b"Title": "TestRes".encode("utf-8"),
        b"Description": "".encode("utf-8"),
    }
    MockExtMDD.get_index.return_value = {
        "index_dict_list": [],
        "meta": {
            "encoding": "UTF-8",
            "stylesheet": "{}",
            "title": "TestRes",
            "description": "",
            "version": "1.0",
        },
    }

    mocker.patch("dict_vocab.mdict_indexer.ExtendedMDX", return_value=MockExtMDX)
    mocker.patch("dict_vocab.mdict_indexer.ExtendedMDD", return_value=MockExtMDD)

    builder = IndexBuilder(
        fname=str(fake_mdx_file),
        encoding="",
        passcode=None,
        force_rebuild=True,
        sql_index=True,
        check=False,
    )

    # 验证两个数据库都被创建
    assert mdx_db_path.exists()
    assert mdd_db_path.exists()

    # 验证 MDD 索引表结构
    conn = sqlite3.connect(str(mdd_db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='MDX_INDEX'")
    assert cursor.fetchone() is not None
    conn.close()
