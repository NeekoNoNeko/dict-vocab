# -*- coding: utf-8 -*-
"""Unit tests for FastAPI dictionary lookup service.

要求：
- Python 3.8+
- pytest
- pytest-mock
- httpx (for FastAPI TestClient)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from dict_vocab.api.main import app, get_dict_builder, DEFAULT_DICT_PATH


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_builder():
    """Create mock IndexBuilder."""
    builder = MagicMock()
    builder.title = "Test Dictionary"
    builder.encoding = "UTF-8"
    builder.mdx_lookup.return_value = ["definition 1", "definition 2"]
    builder.get_mdx_keys.return_value = ["word1", "word2"]
    return builder


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test /health returns ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestListDicts:
    """Test list dictionaries endpoint."""

    def test_list_dicts_empty(self, client):
        """Test /dicts returns empty when no default dict."""
        with patch("dict_vocab.api.main.DEFAULT_DICT_PATH", ""):
            with patch("dict_vocab.api.main.os.path.exists", return_value=False):
                response = client.get("/dicts")
                assert response.status_code == 200
                assert response.json() == []

    def test_list_dicts_with_default(self, client, mock_builder):
        """Test /dicts returns default dict info."""
        with patch("dict_vocab.api.main.DEFAULT_DICT_PATH", "/test/dict.mdx"):
            with patch("dict_vocab.api.main.os.path.exists", return_value=True):
                with patch(
                    "dict_vocab.api.main.get_dict_builder", return_value=mock_builder
                ):
                    response = client.get("/dicts")
                    assert response.status_code == 200
                    data = response.json()
                    assert len(data) == 1
                    assert data[0]["title"] == "Test Dictionary"
                    assert data[0]["encoding"] == "UTF-8"


class TestLookup:
    """Test lookup endpoint."""

    def test_lookup_success(self, client, mock_builder):
        """Test successful word lookup."""
        with patch("dict_vocab.api.main.os.path.exists", return_value=True):
            with patch(
                "dict_vocab.api.main.get_dict_builder", return_value=mock_builder
            ):
                response = client.post(
                    "/lookup", json={"word": "test", "dict_path": "/test/dict.mdx"}
                )
                assert response.status_code == 200
                data = response.json()
                assert data["word"] == "test"
                assert data["definitions"] == ["definition 1", "definition 2"]
                assert data["dict_title"] == "Test Dictionary"

    def test_lookup_no_dict_path(self, client):
        """Test lookup fails without dict path."""
        with patch("dict_vocab.api.main.DEFAULT_DICT_PATH", ""):
            response = client.post("/lookup", json={"word": "test"})
            assert response.status_code == 400
            assert "No dictionary path provided" in response.json()["detail"]

    def test_lookup_dict_not_found(self, client):
        """Test lookup fails when dict not found."""
        with patch("dict_vocab.api.main.os.path.exists", return_value=False):
            response = client.post(
                "/lookup", json={"word": "test", "dict_path": "/nonexistent/dict.mdx"}
            )
            assert response.status_code == 404

    def test_lookup_word_not_found(self, client, mock_builder):
        """Test lookup returns empty definitions for unknown word."""
        mock_builder.mdx_lookup.return_value = []

        with patch("dict_vocab.api.main.os.path.exists", return_value=True):
            with patch(
                "dict_vocab.api.main.get_dict_builder", return_value=mock_builder
            ):
                response = client.post(
                    "/lookup",
                    json={"word": "unknownword123", "dict_path": "/test/dict.mdx"},
                )
                assert response.status_code == 200
                data = response.json()
                assert data["definitions"] == []

    def test_lookup_ignorecase(self, client, mock_builder):
        """Test lookup with ignorecase option."""
        with patch("dict_vocab.api.main.os.path.exists", return_value=True):
            with patch(
                "dict_vocab.api.main.get_dict_builder", return_value=mock_builder
            ):
                response = client.post(
                    "/lookup",
                    json={
                        "word": "Test",
                        "dict_path": "/test/dict.mdx",
                        "ignorecase": True,
                    },
                )
                assert response.status_code == 200
                mock_builder.mdx_lookup.assert_called_with("Test", ignorecase=True)


class TestGetDictBuilder:
    """Test get_dict_builder function."""

    def test_get_dict_builder_creates_new(self, mock_builder):
        """Test get_dict_builder creates new builder."""
        with patch("dict_vocab.api.main.IndexBuilder", return_value=mock_builder):
            with patch("dict_vocab.api.main.os.path.exists", return_value=True):
                builder = get_dict_builder("/test/dict.mdx", force_rebuild=False)
                assert builder == mock_builder

    def test_get_dict_builder_uses_cache(self, mock_builder):
        """Test get_dict_builder reuses cached builder."""
        with patch("dict_vocab.api.main.IndexBuilder", return_value=mock_builder):
            with patch("dict_vocab.api.main.os.path.exists", return_value=True):
                builder1 = get_dict_builder("/test/dict.mdx", force_rebuild=False)
                builder2 = get_dict_builder("/test/dict.mdx", force_rebuild=False)
                assert builder1 is builder2

    def test_get_dict_builder_force_rebuild(self, mock_builder):
        """Test force_rebuild creates new builder."""
        with patch("dict_vocab.api.main.IndexBuilder", return_value=mock_builder):
            with patch("dict_vocab.api.main.os.path.exists", return_value=True):
                builder1 = get_dict_builder("/test/dict.mdx", force_rebuild=False)
                builder2 = get_dict_builder("/test/dict.mdx", force_rebuild=True)
                assert builder1 is not builder2

    def test_get_dict_builder_not_found(self):
        """Test get_dict_builder raises error for missing dict."""
        with patch("dict_vocab.api.main.os.path.exists", return_value=False):
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                get_dict_builder("/nonexistent/dict.mdx")
            assert exc_info.value.status_code == 404
