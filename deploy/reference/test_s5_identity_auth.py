"""
Fork Stage 5 tests — per-client identity & auth (server-side token→author_id).

The MCP SSE server accepts the base MNEMOSYNE_MCP_TOKEN plus any tokens
listed in MNEMOSYNE_MCP_AUTHOR_MAP (JSON {"<token>": "<author_id>"}). A
connection authenticated with a mapped token gets that author_id stamped
on every write, replacing whatever the client asserts — so provenance in
the shared store is trustworthy. The base token keeps the legacy
client-asserted identity (override None).

Run with: pytest tests/test_s5_identity_auth.py -v
"""
from __future__ import annotations

import json

import pytest

from mnemosyne.mcp_server import (
    _AUTHOR_MAP_ENV,
    _BearerTokenMiddleware,
    _parse_author_map,
)


# ---------------------------------------------------------------------------
# _parse_author_map
# ---------------------------------------------------------------------------


class TestParseAuthorMap:
    def test_unset_or_empty_yields_empty(self):
        assert _parse_author_map(None) == {}
        assert _parse_author_map("") == {}
        assert _parse_author_map("   ") == {}

    def test_invalid_json_yields_empty(self):
        assert _parse_author_map("{not json") == {}
        assert _parse_author_map("pi") == {}

    def test_non_dict_json_yields_empty(self):
        assert _parse_author_map('["a", "b"]') == {}
        assert _parse_author_map("42") == {}

    def test_valid_map_parsed(self):
        raw = json.dumps({"tok-pi": "pi", "tok-omp": "omp"})
        assert _parse_author_map(raw) == {"tok-pi": "pi", "tok-omp": "omp"}

    def test_tokens_and_authors_stripped(self):
        raw = json.dumps({"  tok-pi  ": "  pi  "})
        assert _parse_author_map(raw) == {"tok-pi": "pi"}

    def test_blank_author_dropped(self):
        raw = json.dumps({"tok-pi": "pi", "tok-empty": "", "tok-none": None})
        assert _parse_author_map(raw) == {"tok-pi": "pi"}


# ---------------------------------------------------------------------------
# _BearerTokenMiddleware: multi-token auth + author stamping
# ---------------------------------------------------------------------------


def _starlette_available() -> bool:
    try:
        import starlette  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _starlette_available(),
    reason="starlette not installed -- middleware TestClient tests skipped",
)
class TestBearerTokenMiddleware:
    def _make_app(self):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def endpoint(request):
            return JSONResponse(
                {"author": request.scope.get("_mnemosyne_author_id")}
            )

        return Starlette(routes=[Route("/", endpoint=endpoint)])

    def _client(self, token_authors):
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.testclient import TestClient

        app = Starlette(
            routes=self._make_app().routes,
            middleware=[Middleware(_BearerTokenMiddleware, token_authors=token_authors)],
        )
        return TestClient(app)

    def test_mapped_token_stamps_author(self):
        client = self._client({"base": None, "tok-pi": "pi", "tok-omp": "omp"})
        resp = client.get("/", headers={"Authorization": "Bearer tok-pi"})
        assert resp.status_code == 200
        assert resp.json() == {"author": "pi"}

        resp = client.get("/", headers={"Authorization": "Bearer tok-omp"})
        assert resp.json() == {"author": "omp"}

    def test_base_token_keeps_client_asserted_identity(self):
        """Base token maps to None: no override stamp (legacy behaviour)."""
        client = self._client({"base": None, "tok-pi": "pi"})
        resp = client.get("/", headers={"Authorization": "Bearer base"})
        assert resp.status_code == 200
        assert resp.json() == {"author": None}

    def test_missing_token_401(self):
        client = self._client({"base": None, "tok-pi": "pi"})
        resp = client.get("/")
        assert resp.status_code == 401
        assert "missing bearer token" in resp.json().get("error", "").lower()

    def test_unknown_token_401(self):
        client = self._client({"base": None, "tok-pi": "pi"})
        resp = client.get("/", headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401
        assert "invalid bearer token" in resp.json().get("error", "").lower()

    def test_malformed_header_401(self):
        client = self._client({"base": None, "tok-pi": "pi"})
        resp = client.get("/", headers={"Authorization": "Basic cGl0b2s="})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# _build_sse_app integration: mapped tokens accepted, middleware installed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _starlette_available(),
    reason="starlette/mcp not installed -- build_sse_app tests skipped",
)
class TestBuildSseAppAuthorMap:
    def test_author_map_installs_middleware_with_tokens(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv(
            _AUTHOR_MAP_ENV, json.dumps({"tok-pi": "pi", "tok-omp": "omp"})
        )
        from mnemosyne.mcp_server import _build_sse_app

        app = _build_sse_app(host="0.0.0.0")
        middleware_classes = [m.cls for m in app.user_middleware]
        names = [c.__name__ for c in middleware_classes]
        assert any("Bearer" in n for n in names)

    def test_author_map_post_accepts_mapped_token(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv(
            _AUTHOR_MAP_ENV, json.dumps({"tok-pi": "pi", "tok-omp": "omp"})
        )
        from mnemosyne.mcp_server import _build_sse_app
        from starlette.testclient import TestClient

        app = _build_sse_app(host="0.0.0.0")
        client = TestClient(app)
        # A mapped token must pass the auth gate (the /messages POST itself
        # fails later on malformed JSON-RPC, but never with 401).
        resp = client.post(
            "/messages",
            json={"ping": "pong"},
            headers={"Authorization": "Bearer tok-pi"},
        )
        assert resp.status_code != 401

    def test_author_map_post_rejects_unknown_token(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv(
            _AUTHOR_MAP_ENV, json.dumps({"tok-pi": "pi", "tok-omp": "omp"})
        )
        from mnemosyne.mcp_server import _build_sse_app
        from starlette.testclient import TestClient

        app = _build_sse_app(host="0.0.0.0")
        client = TestClient(app)
        resp = client.post(
            "/messages",
            json={"ping": "pong"},
            headers={"Authorization": "Bearer not-a-token"},
        )
        assert resp.status_code == 401
        assert "invalid bearer token" in resp.json().get("error", "").lower()

    def test_no_author_map_base_token_still_valid(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.delenv(_AUTHOR_MAP_ENV, raising=False)
        from mnemosyne.mcp_server import _build_sse_app
        from starlette.testclient import TestClient

        app = _build_sse_app(host="0.0.0.0")
        client = TestClient(app)
        resp = client.post(
            "/messages",
            json={"ping": "pong"},
            headers={"Authorization": "Bearer supersecret"},
        )
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# _create_instance identity precedence (contextvar > args > env)
# ---------------------------------------------------------------------------


class TestCreateInstanceAuthorPrecedence:
    """Server-side override beats client-asserted author_id, which beats
    the MNEMOSYNE_AUTHOR_ID env var."""

    def _patch_constructor(self, monkeypatch):
        captured = {}

        class FakeMnemosyne:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import mnemosyne.core.memory as core_memory

        monkeypatch.setattr(core_memory, "Mnemosyne", FakeMnemosyne)
        return captured

    def test_override_beats_client_arg_and_env(self, monkeypatch):
        from mnemosyne.mcp_tools import _AUTHOR_OVERRIDE, _create_instance

        captured = self._patch_constructor(monkeypatch)
        monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", "env-author")
        tok = _AUTHOR_OVERRIDE.set("pi")
        try:
            _create_instance(author_id="spoofed-client")
        finally:
            _AUTHOR_OVERRIDE.reset(tok)
        assert captured["author_id"] == "pi"

    def test_client_arg_beats_env_without_override(self, monkeypatch):
        from mnemosyne.mcp_tools import _create_instance

        captured = self._patch_constructor(monkeypatch)
        monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", "env-author")
        _create_instance(author_id="client-author")
        assert captured["author_id"] == "client-author"

    def test_env_used_when_no_override_no_arg(self, monkeypatch):
        from mnemosyne.mcp_tools import _create_instance

        captured = self._patch_constructor(monkeypatch)
        monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", "env-author")
        _create_instance()
        assert captured["author_id"] == "env-author"

    def test_override_resets_after_session(self, monkeypatch):
        """After reset the contextvar default is None — legacy behaviour."""
        from mnemosyne.mcp_tools import _AUTHOR_OVERRIDE

        tok = _AUTHOR_OVERRIDE.set("pi")
        _AUTHOR_OVERRIDE.reset(tok)
        assert _AUTHOR_OVERRIDE.get() is None
