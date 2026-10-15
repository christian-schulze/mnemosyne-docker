#!/usr/bin/env python3
"""Stage-5 verification — server-side token→author_id mapping (fork).

Proves the exit criterion: writes through a mapped token are stamped with
the server-side author_id in recall results, regardless of what the client
asserts; the base token keeps the legacy client-asserted identity; unknown
tokens are still rejected.

Two modes:
  --fresh  (default) spin a throwaway container from --image with a temp
           data dir, two mapped tokens + base token, probe it, clean up.
  --live   probe the already-running compose container at 127.0.0.1:8080
           using the real tokens from ~/docker/mnemosyne/.env.

Run with the host venv python (mcp<2 pinned, like probe_sse.py).

Usage:
  python3 deploy/verify_identity_auth.py [--fresh|--live] [--image IMAGE]
"""

import argparse
import asyncio
import json
import os
import random
import shutil
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_ENV = Path.home() / "docker" / "mnemosyne" / ".env"
LIVE_URL = "http://127.0.0.1:8080/sse"

FAILURES = []


def _rand(n: int = 12) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _read_live_tokens():
    """Parse MNEMOSYNE_MCP_TOKEN and MNEMOSYNE_MCP_AUTHOR_MAP from .env."""
    if not LIVE_ENV.is_file():
        raise SystemExit(f"--live needs {LIVE_ENV} (compose .env) — not found")
    env = {}
    for line in LIVE_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip("'\"")
    base = env.get("MNEMOSYNE_MCP_TOKEN")
    raw_map = env.get("MNEMOSYNE_MCP_AUTHOR_MAP")
    if not base:
        raise SystemExit(f"{LIVE_ENV} has no MNEMOSYNE_MCP_TOKEN")
    author_map = {}
    if raw_map:
        author_map = json.loads(raw_map)
    return base, author_map


def _text_of(content) -> str:
    """Extract text from an MCP CallToolResult content block (TextContent)."""
    if not content:
        return ""
    return getattr(content[0], "text", "") or ""


async def _probe_remember(url, token, label, expected_author, marker, batch=False):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    headers = {"Authorization": f"Bearer {token}"}
    async with sse_client(url, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if batch:
                res = await session.call_tool(
                    "mnemosyne_batch",
                    {
                        "operations": [
                            {"action": "remember", "content": f"stage5-batch {marker}", "scope": "global"}
                        ],
                        "author_id": "spoofed-client",  # must be ignored for mapped tokens
                    },
                )
            else:
                res = await session.call_tool(
                    "mnemosyne_remember",
                    {
                        "content": f"stage5-remember {marker}",
                        "scope": "global",
                        "author_id": "spoofed-client",  # must be ignored for mapped tokens
                    },
                )
            text = _text_of(res.content)
            payload = json.loads(text)
            assert payload.get("status") in ("stored", "ok"), f"{label}: write failed: {text[:200]}"

            res2 = await session.call_tool(
                "mnemosyne_recall", {"query": marker, "limit": 5}
            )
            text2 = _text_of(res2.content)
            payload2 = json.loads(text2)
            authors = {r.get("author_id") for r in payload2.get("results", [])}
            if expected_author is not None:
                assert expected_author in authors, (
                    f"{label}: expected author_id={expected_author!r} in recall, got {authors}"
                )
            else:
                assert "spoofed-client" in authors, (
                    f"{label}: base token should keep client-asserted author_id, got {authors}"
                )
            print(f"  PASS {label}: recall author_ids={sorted(authors)} (want {expected_author!r})")


async def _probe_rejects(url):
    """Unknown token must be rejected (401 on the SSE GET)."""
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    headers = {"Authorization": "Bearer definitely-not-a-valid-token"}
    try:
        async with sse_client(url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
        raise AssertionError("unknown token was accepted")
    except Exception as exc:  # noqa: BLE001 - 401 surfaces as HTTPStatusError, possibly in an ExceptionGroup
        if _is_401(exc):
            print("  PASS reject-unknown: 401")
        else:
            raise AssertionError(f"unknown token: unexpected failure: {exc!r}") from exc


def _is_401(exc: BaseException) -> bool:
    """True if exc (or any nested exception) is an HTTPStatusError 401."""
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 401
    # ExceptionGroup / BaseExceptionGroup (py3.11+) — duck-typed so the
    # script also runs on older pythons.
    nested = getattr(exc, "exceptions", None)
    if isinstance(nested, (list, tuple)):
        return any(_is_401(e) for e in nested)
    return False


async def run_checks(url, base, author_map, prefix):
    marker = _rand(6)
    print(f"[{prefix}] identity checks against {url}")

    # 1/2. mapped tokens stamp server-side author, beating spoofed client
    for token, author in author_map.items():
        await _probe_remember(url, token, f"{author}->{author}", author, f"{marker}-{author}", batch=False)

    # 3. override flows through the batch write path too
    first_author = next(iter(author_map.values())) if author_map else None
    if first_author and author_map:
        await _probe_remember(url, next(iter(author_map)), f"{first_author}-batch", first_author, f"{marker}-batch", batch=True)

    # 4. base token keeps client-asserted identity (legacy)
    await _probe_remember(url, base, "base-token-legacy", None, f"{marker}-base", batch=False)

    # 5. unknown token rejected
    await _probe_rejects(url)

    # 6. full-store property: a different author's session still SEES the
    #    mapped author's memories (author is provenance, not isolation)
    if len(author_map) >= 2:
        token_b, author_b = list(author_map.items())[1]
        token_a, author_a = list(author_map.items())[0]
        other_marker = f"{marker}-{author_a}"

        async def _cross_recall(session):
            res = await session.call_tool("mnemosyne_recall", {"query": other_marker, "limit": 5})
            payload = json.loads(_text_of(res.content))
            authors = {r.get("author_id") for r in payload.get("results", [])}
            assert author_a in authors, (
                f"cross-author: {author_b} session must see {author_a}'s memory "
                f"({other_marker}); recall authors={authors}"
            )

        from mcp import ClientSession
        from mcp.client.sse import sse_client

        headers = {"Authorization": f"Bearer {token_b}"}
        async with sse_client(url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await _cross_recall(session)
        print(f"  PASS cross-author: {author_b} session recalls {author_a}'s memory (full store)")


def run_fresh(image: str) -> int:
    data_dir = Path(tempfile.mkdtemp(prefix="stage5-"))
    os.chmod(data_dir, 0o777)  # container runs as uid 1000; avoid root-owned bind
    name = f"mnemo-stage5-{_rand(4)}"
    base = "base-" + _rand(16)
    pi = "pi-" + _rand(24)
    omp = "omp-" + _rand(24)
    author_map = {pi: "pi", omp: "omp"}
    port = None
    try:
        # random host port to avoid clashing with the live container on 8080
        cmd = [
            "docker", "run", "-d", "--name", name,
            "-v", f"{data_dir}:/data",
            "-p", "127.0.0.1::8080",
            "-e", f"MNEMOSYNE_MCP_TOKEN={base}",
            "-e", f"MNEMOSYNE_MCP_AUTHOR_MAP={json.dumps(author_map)}",
            "-e", "MNEMOSYNE_DATA_DIR=/data",
            image,
            "--transport", "sse", "--host", "0.0.0.0", "--port", "8080",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            raise SystemExit(f"docker run failed: {out.stderr[-500:]}")
        port = subprocess.run(
            ["docker", "port", name, "8080/tcp"], capture_output=True, text=True
        ).stdout.strip().rsplit(":", 1)[-1]
        url = f"http://127.0.0.1:{port}/sse"

        # wait for readiness (healthcheck is in-process recall)
        for _ in range(60):
            st = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Health.Status}}", name],
                capture_output=True, text=True,
            ).stdout.strip()
            if st == "healthy":
                break
            time.sleep(1)
        else:
            raise SystemExit("fresh container never became healthy")

        asyncio.run(run_checks(url, base, author_map, "fresh"))
        return 0 if not FAILURES else 1
    finally:
        if name:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
        shutil.rmtree(data_dir, ignore_errors=True)


def run_live() -> int:
    base, author_map = _read_live_tokens()
    if not author_map:
        raise SystemExit(
            f"{LIVE_ENV} has no MNEMOSYNE_MCP_AUTHOR_MAP — Stage 5 not deployed yet"
        )
    asyncio.run(run_checks(LIVE_URL, base, author_map, "live"))
    return 0 if not FAILURES else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="ghcr.io/christian-schulze/mnemosyne-docker:latest")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true", help="throwaway container (default)")
    mode.add_argument("--live", action="store_true", help="probe running compose container")
    args = ap.parse_args()

    return run_live() if args.live else run_fresh(args.image)


if __name__ == "__main__":
    sys.exit(main())
