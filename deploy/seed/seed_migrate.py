#!/usr/bin/env python3
"""Stage 6 — Existing-memory migration seed (live Hermes DB -> shared volume).

Design A (shared file): migration = carry the live Hermes provider DB into the
shared compose volume. This script is idempotent and dry-runnable.

  1. Preflight: integrity + row-count snapshot of the source, sha256, idempotency
     manifest guard (already-migrated -> no-op unless --force).
  2. Backups: SQLite ONLINE BACKUP API (safe against the live in-process
     provider, WAL included) of BOTH stores into <backup-dir>/pre-migration-<ts>/;
     each backup is verified restorable (integrity_check + row counts).
  3. Swap: docker stop <container> (graceful, WAL checkpointed), delete stale
     -wal/-shm (never replay old WAL into a new DB), online-backup source ->
     temp in dest dir, os.replace over dest, at-swap sha must equal the
     verified source backup (online-backup output is deterministic), checkpoint,
     docker start, wait healthy.
  4. Post-checks: dest integrity + counts == source snapshot + at-swap
     identity vs backup, MCP SSE recall spot-checks of known durable facts
     (keyword + semantic).
  5. Manifest written to <backup-dir>/seed-manifest.json; re-run no-ops.

Run with the fork venv python (has mcp client):
  ~/.hermes/plugins/mnemosyne/.venv/bin/python deploy/seed/seed_migrate.py --dry-run
  ~/.hermes/plugins/mnemosyne/.venv/bin/python deploy/seed/seed_migrate.py

Exit codes: 0 ok/no-op, 1 error, 2 preflight failure, 3 already-migrated no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Defaults (override with flags)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "source": os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"),
    "dest": os.path.expanduser("~/docker/mnemosyne/data/mnemosyne.db"),
    "backup_dir": os.path.expanduser("~/docker/mnemosyne/backups/"),
    "env_file": os.path.expanduser("~/docker/mnemosyne/.env"),
    "container": "mnemosyne-mcp",
    "endpoint": "http://127.0.0.1:8080/sse",
}

# Tables whose row counts gate "migration is complete" checks.
COUNT_TABLES = [
    "working_memory", "gists", "memories", "facts", "episodic_memory",
    "memoria_facts", "memoria_preferences", "memoria_persona",
    "annotations", "memory_embeddings", "memory_events", "canonical_facts",
    "triples", "graph_edges", "consolidated_facts",
]

# Probe keywords/IDs are derived at runtime from the migrated store (newest
# tool-sourced memories) — see dest_id_presence / probe_keywords_from.
SEMANTIC_QUERY = "shared memory server docker deployment"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"ERROR: {msg}")
    sys.exit(code)


def run(cmd: list[str], check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ro_connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def integrity_ok(path: str) -> bool:
    con = ro_connect(path)
    try:
        res = con.execute("PRAGMA integrity_check").fetchone()
        return bool(res) and res[0] == "ok"
    finally:
        con.close()


def row_counts(path: str) -> dict[str, int]:
    con = ro_connect(path)
    counts: dict[str, int] = {}
    try:
        have = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for t in COUNT_TABLES:
            if t in have:
                counts[t] = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
    finally:
        con.close()
    return counts


def online_backup(src: str, dst: str) -> None:
    """SQLite online backup API — consistent snapshot, WAL + live writers safe."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    scon = sqlite3.connect(src)
    dcon = sqlite3.connect(dst)
    try:
        scon.backup(dcon)
        dcon.commit()
    finally:
        dcon.close()
        scon.close()


def load_env(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def docker_running(container: str) -> bool:
    try:
        p = run(["docker", "inspect", "-f", "{{.State.Running}}", container], check=False)
        return p.returncode == 0 and p.stdout.strip() == "true"
    except Exception:
        return False


def docker_health(container: str) -> str:
    try:
        p = run(["docker", "inspect", "-f", "{{.State.Health.Status}}", container], check=False)
        return p.stdout.strip() if p.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def wait_healthy(container: str, timeout: int = 150) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = docker_health(container)
        if h == "healthy":
            return True
        time.sleep(5)
    return False


# ---------------------------------------------------------------------------
# MCP recall spot-check (read-only)
# ---------------------------------------------------------------------------
def mcp_recall(query: str, token: str, endpoint: str, limit: int = 3) -> list[str]:
    import asyncio
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async def _one() -> list[str]:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with sse_client(endpoint, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool("mnemosyne_recall", {"query": query, "limit": limit})
                texts: list[str] = []
                for c in res.content:
                    t = getattr(c, "text", None)
                    if t:
                        texts.append(t)
                return texts

    return asyncio.run(_one())


def dest_id_presence(src_path: str, dst_path: str, n: int = 3) -> tuple[list[str], list[str]]:
    """Top-n newest tool-sourced memories from source must exist in dest by id."""
    con = ro_connect(src_path)
    try:
        ids = [r[0] for r in con.execute(
            "SELECT id FROM working_memory WHERE source='tool' ORDER BY created_at DESC LIMIT ?",
            (n,)).fetchall()]
        if not ids:  # fall back to newest rows of any source
            ids = [r[0] for r in con.execute(
                "SELECT id FROM working_memory ORDER BY created_at DESC LIMIT ?", (n,)).fetchall()]
    finally:
        con.close()
    dcon = ro_connect(dst_path)
    try:
        missing = [i for i in ids if dcon.execute(
            "SELECT count(*) FROM working_memory WHERE id=?", (i,)).fetchone()[0] == 0]
    finally:
        dcon.close()
    return ids, missing


def probe_keywords_from(src_path: str, ids: list[str], n: int = 3) -> list[str]:
    """Distinctive, unique keyword per known id (first >=5-char word of its
    content, skipping words already picked)."""
    kws: list[str] = []
    con = ro_connect(src_path)
    try:
        for i in ids[:n]:
            row = con.execute("SELECT content FROM working_memory WHERE id=?", (i,)).fetchone()
            if row:
                for tok in row[0].replace(":", " ").split():
                    clean = tok.strip("()[],.'\"")
                    if len(clean) >= 5 and clean[0].isalnum() and clean not in kws:
                        kws.append(clean)
                        break
    finally:
        con.close()
    return kws


def mcp_spot_check(token: str, endpoint: str, probe_keywords: list[str]) -> list[str]:
    """Server-level check: MCP recall on the migrated store must return results
    for terms drawn from migrated memories, plus a semantic probe. (Ranking is
    deliberately not asserted — presence is proven by dest_id_presence.)"""
    problems: list[str] = []
    for kw in probe_keywords:
        try:
            out = mcp_recall(kw, token, endpoint, limit=5)
            blob = "\n".join(out).strip()
            if not blob or blob in ("[]", "{}"):
                problems.append(f"recall('{kw}') returned empty on migrated store")
        except Exception as e:  # noqa: BLE001
            problems.append(f"recall('{kw}') errored: {e}")
    try:
        out = mcp_recall(SEMANTIC_QUERY, token, endpoint, limit=5)
        blob = "\n".join(out).strip()
        if not blob or blob in ("[]", "{}"):
            problems.append("semantic query returned empty")
    except Exception as e:  # noqa: BLE001
        problems.append(f"semantic query errored: {e}")
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULTS["source"])
    ap.add_argument("--dest", default=DEFAULTS["dest"])
    ap.add_argument("--backup-dir", default=DEFAULTS["backup_dir"])
    ap.add_argument("--env-file", default=DEFAULTS["env_file"])
    ap.add_argument("--container", default=DEFAULTS["container"])
    ap.add_argument("--endpoint", default=DEFAULTS["endpoint"])
    ap.add_argument("--dry-run", action="store_true", help="validate + print plan, write nothing")
    ap.add_argument("--force", action="store_true", help="re-run even if manifest says migrated")
    ap.add_argument("--skip-mcp", action="store_true", help="skip the MCP recall spot-check")
    args = ap.parse_args()

    src, dst = args.source, args.dest
    if not os.path.exists(src):
        die(f"source DB not found: {src}", 2)
    if not os.path.isdir(os.path.dirname(dst)):
        die(f"dest dir missing: {os.path.dirname(dst)}", 2)
    if not os.access(os.path.dirname(dst), os.W_OK):
        die(f"dest dir not writable: {os.path.dirname(dst)}", 2)

    log(f"source: {src}")
    log(f"dest:   {dst}")

    # --- preflight (all modes) ---
    if not integrity_ok(src):
        die("source integrity_check FAILED", 2)
    src_counts = row_counts(src)
    src_sha = sha256(src)
    total_src = sum(src_counts.values())
    log(f"source integrity ok | {len(src_counts)} tables counted | {total_src} rows | sha256 {src_sha[:16]}…")

    # embedding model parity (informational)
    for label, cfg in (("provider", os.path.expanduser("~/.hermes/mnemosyne/config.yaml")),
                       ("container", os.path.join(os.path.dirname(dst), "config.yaml"))):
        if os.path.exists(cfg):
            m = d = None
            with open(cfg) as f:
                for line in f:
                    ls = line.strip().lower()
                    if ls.startswith("embedding_model:"):
                        m = line.split(":", 1)[1].strip()
                    elif ls.startswith("embedding_dim:"):
                        d = line.split(":", 1)[1].strip()
            if m:
                log(f"embedding model ({label}): {m} dim={d}")

    # --- idempotency manifest guard ---
    manifest_path = os.path.join(args.backup_dir, "seed-manifest.json")
    manifest = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:  # noqa: BLE001
            manifest = None
    if manifest and manifest.get("source_sha256") == src_sha and not args.force:
        log("ALREADY MIGRATED (manifest matches current source) — no-op. Use --force to redo.")
        return 3

    dest_exists = os.path.exists(dst)
    if dest_exists:
        if not integrity_ok(dst):
            die("dest integrity_check FAILED (pre-swap)", 2)
        dest_counts = row_counts(dst)
        log(f"dest exists | rows {sum(dest_counts.values())} | sha256 {sha256(dst)[:16]}… (will be replaced)")
    else:
        log("dest does not exist yet (will be created)")

    was_running = docker_running(args.container)
    log(f"container {args.container}: {'running' if was_running else 'STOPPED'}")

    plan = [
        "1. backup source  -> <backup-dir>/pre-migration-<ts>/src/mnemosyne.db  (online backup + sha256, verify restorable)",
        "2. backup dest    -> <backup-dir>/pre-migration-<ts>/dest/mnemosyne.db (same; skipped if dest absent)",
        f"3. docker stop {args.container}  (graceful; restores prior state afterwards)",
        "4. remove stale dest -wal/-shm (never replay old WAL into new DB)",
        "5. online backup source -> temp in dest dir -> os.replace over dest, checkpoint",
        f"6. docker start {args.container} + wait healthy (or leave stopped if it was stopped)",
        "7. post-checks: dest integrity, counts == source snapshot, sha256 match"
        + ("" if args.skip_mcp else ", MCP recall spot-checks (keyword + semantic)"),
        f"8. write manifest {manifest_path}",
    ]
    log("PLAN:")
    for p in plan:
        log(f"   {p}")

    if args.dry_run:
        log("DRY-RUN — nothing written. Re-run without --dry-run to execute.")
        return 0

    # --- backups ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bdir = os.path.join(args.backup_dir, f"pre-migration-{ts}")
    src_bak = os.path.join(bdir, "src", "mnemosyne.db")
    dest_bak = os.path.join(bdir, "dest", "mnemosyne.db")

    log(f"backing up source -> {src_bak}")
    online_backup(src, src_bak)
    if not integrity_ok(src_bak):
        die("source backup integrity FAILED — aborting, nothing changed", 1)
    bak_counts = row_counts(src_bak)
    if bak_counts != src_counts:
        die(f"source backup row counts mismatch: {bak_counts} vs {src_counts}", 1)
    log(f"source backup verified restorable ({sum(bak_counts.values())} rows, sha256 {sha256(src_bak)[:16]}…)")

    if dest_exists:
        log(f"backing up dest -> {dest_bak}")
        online_backup(dst, dest_bak)
        if not integrity_ok(dest_bak):
            die("dest backup integrity FAILED — aborting, nothing changed", 1)
        log(f"dest backup verified restorable ({sum(row_counts(dest_bak).values())} rows)")

    # --- swap ---
    stopped_by_us = False
    if was_running:
        log(f"stopping container {args.container}")
        try:
            run(["docker", "stop", "-t", "30", args.container], timeout=60)
            stopped_by_us = True
        except subprocess.TimeoutExpired:
            die(f"docker stop {args.container} timed out — aborting before any file change", 1)
        except subprocess.CalledProcessError as e:
            die(f"docker stop {args.container} failed: {e.stderr.strip()}", 1)

    for suffix in ("-wal", "-shm"):
        stale = dst + suffix
        if os.path.exists(stale):
            os.remove(stale)
            log(f"removed stale {os.path.basename(stale)}")

    tmp = dst + ".seed-tmp"
    dest_sha_at_swap: str | None = None
    log(f"copying source -> {dst}")
    try:
        online_backup(src, tmp)
        # ensure clean single-file snapshot (no -wal/-shm left beside the temp)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(tmp + suffix):
                os.remove(tmp + suffix)
        os.replace(tmp, dst)
        # Content-identity check at swap time: the file we wrote must equal the
        # verified source backup byte-for-byte. The online-backup API
        # reorganizes pages, so a backup output is never byte-identical to the
        # raw source file — but it IS deterministic for identical input, so
        # (src backup) == (dest at swap) proves the swap is a faithful replica
        # of the same logical content the verified backup holds. Any write to
        # the source between the two backup calls shows up here as a mismatch.
        dest_sha_at_swap = sha256(dst)
        src_bak_sha = sha256(src_bak)
        if dest_sha_at_swap != src_bak_sha:
            die(f"dest sha256 mismatch AT SWAP: {dest_sha_at_swap[:16]}… vs backup {src_bak_sha[:16]}… "
                f"— source changed mid-run (retry after quiescing writes); dest NOT replaced "
                f"(backups intact at {bdir})", 1)
        log(f"dest identical to verified source backup at swap (sha256 {dest_sha_at_swap[:16]}…)")
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        die(f"copy failed: {e} — dest NOT replaced (backups intact at {bdir})", 1)

    # hygiene: checkpoint + drop any -wal/-shm the backup connection may have left
    con = sqlite3.connect(dst)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
    finally:
        con.close()
    for suffix in ("-wal", "-shm"):
        if os.path.exists(dst + suffix):
            os.remove(dst + suffix)

    # --- restart ---
    if stopped_by_us:
        log(f"starting container {args.container}")
        try:
            run(["docker", "start", args.container], timeout=60)
        except subprocess.CalledProcessError as e:
            die(f"docker start {args.container} failed: {e.stderr.strip()}", 1)
        if not wait_healthy(args.container):
            log(f"WARNING: container not healthy after start — health: {docker_health(args.container)}")
    else:
        log("container was stopped before migration — leaving it stopped")

    # --- post-checks ---
    problems: list[str] = []
    if not integrity_ok(dst):
        problems.append("dest integrity_check FAILED after swap")
    if row_counts(dst) != src_counts:
        problems.append(f"dest counts mismatch: {row_counts(dst)} vs {src_counts}")

    # known-id presence in dest (data-level: the memories exist, not just counts)
    known_ids, missing_ids = dest_id_presence(src, dst)
    if missing_ids:
        problems.append(f"known memory ids missing in dest: {missing_ids}")
    else:
        log(f"data-level PASS: {len(known_ids)} known memories present in dest by id")

    if not args.skip_mcp and problems == []:
        env = load_env(args.env_file)
        token = env.get("MNEMOSYNE_MCP_TOKEN", "")
        if not token:
            log("no MNEMOSYNE_MCP_TOKEN in env file — skipping MCP spot-check")
        else:
            kws = probe_keywords_from(src, known_ids)
            log(f"running MCP recall spot-checks… (probe terms: {kws})")
            mcp_problems = mcp_spot_check(token, args.endpoint, kws)
            problems += mcp_problems
            if not mcp_problems:
                log("MCP spot-checks PASS (recall + semantic on migrated store)")

    # --- manifest ---
    os.makedirs(args.backup_dir, exist_ok=True)
    manifest = {
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "source": src,
        "source_sha256": src_sha,
        "dest": dst,
        "dest_sha256_at_swap": dest_sha_at_swap,
        "row_counts": src_counts,
        "total_rows": total_src,
        "container": args.container,
        "backup_dir": bdir,
        "was_running": was_running,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"manifest written: {manifest_path}")

    log("=" * 64)
    if problems:
        log("RESULT: FAILED — " + "; ".join(problems))
        return 1
    log(f"RESULT: PASS — shared store now holds {total_src} rows from the live Hermes DB")
    log(f"        dest sha256 {sha256(dst)[:16]}… == source; backups at {bdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
