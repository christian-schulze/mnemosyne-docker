"""Stage-3 concurrency hardening tests (fork of AxDSan/mnemosyne).

Covers the three Stage-3 deliverables:

1. ``MNEMOSYNE_BUSY_TIMEOUT_MS`` env override — honored by every connection
   factory (beam, legacy ``core.memory``, veracity consolidator). Default
   5000ms is the spike-proven value.
2. Single-writer consolidation gate — ``MNEMOSYNE_CONSOLIDATOR=0`` makes
   sleep()/sleep_all_sessions() return status="skipped" without touching the
   DB; the cross-process flock (``_ConsolidationLock``) excludes a second
   process even when consolidation is enabled in both.
3. Multi-process contention — the promoted spike harness: 2-3 processes
   (writers + consolidator) on ONE SQLite file, asserting zero
   "database is locked" errors. Marked ``contention`` (slower, spawns
   subprocesses); excluded from the main CI test job, run by the dedicated
   ``contention`` job.

Multi-process tests never touch the live memory store: every run uses a
pytest tmp_path DB. Durations default to 8s and can be scaled via
``MNEMOSYNE_CONTENTION_SECONDS``.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory, _ConsolidationLock

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_ERROR = "database is locked"


# ---------------------------------------------------------------------------
# busy_timeout env override
# ---------------------------------------------------------------------------

class TestBusyTimeoutEnv:

    def test_default_is_5000(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_BUSY_TIMEOUT_MS", raising=False)
        beam = BeamMemory(session_id="bt-default", db_path=tmp_path / "d.db")
        (ms,) = beam.conn.execute("PRAGMA busy_timeout").fetchone()
        assert ms == 5000

    def test_beam_connection_honors_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BUSY_TIMEOUT_MS", "12000")
        beam = BeamMemory(session_id="bt-beam", db_path=tmp_path / "b.db")
        (ms,) = beam.conn.execute("PRAGMA busy_timeout").fetchone()
        assert ms == 12000

    def test_legacy_memory_connection_honors_env(self, tmp_path, monkeypatch):
        from mnemosyne.core import memory
        monkeypatch.setenv("MNEMOSYNE_BUSY_TIMEOUT_MS", "7777")
        conn = memory._get_connection(tmp_path / "legacy.db")
        try:
            (ms,) = conn.execute("PRAGMA busy_timeout").fetchone()
        finally:
            conn.close()
        assert ms == 7777

    def test_veracity_consolidator_honors_env(self, tmp_path, monkeypatch):
        from mnemosyne.core.veracity_consolidation import VeracityConsolidator
        monkeypatch.setenv("MNEMOSYNE_BUSY_TIMEOUT_MS", "9999")
        vc = VeracityConsolidator(db_path=tmp_path / "vc.db")
        try:
            (ms,) = vc.conn.execute("PRAGMA busy_timeout").fetchone()
        finally:
            vc.close()
        assert ms == 9999

    def test_invalid_env_falls_back_to_5000(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BUSY_TIMEOUT_MS", "not-a-number")
        beam = BeamMemory(session_id="bt-bad", db_path=tmp_path / "bad.db")
        (ms,) = beam.conn.execute("PRAGMA busy_timeout").fetchone()
        assert ms == 5000


# ---------------------------------------------------------------------------
# Single-writer consolidation gate
# ---------------------------------------------------------------------------

class TestConsolidatorGate:

    def test_gate_disabled_skips_sleep(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATOR", "0")
        beam = BeamMemory(session_id="gate-off", db_path=tmp_path / "g.db")
        r = beam.sleep(force=True)
        assert r["status"] == "skipped"
        assert "MNEMOSYNE_CONSOLIDATOR" in r["message"]

    def test_gate_disabled_skips_sleep_all_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATOR", "0")
        beam = BeamMemory(session_id="gate-off-as", db_path=tmp_path / "g2.db")
        r = beam.sleep_all_sessions(force=True)
        assert r["status"] == "skipped"

    def test_gate_explicit_enabled_runs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATOR", "1")
        beam = BeamMemory(session_id="gate-on", db_path=tmp_path / "g3.db")
        # Empty DB + enabled -> no_op (i.e. it actually ran), never "skipped".
        r = beam.sleep(force=True)
        assert r["status"] != "skipped"

    def test_gate_unset_defaults_enabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATOR", raising=False)
        beam = BeamMemory(session_id="gate-default", db_path=tmp_path / "g4.db")
        r = beam.sleep(force=True)
        assert r["status"] != "skipped"

    def test_flock_noop_on_in_memory_db(self, monkeypatch):
        """sleep() on an in-memory DB must not try to flock a file path."""
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATOR", raising=False)
        beam = BeamMemory(session_id="gate-mem", db_path=":memory:")
        r = beam.sleep(force=True)
        # Empty in-memory DB + enabled -> no_op (i.e. it actually ran).
        assert r["status"] != "skipped"

    def test_flock_excludes_second_process(self, tmp_path):
        """Hold the consolidator lock in THIS process; a second process
        calling sleep() with consolidation enabled must return 'skipped'
        rather than queueing or consolidating concurrently."""
        db = tmp_path / "flock.db"
        lock = _ConsolidationLock(db)
        assert lock.acquire() is True
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            env["MNEMOSYNE_CONSOLIDATOR"] = "1"
            env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
            code = (
                "import sys; sys.path.insert(0, %r); "
                "from mnemosyne.core.beam import BeamMemory; "
                "m = BeamMemory(session_id='flock-child', db_path=%r); "
                "r = m.sleep(force=True); "
                "print('status=' + r['status']); "
                "sys.exit(0 if r['status'] == 'skipped' else 1)"
                % (str(REPO_ROOT), str(db))
            )
            out = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert out.returncode == 0, f"child stderr: {out.stderr}"
            assert "status=skipped" in out.stdout, out.stdout
        finally:
            lock.release()


# ---------------------------------------------------------------------------
# Multi-process contention (promoted spike harness)
# ---------------------------------------------------------------------------

def _duration() -> float:
    return float(os.environ.get("MNEMOSYNE_CONTENTION_SECONDS", "8"))


def _pre_seed(db: Path, count: int = 300) -> None:
    """Create schema + seed facts in a single process (avoids first-open
    races when the concurrent processes all start at once)."""
    mem = BeamMemory(session_id="contention-seed", db_path=db)
    for i in range(count):
        mem.remember(
            content=f"seed fact {i}: lorem ipsum dolor sit amet consectetur",
            source="contention_seed",
            importance=0.5,
            scope="global",
        )


def _writer_a(db: Path, seconds: float) -> None:
    """Burst writer: rapid-fire global remembers — heavy write pressure."""
    mem = BeamMemory(session_id="contention_writerA", db_path=db)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        try:
            for i in range(10):
                mem.remember(
                    content=f"burst fact {n}-{i}: alpha beta gamma delta epsilon",
                    source="contention",
                    importance=0.6,
                    scope="global",
                )
            n += 1
            time.sleep(0.2)
        except Exception as e:  # noqa: BLE001
            print(f"[writerA] ERROR {type(e).__name__}: {e}", flush=True)
    print(f"[writerA] done: {n} bursts in {seconds:.0f}s", flush=True)


def _writer_b(db: Path, seconds: float) -> None:
    """Light writer + reader: recall then a single write — agent-like."""
    mem = BeamMemory(session_id="contention_writerB", db_path=db)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        try:
            mem.recall("alpha beta", top_k=5)
            mem.remember(
                content=f"agent fact {n}: delta epsilon zeta eta",
                source="contention",
                importance=0.5,
                scope="global",
            )
            n += 1
            time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            print(f"[writerB] ERROR {type(e).__name__}: {e}", flush=True)
    print(f"[writerB] done: {n} cycles in {seconds:.0f}s", flush=True)


def _consolidator(db: Path, seconds: float) -> None:
    """Heavy consolidation: sleep(force=True) repeatedly — the highest-risk
    writer (long LLM windows in production; aaak fallback here)."""
    mem = BeamMemory(session_id="contention_consolidator", db_path=db)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        try:
            mem.sleep(force=True)
            n += 1
            time.sleep(3.0)
        except Exception as e:  # noqa: BLE001
            print(f"[consolidator] ERROR {type(e).__name__}: {e}", flush=True)
    print(f"[consolidator] done: {n} sleep runs in {seconds:.0f}s", flush=True)


_ROLES = {
    "writerA": _writer_a,
    "writerB": _writer_b,
    "consolidator": _consolidator,
}


def _run_roles(db: Path, roles, seconds: float) -> str:
    """Spawn one subprocess per role on the SAME db; return concatenated
    output. Each child runs the fork code (PYTHONPATH forced) with
    consolidation explicitly enabled and embeddings off (deterministic,
    no HF downloads — lock behavior is independent of vectors)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MNEMOSYNE_CONSOLIDATOR"] = "1"
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, role, str(db), str(seconds)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for role in roles
    ]
    outputs = []
    for p in procs:
        out, _ = p.communicate(timeout=seconds + 60)
        outputs.append(out)
    return "\n".join(outputs)


def _assert_zero_lock_errors(output: str) -> None:
    lock_lines = [ln for ln in output.splitlines() if LOCK_ERROR in ln]
    other_errors = [ln for ln in output.splitlines() if "ERROR" in ln and LOCK_ERROR not in ln]
    assert not lock_lines, "'database is locked' errors:\n" + "\n".join(lock_lines)
    assert not other_errors, "other errors:\n" + "\n".join(other_errors)


@pytest.mark.contention
def test_two_writers_no_lock_errors(tmp_path):
    db = tmp_path / "two.db"
    _pre_seed(db)
    out = _run_roles(db, ["writerA", "writerB"], _duration())
    _assert_zero_lock_errors(out)


@pytest.mark.contention
def test_writer_plus_consolidator_no_lock_errors(tmp_path):
    db = tmp_path / "wcons.db"
    _pre_seed(db)
    out = _run_roles(db, ["writerA", "consolidator"], _duration())
    _assert_zero_lock_errors(out)


@pytest.mark.contention
def test_three_way_no_lock_errors(tmp_path):
    db = tmp_path / "three.db"
    _pre_seed(db)
    out = _run_roles(db, ["writerA", "writerB", "consolidator"], _duration())
    _assert_zero_lock_errors(out)


@pytest.mark.contention
def test_three_burst_writers_no_lock_errors(tmp_path):
    """Spike scenario 4: three simultaneous burst writers."""
    db = tmp_path / "burst.db"
    _pre_seed(db)
    out = _run_roles(db, ["writerA", "writerA", "writerA"], _duration())
    _assert_zero_lock_errors(out)


if __name__ == "__main__":
    # Subprocess dispatch for the contention roles: `module role db seconds`
    role, db, seconds = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
    _ROLES[role](db, seconds)
