#!/usr/bin/env python3
"""Stage-4 config-env verification harness (issue #482) — run on the HOST.

Sets every ``MNEMOSYNE_*`` config env var to a distinct value, runs the
probe inside a FRESH container (fresh data dir), and asserts the effective
runtime configuration reflects every key:

- **cfg** — all ~106 mapped keys honored by the config system
  (``get_config().get`` after auto-seed).
- **const** — the ~50 module-level constants that previously read
  ``os.environ`` at import time now reflect the env via the config bridge.
- **acc** — call-site accessors (model-refresh thresholds, filters, recall
  gates) reflect the env.

Exit 0 only when every key is honored. Usage:

    python3 deploy/verify_config_env.py [--image ghcr.io/christian-schulze/mnemosyne-docker:latest]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "deploy" / "verify_config_probe.py"

sys.path.insert(0, str(REPO_ROOT))
from mnemosyne.core.config import DEFAULTS, ENV_VAR_MAP  # noqa: E402

CONSUMER_KINDS = {
    "wm_max_items": "int", "wm_ttl_hours": "int", "wm_bump_cap_hours": "int",
    "wm_pinned_ids": "set", "ep_limit": "int", "sleep_batch": "int",
    "sp_max": "int", "recency_halflife": "float", "tier2_days": "int",
    "tier3_days": "int", "tier1_weight": "float", "tier2_weight": "float",
    "tier3_weight": "float", "degrade_batch": "int", "smart_compress": "bool",
    "tier3_max_chars": "int", "vec_type": "str", "embedding_dim": "int",
    "llm_enabled": "bool", "llm_max_tokens": "int", "llm_n_threads": "int",
    "llm_n_ctx": "int", "llm_base_url": "str", "llm_api_key": "str",
    "llm_model": "str", "llm_timeout": "float", "llm_fallback_models": "list",
    "llm_fallback_base_url": "str", "llm_fallback_api_key": "str",
    "host_llm_enabled": "bool", "host_llm_provider": "str",
    "host_llm_model": "str", "host_llm_n_ctx": "int", "sleep_prompt": "str",
    "shmr_batch_size": "int", "shmr_max_iterations": "int",
    "shmr_similarity_threshold": "float", "shmr_harmony_threshold": "float",
    "shmr_model": "str", "shmr_min_cluster_size": "int",
    "shmr_temperature": "float", "llm_conflict_detection": "bool",
    "conflict_llm_base_url": "str", "conflict_llm_api_key": "str",
    "conflict_llm_model": "str", "persona_interval": "int",
    "persona_daily_sync_hour": "int", "persona_token_cap": "int",
    "embedding_model": "str", "embedding_api_url": "str",
    "embedding_api_key": "str", "fastembed_cache_dir": "str",
    "data_dir": "path",
}

# Special env values that must be VALID for the consumer (not arbitrary).
SPECIAL_ENV = {
    "vec_type": "float32",
    "data_dir": "/data",
    "wm_pinned_ids": "11111,22222",
    "llm_fallback_models": "cfgtest-fb1,cfgtest-fb2",
    "llm_repo": "cfgtest/repo",
    "llm_file": "cfgtest-file.gguf",
    "ignore_patterns": "cfgtest-p1\ncfgtest-p2",
    "write_classifier": "strict",
    "sleep_model_refresh_categories": "model:user,model:workflow",
    "embedding_dim": "768",
    "llm_n_ctx": "4096",
    "host_llm_n_ctx": "32768",
    "shmr_max_iterations": "8",
    "persona_interval": "120",
    "persona_token_cap": "4000",
    "sleep_batch": "900",
    "sp_max": "800",
    "ep_limit": "700",
    "wm_max_items": "600",
    "wm_ttl_hours": "72",
    "wm_bump_cap_hours": "48",
    "tier2_days": "90",
    "tier3_days": "365",
    "degrade_batch": "500",
    "tier3_max_chars": "1000",
    "recency_halflife": "48.5",
    "llm_timeout": "42.5",
    "shmr_similarity_threshold": "0.65",
    "shmr_harmony_threshold": "0.55",
    "shmr_temperature": "0.45",
    "tier1_weight": "0.9",
    "tier2_weight": "0.7",
    "tier3_weight": "0.4",
    "llm_max_tokens": "1500",
    "llm_n_threads": "6",
    "sleep_model_refresh_max_tokens": "2048",
    "sleep_model_refresh_temperature": "0.2",
    "sleep_model_refresh_auto_apply_min_confidence": "0.55",
    "sleep_model_refresh_min_evidence": "4",
    "sleep_model_refresh_conflict_min_confidence": "0.66",
    "sleep_model_refresh_conflict_min_evidence": "5",
    "persona_daily_sync_hour": "6",
    "sync_port": "8877",
    "prefetch_content_chars": "1500",
    "sync_turn_user_limit": "25",
    "sync_turn_assistant_limit": "30",
    "reflect_max_calls_per_session": "5",
    "temporal_halflife_hours": "36.5",
    "vec_weight": "0.6",
    "fts_weight": "0.25",
    "importance_weight": "0.15",
}


def env_value(key: str, i: int) -> str:
    """Distinct, valid env value per key (mirrors seed type-coercion)."""
    if key in SPECIAL_ENV:
        return SPECIAL_ENV[key]
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return "false" if default else "true"  # flip vs default to prove binding
    if isinstance(default, int):
        return str(90000 + i)
    if isinstance(default, float):
        return f"{0.51 + i / 100:.4f}"
    return f"cfgtest-{key}"


def coerce_env(key: str, raw: str):
    """Mirror MnemosyneConfig._seed coercion exactly."""
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            return default
    return raw


def expected_const(key: str, raw: str):
    kind = CONSUMER_KINDS[key]
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        return coerce_env(key, raw)
    if kind == "set":
        return sorted(p.strip() for p in raw.split(",") if p.strip())
    if kind == "list":
        return [m.strip() for m in raw.split(",") if m.strip()]
    if kind == "path":
        return raw
    return raw


def _acc_expect(env_vars: dict) -> dict:
    """Expected accessor results derived from the env vars set."""
    return {
        "sleep_model_refresh_enabled": coerce_env(
            "sleep_model_refresh_enabled", env_vars["MNEMOSYNE_SLEEP_MODEL_REFRESH_ENABLED"]
        ),
        "sleep_model_refresh_categories": "['model:user', 'model:workflow']",
        "sleep_model_refresh_auto_apply": coerce_env(
            "sleep_model_refresh_auto_apply", env_vars["MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY"]
        ),
        "sleep_model_refresh_auto_apply_min_confidence": float(
            env_vars["MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY_MIN_CONFIDENCE"]
        ),
        "sleep_model_refresh_min_evidence": int(
            env_vars["MNEMOSYNE_SLEEP_MODEL_REFRESH_MIN_EVIDENCE"]
        ),
        "sleep_model_refresh_conflict_min_confidence": float(
            env_vars["MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_CONFIDENCE"]
        ),
        "sleep_model_refresh_conflict_min_evidence": max(
            int(env_vars["MNEMOSYNE_SLEEP_MODEL_REFRESH_MIN_EVIDENCE"]),
            int(env_vars["MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_EVIDENCE"]),
        ),
        "ignore_patterns": ["cfgtest-p1", "cfgtest-p2"],
        "write_classifier": "strict",
        "no_embeddings": True,
        "embeddings_via_api": True,
        "enhanced_recall": True,
        "cross_session": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="ghcr.io/christian-schulze/mnemosyne-docker:latest")
    args = ap.parse_args()

    if not PROBE.exists():
        print(f"probe not found: {PROBE}")
        return 2

    env_vars = {ENV_VAR_MAP[k]: env_value(k, i) for i, k in enumerate(sorted(ENV_VAR_MAP))}
    env_vars["MNEMOSYNE_DATA_DIR"] = "/data"

    work = Path(tempfile.mkdtemp(prefix="stage4-verify-"))
    data_dir = work / "data"
    data_dir.mkdir()
    os.chown(data_dir, 1000, 1000)  # uid-1000 container must write the seed

    cmd = [
        "docker", "run", "--rm",
        "--user", "1000:1000",
        "-v", f"{data_dir}:/data",
        "-v", f"{REPO_ROOT / 'deploy'}:/verify:ro",
        "-e", "MNEMOSYNE_DATA_DIR=/data",
        "-e", "MNEMOSYNE_CONSOLIDATOR=0",
    ]
    for k, v in sorted(env_vars.items()):
        cmd += ["-e", f"{k}={v}"]
    cmd += ["--entrypoint", "python3", args.image, "/verify/verify_config_probe.py"]

    print(f"data dir : {data_dir}")
    print(f"image    : {args.image}")
    print(f"env vars : {len(env_vars)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print("PROBE FAILED:\n", proc.stderr[-4000:])
        return proc.returncode

    report = json.loads(proc.stdout.strip().splitlines()[-1])

    failures = []
    checked = 0

    # --- cfg: 100% of mapped keys honored by the config system ---
    # With a fresh data dir the (lazy) seed has not materialized config.yaml,
    # so get() returns the raw env STRING for each key; with a config.yaml
    # present it returns the typed YAML value. Accept either — both prove the
    # env value is the effective config.
    for key in sorted(ENV_VAR_MAP):
        checked += 1
        raw = env_vars[ENV_VAR_MAP[key]]
        expected = coerce_env(key, raw)
        got = report["cfg"].get(key)
        if not (got == expected or str(got) == raw):
            failures.append(f"cfg:{key}: got {got!r} want {raw!r} (typed {expected!r})")

    # --- const: module-level consumers ---
    for key, kind in CONSUMER_KINDS.items():
        checked += 1
        expected = expected_const(key, env_vars[ENV_VAR_MAP[key]])
        got = report["const"].get(key)
        if got != expected:
            failures.append(f"const:{key}: got {got!r} want {expected!r}")

    # --- acc: call-site accessors ---
    acc_expect = _acc_expect(env_vars)
    for key, expected in acc_expect.items():
        checked += 1
        got = report["acc"].get(key)
        if got != expected:
            failures.append(f"acc:{key}: got {got!r} want {expected!r}")

    # --- report ---
    print(f"\nchecked {checked} assertions ({len(ENV_VAR_MAP)} cfg keys + "
          f"{len(CONSUMER_KINDS)} constants + {len(acc_expect)} accessors)")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print("  " + f)
        return 1
    print("\nALL CONFIG KEYS HONORED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
