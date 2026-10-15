"""Stage-4 config-env binding tests (fork of AxDSan/mnemosyne, issue #482).

Proves the fork delta for the "50 silently-ignored config keys" bug: module
constants and runtime accessors resolve through the central config
(``config.yaml > env var > code default``) instead of reading ``os.environ``
into import-time constants.

Three phases, each a single subprocess with an isolated temp data dir (module
state is frozen at import; the live memory store is never touched):

1. **Env-driven** — every mapped key set via ``MNEMOSYNE_*`` env var in a
   fresh dir: the config auto-seed captures env, so constants must reflect
   the env values.
2. **Config-driven** — a pre-written ``config.yaml`` (no env vars): constants
   must reflect the config values, proving the previously-ignored keys now
   take effect from config.yaml.
3. **Behavior-neutral** — fresh dir, no env, no config: constants must equal
   the code defaults (a fresh seed must not change behaviour vs pre-fork).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Consumers: config key -> (module, attribute) evaluated by the probe
# ---------------------------------------------------------------------------

# (config_key, module, attribute, kind)  kind: int|float|bool|str|set|path
CONSUMERS = [
    # beam.py — tiers / lifecycle
    ("wm_max_items", "mnemosyne.core.beam", "WORKING_MEMORY_MAX_ITEMS", "int"),
    ("wm_ttl_hours", "mnemosyne.core.beam", "WORKING_MEMORY_TTL_HOURS", "int"),
    ("wm_bump_cap_hours", "mnemosyne.core.beam", "WM_BUMP_CAP_HOURS", "int"),
    ("wm_pinned_ids", "mnemosyne.core.beam", "WM_PINNED_IDS", "set"),
    ("ep_limit", "mnemosyne.core.beam", "EPISODIC_RECALL_LIMIT", "int"),
    ("sleep_batch", "mnemosyne.core.beam", "SLEEP_BATCH_SIZE", "int"),
    ("sp_max", "mnemosyne.core.beam", "SCRATCHPAD_MAX_ITEMS", "int"),
    ("recency_halflife", "mnemosyne.core.beam", "RECENCY_HALFLIFE_HOURS", "float"),
    ("tier2_days", "mnemosyne.core.beam", "TIER2_DAYS", "int"),
    ("tier3_days", "mnemosyne.core.beam", "TIER3_DAYS", "int"),
    ("tier1_weight", "mnemosyne.core.beam", "TIER1_WEIGHT", "float"),
    ("tier2_weight", "mnemosyne.core.beam", "TIER2_WEIGHT", "float"),
    ("tier3_weight", "mnemosyne.core.beam", "TIER3_WEIGHT", "float"),
    ("degrade_batch", "mnemosyne.core.beam", "DEGRADE_BATCH_SIZE", "int"),
    ("smart_compress", "mnemosyne.core.beam", "SMART_COMPRESS", "bool"),
    ("tier3_max_chars", "mnemosyne.core.beam", "TIER3_MAX_CHARS", "int"),
    ("vec_type", "mnemosyne.core.beam", "VEC_TYPE", "str"),
    ("embedding_dim", "mnemosyne.core.beam", "EMBEDDING_DIM", "int"),
    # local_llm.py
    ("llm_enabled", "mnemosyne.core.local_llm", "LLM_ENABLED", "bool"),
    ("llm_max_tokens", "mnemosyne.core.local_llm", "LLM_MAX_TOKENS", "int"),
    ("llm_n_threads", "mnemosyne.core.local_llm", "LLM_N_THREADS", "int"),
    ("llm_n_ctx", "mnemosyne.core.local_llm", "LLM_N_CTX", "int"),
    ("llm_base_url", "mnemosyne.core.local_llm", "LLM_BASE_URL", "str"),
    ("llm_api_key", "mnemosyne.core.local_llm", "LLM_API_KEY", "str"),
    ("llm_model", "mnemosyne.core.local_llm", "LLM_REMOTE_MODEL", "str"),
    ("llm_timeout", "mnemosyne.core.local_llm", "LLM_TIMEOUT", "float"),
    ("host_llm_enabled", "mnemosyne.core.local_llm", "HOST_LLM_ENABLED", "bool"),
    ("host_llm_provider", "mnemosyne.core.local_llm", "HOST_LLM_PROVIDER", "str"),
    ("host_llm_model", "mnemosyne.core.local_llm", "HOST_LLM_MODEL", "str"),
    ("host_llm_n_ctx", "mnemosyne.core.local_llm", "HOST_LLM_N_CTX", "int"),
    ("sleep_prompt", "mnemosyne.core.local_llm", "SLEEP_PROMPT", "str"),
    # shmr.py
    ("shmr_batch_size", "mnemosyne.core.shmr", "SHMR_BATCH_SIZE", "int"),
    ("shmr_max_iterations", "mnemosyne.core.shmr", "SHMR_MAX_ITERATIONS", "int"),
    ("shmr_similarity_threshold", "mnemosyne.core.shmr", "SHMR_SIMILARITY_THRESHOLD", "float"),
    ("shmr_harmony_threshold", "mnemosyne.core.shmr", "SHMR_HARMONY_THRESHOLD", "float"),
    ("shmr_model", "mnemosyne.core.shmr", "SHMR_MODEL", "str"),
    ("shmr_min_cluster_size", "mnemosyne.core.shmr", "SHMR_MIN_CLUSTER_SIZE", "int"),
    ("shmr_temperature", "mnemosyne.core.shmr", "SHMR_TEMPERATURE", "float"),
    # llm_conflict_detector.py
    ("llm_conflict_detection", "mnemosyne.core.llm_conflict_detector", "LLM_CONFLICT_DETECTION_ENABLED", "bool"),
    ("conflict_llm_base_url", "mnemosyne.core.llm_conflict_detector", "CONFLICT_LLM_BASE_URL", "str"),
    ("conflict_llm_api_key", "mnemosyne.core.llm_conflict_detector", "CONFLICT_LLM_API_KEY", "str"),
    ("conflict_llm_model", "mnemosyne.core.llm_conflict_detector", "CONFLICT_LLM_MODEL", "str"),
    # persona.py
    ("persona_interval", "mnemosyne.core.persona", "DEFAULT_INTERVAL", "int"),
    ("persona_daily_sync_hour", "mnemosyne.core.persona", "DEFAULT_DAILY_SYNC_HOUR", "int"),
    ("persona_token_cap", "mnemosyne.core.persona", "DEFAULT_TOKEN_CAP", "int"),
    # embeddings.py
    ("embedding_model", "mnemosyne.core.embeddings", "_DEFAULT_MODEL", "str"),
    ("embedding_api_url", "mnemosyne.core.embeddings", "_OPENAI_BASE_URL", "str"),
    ("embedding_api_key", "mnemosyne.core.embeddings", "_OPENAI_API_KEY", "str"),
    ("fastembed_cache_dir", "mnemosyne.core.embeddings", "_FASTEMBED_CACHE_DIR", "str"),
]

# Accessor-style consumers: (config_key, module, expression)
ACCESSORS = [
    ("sleep_model_refresh_enabled", "mnemosyne.core.model_refresh", "sleep_model_refresh_enabled()"),
    ("sleep_model_refresh_categories", "mnemosyne.core.model_refresh", "str(sorted(_allowed_categories_from_env()))"),
    ("sleep_model_refresh_auto_apply", "mnemosyne.core.model_refresh", "auto_apply_enabled()"),
    ("sleep_model_refresh_auto_apply_min_confidence", "mnemosyne.core.model_refresh", "auto_apply_min_confidence()"),
    ("sleep_model_refresh_min_evidence", "mnemosyne.core.model_refresh", "auto_apply_min_evidence()"),
    ("sleep_model_refresh_conflict_min_confidence", "mnemosyne.core.model_refresh", "auto_apply_conflict_min_confidence()"),
    ("sleep_model_refresh_conflict_min_evidence", "mnemosyne.core.model_refresh", "auto_apply_conflict_min_evidence()"),
    ("ignore_patterns", "mnemosyne.core.filters", "_load_ignore_patterns_from_env()"),
    ("write_classifier", "mnemosyne.core.filters", "_load_classifier_mode()"),
    ("no_embeddings", "mnemosyne.core.embeddings", "_is_disabled()"),
    ("embeddings_via_api", "mnemosyne.core.embeddings", "_is_api_model('qwen/qwen3-embedding-0.6b')"),
    ("enhanced_recall", "mnemosyne.core.memory", "get_bool('enhanced_recall', False)"),
    ("cross_session", "mnemosyne.core.config", "resolve_beam_runtime().cross_session"),
]

# Env var names per key (from ENV_VAR_MAP)
ENV_MAP = {
    "wm_max_items": "MNEMOSYNE_WM_MAX_ITEMS",
    "wm_ttl_hours": "MNEMOSYNE_WM_TTL_HOURS",
    "wm_bump_cap_hours": "MNEMOSYNE_WM_BUMP_CAP_HOURS",
    "wm_pinned_ids": "MNEMOSYNE_WM_PINNED_IDS",
    "ep_limit": "MNEMOSYNE_EP_LIMIT",
    "sleep_batch": "MNEMOSYNE_SLEEP_BATCH",
    "sp_max": "MNEMOSYNE_SP_MAX",
    "recency_halflife": "MNEMOSYNE_RECENCY_HALFLIFE",
    "tier2_days": "MNEMOSYNE_TIER2_DAYS",
    "tier3_days": "MNEMOSYNE_TIER3_DAYS",
    "tier1_weight": "MNEMOSYNE_TIER1_WEIGHT",
    "tier2_weight": "MNEMOSYNE_TIER2_WEIGHT",
    "tier3_weight": "MNEMOSYNE_TIER3_WEIGHT",
    "degrade_batch": "MNEMOSYNE_DEGRADE_BATCH",
    "smart_compress": "MNEMOSYNE_SMART_COMPRESS",
    "tier3_max_chars": "MNEMOSYNE_TIER3_MAX_CHARS",
    "vec_type": "MNEMOSYNE_VEC_TYPE",
    "embedding_dim": "MNEMOSYNE_EMBEDDING_DIM",
    "llm_enabled": "MNEMOSYNE_LLM_ENABLED",
    "llm_max_tokens": "MNEMOSYNE_LLM_MAX_TOKENS",
    "llm_n_threads": "MNEMOSYNE_LLM_N_THREADS",
    "llm_n_ctx": "MNEMOSYNE_LLM_N_CTX",
    "llm_base_url": "MNEMOSYNE_LLM_BASE_URL",
    "llm_api_key": "MNEMOSYNE_LLM_API_KEY",
    "llm_model": "MNEMOSYNE_LLM_MODEL",
    "llm_timeout": "MNEMOSYNE_LLM_TIMEOUT",
    "host_llm_enabled": "MNEMOSYNE_HOST_LLM_ENABLED",
    "host_llm_provider": "MNEMOSYNE_HOST_LLM_PROVIDER",
    "host_llm_model": "MNEMOSYNE_HOST_LLM_MODEL",
    "host_llm_n_ctx": "MNEMOSYNE_HOST_LLM_N_CTX",
    "sleep_prompt": "MNEMOSYNE_SLEEP_PROMPT",
    "shmr_batch_size": "MNEMOSYNE_SHMR_BATCH_SIZE",
    "shmr_max_iterations": "MNEMOSYNE_SHMR_MAX_ITERATIONS",
    "shmr_similarity_threshold": "MNEMOSYNE_SHMR_SIMILARITY_THRESHOLD",
    "shmr_harmony_threshold": "MNEMOSYNE_SHMR_HARMONY_THRESHOLD",
    "shmr_model": "MNEMOSYNE_SHMR_MODEL",
    "shmr_min_cluster_size": "MNEMOSYNE_SHMR_MIN_CLUSTER_SIZE",
    "shmr_temperature": "MNEMOSYNE_SHMR_TEMPERATURE",
    "llm_conflict_detection": "MNEMOSYNE_LLM_CONFLICT_DETECTION",
    "conflict_llm_base_url": "MNEMOSYNE_CONFLICT_LLM_BASE_URL",
    "conflict_llm_api_key": "MNEMOSYNE_CONFLICT_LLM_API_KEY",
    "conflict_llm_model": "MNEMOSYNE_CONFLICT_LLM_MODEL",
    "persona_interval": "MNEMOSYNE_PERSONA_INTERVAL",
    "persona_daily_sync_hour": "MNEMOSYNE_PERSONA_DAILY_SYNC_HOUR",
    "persona_token_cap": "MNEMOSYNE_PERSONA_TOKEN_CAP",
    "embedding_model": "MNEMOSYNE_EMBEDDING_MODEL",
    "embedding_api_url": "MNEMOSYNE_EMBEDDING_API_URL",
    "embedding_api_key": "MNEMOSYNE_EMBEDDING_API_KEY",
    "fastembed_cache_dir": "MNEMOSYNE_FASTEMBED_CACHE_DIR",
    "sleep_model_refresh_enabled": "MNEMOSYNE_SLEEP_MODEL_REFRESH_ENABLED",
    "sleep_model_refresh_categories": "MNEMOSYNE_SLEEP_MODEL_REFRESH_CATEGORIES",
    "sleep_model_refresh_max_tokens": "MNEMOSYNE_SLEEP_MODEL_REFRESH_MAX_TOKENS",
    "sleep_model_refresh_temperature": "MNEMOSYNE_SLEEP_MODEL_REFRESH_TEMPERATURE",
    "sleep_model_refresh_auto_apply": "MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY",
    "sleep_model_refresh_auto_apply_min_confidence": "MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY_MIN_CONFIDENCE",
    "sleep_model_refresh_min_evidence": "MNEMOSYNE_SLEEP_MODEL_REFRESH_MIN_EVIDENCE",
    "sleep_model_refresh_conflict_min_confidence": "MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_CONFIDENCE",
    "sleep_model_refresh_conflict_min_evidence": "MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_EVIDENCE",
    "ignore_patterns": "MNEMOSYNE_IGNORE_PATTERNS",
    "write_classifier": "MNEMOSYNE_WRITE_CLASSIFIER",
    "no_embeddings": "MNEMOSYNE_NO_EMBEDDINGS",
    "embeddings_via_api": "MNEMOSYNE_EMBEDDINGS_VIA_API",
    "enhanced_recall": "MNEMOSYNE_ENHANCED_RECALL",
}

# Distinct test value per key, chosen to differ from every code default.
# (config_key, kind, env_string, expected_typed)
TEST_VALUES = [
    ("wm_max_items", "int", "99991", 99991),
    ("wm_ttl_hours", "int", "99992", 99992),
    ("wm_bump_cap_hours", "int", "99993", 99993),
    ("wm_pinned_ids", "set", "11111,22222", ["11111", "22222"]),
    ("ep_limit", "int", "99994", 99994),
    ("sleep_batch", "int", "99995", 99995),
    ("sp_max", "int", "99996", 99996),
    ("recency_halflife", "float", "999.5", 999.5),
    ("tier2_days", "int", "998", 998),
    ("tier3_days", "int", "997", 997),
    ("tier1_weight", "float", "0.91", 0.91),
    ("tier2_weight", "float", "0.92", 0.92),
    ("tier3_weight", "float", "0.93", 0.93),
    ("degrade_batch", "int", "996", 996),
    ("smart_compress", "bool", "false", False),
    ("tier3_max_chars", "int", "995", 995),
    ("vec_type", "str", "float32", "float32"),
    ("embedding_dim", "int", "768", 768),
    ("llm_enabled", "bool", "false", False),
    ("llm_max_tokens", "int", "994", 994),
    ("llm_n_threads", "int", "993", 993),
    ("llm_n_ctx", "int", "992", 992),
    ("llm_base_url", "str", "http://cfgtest.local/v1", "http://cfgtest.local/v1"),
    ("llm_api_key", "str", "cfgtest-llm-key", "cfgtest-llm-key"),
    ("llm_model", "str", "cfgtest-model", "cfgtest-model"),
    ("llm_timeout", "float", "12.5", 12.5),
    ("host_llm_enabled", "bool", "true", True),
    ("host_llm_provider", "str", "cfgtest-provider", "cfgtest-provider"),
    ("host_llm_model", "str", "cfgtest-host-model", "cfgtest-host-model"),
    ("host_llm_n_ctx", "int", "991", 991),
    ("sleep_prompt", "str", "cfgtest-prompt", "cfgtest-prompt"),
    ("shmr_batch_size", "int", "990", 990),
    ("shmr_max_iterations", "int", "989", 989),
    ("shmr_similarity_threshold", "float", "0.81", 0.81),
    ("shmr_harmony_threshold", "float", "0.82", 0.82),
    ("shmr_model", "str", "cfgtest-shmr", "cfgtest-shmr"),
    ("shmr_min_cluster_size", "int", "988", 988),
    ("shmr_temperature", "float", "0.83", 0.83),
    ("llm_conflict_detection", "bool", "true", True),
    ("conflict_llm_base_url", "str", "http://conflict.test/v1", "http://conflict.test/v1"),
    ("conflict_llm_api_key", "str", "cfgtest-conflict-key", "cfgtest-conflict-key"),
    ("conflict_llm_model", "str", "cfgtest-conflict-model", "cfgtest-conflict-model"),
    ("persona_interval", "int", "987", 987),
    ("persona_daily_sync_hour", "int", "986", 986),
    ("persona_token_cap", "int", "985", 985),
    ("embedding_model", "str", "cfgtest/embed-model", "cfgtest/embed-model"),
    ("embedding_api_url", "str", "http://embed.test/v1", "http://embed.test/v1"),
    ("embedding_api_key", "str", "cfgtest-embed-key", "cfgtest-embed-key"),
    ("fastembed_cache_dir", "str", "/cfgtest/cache", "/cfgtest/cache"),
    ("sleep_model_refresh_enabled", "bool", "false", False),
    ("sleep_model_refresh_categories", "str", "model:user,model:workflow", "model:user,model:workflow"),
    ("sleep_model_refresh_max_tokens", "int", "2048", 2048),
    ("sleep_model_refresh_temperature", "float", "0.1", 0.1),
    ("sleep_model_refresh_auto_apply", "bool", "false", False),
    ("sleep_model_refresh_auto_apply_min_confidence", "float", "0.5", 0.5),
    ("sleep_model_refresh_min_evidence", "int", "7", 7),
    ("sleep_model_refresh_conflict_min_confidence", "float", "0.6", 0.6),
    ("sleep_model_refresh_conflict_min_evidence", "int", "8", 8),
    ("ignore_patterns", "str", "cfgtest-pattern-a\ncfgtest-pattern-b", ["cfgtest-pattern-a", "cfgtest-pattern-b"]),
    ("write_classifier", "str", "strict", "strict"),
    ("no_embeddings", "bool", "true", True),
    ("embeddings_via_api", "bool", "true", True),
    ("enhanced_recall", "bool", "true", True),
]

PROBE = r"""
import json, os
mods = {}
for m in set(mod for _, mod, _, _ in CONSUMERS) | set(mod for _, mod, _ in ACCESSORS):
    mods[m] = __import__(m, fromlist=["*"])
out = {}
for key, mod, attr, kind in CONSUMERS:
    val = getattr(mods[mod], attr)
    if kind == "set":
        val = sorted(val)
    elif kind == "path":
        val = str(val)
    out["const:" + key] = val
for key, mod, expr in ACCESSORS:
    out["acc:" + key] = eval(expr, mods[mod].__dict__)
print(json.dumps(out, sort_keys=True))
"""


def _probe(data_dir, extra_env=None, config_yaml=None):
    """Run the probe in a subprocess with an isolated data dir."""
    env = dict(os.environ)
    env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    # Never let a pre-existing OPENAI_API_KEY leak into the probe.
    env.pop("OPENAI_API_KEY", None)
    if extra_env:
        env.update(extra_env)
    if config_yaml is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "config.yaml").write_text(config_yaml)
    code = (
        "import json, sys\n"
        + "CONSUMERS = " + repr(CONSUMERS) + "\n"
        + "ACCESSORS = " + repr(ACCESSORS) + "\n"
        + PROBE
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-4000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture()
def data_dir(tmp_path):
    return tmp_path / "data"


# ---------------------------------------------------------------------------
# Phase 1: env-driven (fresh dir; env vars captured by the auto-seed)
# ---------------------------------------------------------------------------

def test_env_driven_constants(data_dir):
    extra_env = {ENV_MAP[key]: val for key, _, val, _ in TEST_VALUES}
    extra_env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    got = _probe(data_dir, extra_env=extra_env)
    consumer_keys = {key for key, _, _, _ in CONSUMERS}
    for key, kind, env_val, expected in TEST_VALUES:
        if key not in consumer_keys:
            continue  # accessor-only keys asserted below
        assert got["const:" + key] == expected, f"{key}: env={env_val!r} -> {got['const:'+key]!r} (want {expected!r})"
    # accessor keys
    assert got["acc:sleep_model_refresh_enabled"] is False
    assert got["acc:sleep_model_refresh_auto_apply"] is False
    assert got["acc:sleep_model_refresh_categories"] == "['model:user', 'model:workflow']"
    assert got["acc:sleep_model_refresh_auto_apply_min_confidence"] == 0.5
    assert got["acc:sleep_model_refresh_min_evidence"] == 7
    assert got["acc:sleep_model_refresh_conflict_min_confidence"] == 0.6
    assert got["acc:sleep_model_refresh_conflict_min_evidence"] == 8
    assert got["acc:write_classifier"] == "strict"
    assert got["acc:ignore_patterns"] == ["cfgtest-pattern-a", "cfgtest-pattern-b"]
    assert got["acc:no_embeddings"] is True
    assert got["acc:embeddings_via_api"] is True
    assert got["acc:enhanced_recall"] is True


# ---------------------------------------------------------------------------
# Phase 2: config-driven (pre-written config.yaml, no env vars)
# ---------------------------------------------------------------------------

def test_config_driven_constants(data_dir):
    import yaml as _yaml

    cfg: dict = {}
    for key, kind, env_val, expected in TEST_VALUES:
        if kind in ("int", "float", "bool"):
            cfg[key] = expected
        elif kind == "set":
            cfg[key] = "11111,22222"
        elif kind == "str":
            cfg[key] = env_val if isinstance(expected, list) else expected
    got = _probe(data_dir, config_yaml=_yaml.safe_dump(cfg))
    consumer_keys = {key for key, _, _, _ in CONSUMERS}
    for key, kind, env_val, expected in TEST_VALUES:
        if key not in consumer_keys:
            continue  # accessor-only keys asserted below
        assert got["const:" + key] == expected, (
            f"{key}: config-driven -> {got['const:'+key]!r} (want {expected!r})"
        )
    assert got["acc:write_classifier"] == "strict"
    assert got["acc:ignore_patterns"] == ["cfgtest-pattern-a", "cfgtest-pattern-b"]
    assert got["acc:sleep_model_refresh_min_evidence"] == 7
    assert got["acc:sleep_model_refresh_conflict_min_evidence"] == 8


# ---------------------------------------------------------------------------
# Phase 3: behavior-neutral (fresh dir, no env, no config -> code defaults)
# ---------------------------------------------------------------------------

def test_behavior_neutral_defaults(data_dir):
    got = _probe(data_dir)
    assert got["const:wm_max_items"] == 10000
    assert got["const:wm_ttl_hours"] == 168
    assert got["const:wm_bump_cap_hours"] == 24
    assert got["const:ep_limit"] == 50000
    assert got["const:sleep_batch"] == 5000
    assert got["const:sp_max"] == 1000
    assert got["const:recency_halflife"] == 168.0
    assert got["const:tier2_days"] == 30
    assert got["const:tier3_days"] == 180
    assert got["const:tier1_weight"] == 1.0
    assert got["const:tier2_weight"] == 0.5
    assert got["const:tier3_weight"] == 0.25
    assert got["const:degrade_batch"] == 100
    assert got["const:smart_compress"] is True
    assert got["const:tier3_max_chars"] == 300
    assert got["const:vec_type"] == "int8"
    assert got["const:embedding_dim"] == 384
    assert got["const:llm_enabled"] is True
    assert got["const:llm_max_tokens"] == 2048
    assert got["const:llm_n_threads"] == 4
    assert got["const:llm_n_ctx"] == 2048
    assert got["const:host_llm_enabled"] is False
    assert got["const:host_llm_n_ctx"] == 32000
    assert got["const:shmr_batch_size"] == 50
    assert got["const:shmr_max_iterations"] == 3
    assert got["const:shmr_similarity_threshold"] == 0.7
    assert got["const:shmr_harmony_threshold"] == 0.6
    assert got["const:shmr_min_cluster_size"] == 2
    assert got["const:shmr_temperature"] == 0.2
    assert got["const:llm_conflict_detection"] is False
    assert got["const:persona_interval"] == 50
    assert got["const:persona_daily_sync_hour"] == 3
    assert got["const:persona_token_cap"] == 1500
    assert got["const:embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert got["const:embedding_api_url"] == "https://openrouter.ai/api/v1"
    assert got["acc:sleep_model_refresh_enabled"] is True
    assert got["acc:sleep_model_refresh_auto_apply"] is True
    assert got["acc:sleep_model_refresh_auto_apply_min_confidence"] == 0.9
    assert got["acc:sleep_model_refresh_min_evidence"] == 2
    assert got["acc:sleep_model_refresh_conflict_min_confidence"] == 0.98
    assert got["acc:sleep_model_refresh_conflict_min_evidence"] == 3
    assert got["acc:write_classifier"] == "off"
    assert got["acc:no_embeddings"] is False
    assert got["acc:enhanced_recall"] is False


# ---------------------------------------------------------------------------
# REGRESSION: the exact reproduction from upstream issue #482
# ---------------------------------------------------------------------------

def test_issue482_reproduction(data_dir):
    """config.yaml `cross_session: true` + no env -> the recall gate honors it."""
    got = _probe(data_dir, config_yaml="cross_session: true\n")
    # The upstream repro asserted _cross_session_enabled() disagreed with
    # config. Post-fix (upstream #510 + this fork delta) they must agree.
    assert got["acc:cross_session"] is True
