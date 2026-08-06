#!/usr/bin/env python3
"""Stage-4 in-container config probe (runs INSIDE the mnemosyne MCP image).

Prints a JSON report of the effective runtime configuration:

- ``cfg``: ``get_config().get(key)`` for every mapped config key (the 100%
  coverage claim at the config-system level).
- ``const``: module-level constants for every config key consumed at import
  time (the ~50 previously silently-ignored keys — the issue #482 fix).
- ``acc``: runtime accessor results for keys read at call sites.

The host harness ``verify_config_env.py`` sets every ``MNEMOSYNE_*`` env var
to a distinct value, runs this probe in a FRESH container with a FRESH data
dir, and asserts the report reflects the env values.

Usage (in container):
    python3 /verify/verify_config_probe.py
"""

import json
import os

from mnemosyne.core.config import ENV_VAR_MAP, get_config

# (config_key, module, attribute, kind) — module-level constants
CONSUMERS = [
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
    ("llm_enabled", "mnemosyne.core.local_llm", "LLM_ENABLED", "bool"),
    ("llm_max_tokens", "mnemosyne.core.local_llm", "LLM_MAX_TOKENS", "int"),
    ("llm_n_threads", "mnemosyne.core.local_llm", "LLM_N_THREADS", "int"),
    ("llm_n_ctx", "mnemosyne.core.local_llm", "LLM_N_CTX", "int"),
    ("llm_base_url", "mnemosyne.core.local_llm", "LLM_BASE_URL", "str"),
    ("llm_api_key", "mnemosyne.core.local_llm", "LLM_API_KEY", "str"),
    ("llm_model", "mnemosyne.core.local_llm", "LLM_REMOTE_MODEL", "str"),
    ("llm_timeout", "mnemosyne.core.local_llm", "LLM_TIMEOUT", "float"),
    ("llm_fallback_models", "mnemosyne.core.local_llm", "LLM_FALLBACK_MODELS", "list"),
    ("llm_fallback_base_url", "mnemosyne.core.local_llm", "LLM_FALLBACK_BASE_URL", "str"),
    ("llm_fallback_api_key", "mnemosyne.core.local_llm", "LLM_FALLBACK_API_KEY", "str"),
    ("host_llm_enabled", "mnemosyne.core.local_llm", "HOST_LLM_ENABLED", "bool"),
    ("host_llm_provider", "mnemosyne.core.local_llm", "HOST_LLM_PROVIDER", "str"),
    ("host_llm_model", "mnemosyne.core.local_llm", "HOST_LLM_MODEL", "str"),
    ("host_llm_n_ctx", "mnemosyne.core.local_llm", "HOST_LLM_N_CTX", "int"),
    ("sleep_prompt", "mnemosyne.core.local_llm", "SLEEP_PROMPT", "str"),
    ("shmr_batch_size", "mnemosyne.core.shmr", "SHMR_BATCH_SIZE", "int"),
    ("shmr_max_iterations", "mnemosyne.core.shmr", "SHMR_MAX_ITERATIONS", "int"),
    ("shmr_similarity_threshold", "mnemosyne.core.shmr", "SHMR_SIMILARITY_THRESHOLD", "float"),
    ("shmr_harmony_threshold", "mnemosyne.core.shmr", "SHMR_HARMONY_THRESHOLD", "float"),
    ("shmr_model", "mnemosyne.core.shmr", "SHMR_MODEL", "str"),
    ("shmr_min_cluster_size", "mnemosyne.core.shmr", "SHMR_MIN_CLUSTER_SIZE", "int"),
    ("shmr_temperature", "mnemosyne.core.shmr", "SHMR_TEMPERATURE", "float"),
    ("llm_conflict_detection", "mnemosyne.core.llm_conflict_detector", "LLM_CONFLICT_DETECTION_ENABLED", "bool"),
    ("conflict_llm_base_url", "mnemosyne.core.llm_conflict_detector", "CONFLICT_LLM_BASE_URL", "str"),
    ("conflict_llm_api_key", "mnemosyne.core.llm_conflict_detector", "CONFLICT_LLM_API_KEY", "str"),
    ("conflict_llm_model", "mnemosyne.core.llm_conflict_detector", "CONFLICT_LLM_MODEL", "str"),
    ("persona_interval", "mnemosyne.core.persona", "DEFAULT_INTERVAL", "int"),
    ("persona_daily_sync_hour", "mnemosyne.core.persona", "DEFAULT_DAILY_SYNC_HOUR", "int"),
    ("persona_token_cap", "mnemosyne.core.persona", "DEFAULT_TOKEN_CAP", "int"),
    ("embedding_model", "mnemosyne.core.embeddings", "_DEFAULT_MODEL", "str"),
    ("embedding_api_url", "mnemosyne.core.embeddings", "_OPENAI_BASE_URL", "str"),
    ("embedding_api_key", "mnemosyne.core.embeddings", "_OPENAI_API_KEY", "str"),
    ("fastembed_cache_dir", "mnemosyne.core.embeddings", "_FASTEMBED_CACHE_DIR", "str"),
    ("data_dir", "mnemosyne.core.beam", "_default_data_dir()", "path"),
]

# (config_key, module, expression) — call-site accessors
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


def main() -> int:
    mods = {}
    for m in set(mod for _, mod, _, _ in CONSUMERS) | set(mod for _, mod, _ in ACCESSORS):
        mods[m] = __import__(m, fromlist=["*"])

    report = {"cfg": {}, "const": {}, "acc": {}}

    for key in ENV_VAR_MAP:
        report["cfg"][key] = get_config().get(key)

    for key, mod, attr, kind in CONSUMERS:
        if attr.endswith("()"):
            val = eval(attr, mods[mod].__dict__)
        else:
            val = getattr(mods[mod], attr)
        if kind == "set":
            val = sorted(val)
        elif kind == "path":
            val = str(val)
        report["const"][key] = val

    for key, mod, expr in ACCESSORS:
        report["acc"][key] = eval(expr, mods[mod].__dict__)

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
