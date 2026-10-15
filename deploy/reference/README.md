# Reference artifacts from deltas dropped in the 2026-10-15 upstream sync

These files came from fork commits that upstream superseded. They are kept as
reference, **not collected by pytest and not part of the package** — nothing here
runs in CI.

    verify_config_env.py         from a5cb9ec "resolve module constants through central config" (#482)
    verify_config_probe.py       from a5cb9ec (the in-container probe the harness runs)
    test_config_env_binding.py   from a5cb9ec (418 lines) — the gate that commit was written against

    verify_identity_auth.py      from 46d1a54 "server-side token→author_id mapping for MCP SSE" (Stage 5)
    test_s5_identity_auth.py     from 46d1a54 — imports the fork's private `_AUTHOR_MAP_ENV`

Why the commits were dropped and what upstream does instead:

- **a5cb9ec** — upstream has the central config (`mnemosyne/core/config.py`
  `get_int/get_bool/get_str`, `mnemosyne/hermes_config.py:read_hermes_config_key`),
  and `beam.py` uses it, but `embeddings.py`, `local_llm.py` and
  `llm_conflict_detector.py` still read the environment directly. So the fork's
  refactor is only *partly* superseded: the fork's modules deliberately resolve
  module constants through the central config, and upstream's do not. Porting it
  means re-applying ~1,100 lines across modules upstream has rewritten; this
  harness is the gate for whoever does that.
- **46d1a54** — upstream's `mnemosyne/mcp_tools.py:185-205` maps the authenticated
  token to `author_id` and **rejects** a client-supplied `author_id` that disagrees
  with the authenticated identity ("omit author_id or match the …"). The fork
  **ignored** the spoofed value instead of erroring. Same property, stricter
  behaviour upstream, different private API — hence the parked test.

If a future upstream sync makes either behaviour unacceptable, port the delta and
delete the corresponding entry here.
