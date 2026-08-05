# Mnemosyne MCP Server — hardened fork image
# Fork: christian-schulze/mnemosyne-docker (base: AxDSan/mnemosyne @33540d2, MIT)
#
# Fork deltas baked in:
#   - mcp pin dropped after the upstream sync: upstream migrated to the mcp 2.x
#     API (the 1.x list_tools/call_tool decorators were removed in 2.0) and
#     excludes the bad release, so pyproject's mcp>=2.0.0,!=2.1.0 is what ships
#     here — the old "mcp>=1.28,<2" spike pin would install a 1.x SDK this code
#     no longer runs on
#   - install from /src (repo build = v3.15.1, not PyPI 3.14.0) with
#     [mcp,embeddings] extras (sqlite-vec ships in the image)
#   - non-root runtime user mcp (uid 1000) — DB ownership on bind mounts
#     matches host --user $(id -u):$(id -g) (spike finding: root-owned DB ->
#     host "readonly database")
#   - baked bge-small-en-v1.5 ONNX embedding model (~65MB) via
#     FASTEMBED_CACHE_PATH — no runtime download
#   - serverInfo.version reports pkg version (3.15.1), not the mcp lib version
#
# Build requires fastembed-models/ in the build context (gitignored; populated
# from ~/.hermes/cache/fastembed before building — see stage 1 record).
# MiniCPM5-1B-Q4_K_M.gguf (host_llm) intentionally excluded.
#
# Build:
#   docker build -t mnemosyne-mcp .
#
# Run (stdio — for Claude Desktop, Cursor, etc.):
#   docker run -i --rm -v mnemosyne-data:/data mnemosyne-mcp
#
# Run (SSE — for web clients):
#   docker run -d --rm -p 8080:8080 \
#     -v mnemosyne-data:/data \
#     --env MNEMOSYNE_MCP_TOKEN="${MNEMOSYNE_MCP_TOKEN:?set MNEMOSYNE_MCP_TOKEN}" \
#     mnemosyne-mcp --transport sse --host 0.0.0.0 --port 8080
#
# Run (Streamable HTTP — native MCP http transport, single GET/POST/DELETE /mcp endpoint):
#   docker run -d --rm -p 8080:8080 \
#     -v mnemosyne-data:/data \
#     --env MNEMOSYNE_MCP_TOKEN="${MNEMOSYNE_MCP_TOKEN:?set MNEMOSYNE_MCP_TOKEN}" \
#     --env MNEMOSYNE_MCP_ALLOWED_HOSTS="${MNEMOSYNE_MCP_ALLOWED_HOSTS:?set MNEMOSYNE_MCP_ALLOWED_HOSTS}" \
#     mnemosyne-mcp --transport streamable-http --host 0.0.0.0 --port 8080
#
# With custom data directory:
#   docker run -i --rm \
#     -v /host/path/to/data:/data \
#     -e MNEMOSYNE_DATA_DIR=/data \
#     mnemosyne-mcp

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Mnemosyne MCP Server (fork)"
LABEL org.opencontainers.image.description="Universal memory layer MCP server — hardened multi-agent fork"
LABEL org.opencontainers.image.source="https://github.com/christian-schulze/mnemosyne-docker"
LABEL org.opencontainers.image.licenses="MIT"

# Repo build context (dockerignore keeps tests/scripts/docs/out)
COPY . /src

# Install the fork package + extras (pyproject now pins the mcp 2.x SDK itself)
RUN pip install --no-cache-dir "/src[mcp,embeddings]" && \
    rm -rf /src

# Non-root runtime user, uid 1000 (matches host uid for bind-mounted /data)
RUN useradd --create-home --shell /bin/bash --uid 1000 mcp
ENV HOME=/home/mcp

# Bake embedding model (fork delta #3)
ENV FASTEMBED_CACHE_PATH=/home/mcp/.cache/fastembed
COPY --chown=mcp:mcp fastembed-models/ /home/mcp/.cache/fastembed/

# Default data directory (overridable via MNEMOSYNE_DATA_DIR)
ENV MNEMOSYNE_DATA_DIR=/data
VOLUME /data

USER mcp

# Health check: exercise the memory layer (recall path incl. embeddings)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "from mnemosyne import recall; recall('health', top_k=1)" || exit 1

# Default: stdio transport (compose overrides CMD, not ENTRYPOINT)
ENTRYPOINT ["mnemosyne", "mcp"]
CMD []
