# Mnemosyne MCP Server — hardened fork image
# Fork: christian-schulze/mnemosyne-docker (base: AxDSan/mnemosyne @33540d2, MIT)
#
# Fork deltas baked in:
#   - mcp>=1.28,<2 pin (spike fix: bare [mcp] extra resolves mcp 2.0.0 ->
#     Server.list_tools AttributeError crash)
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

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Mnemosyne MCP Server (fork)"
LABEL org.opencontainers.image.description="Universal memory layer MCP server — hardened multi-agent fork"
LABEL org.opencontainers.image.source="https://github.com/christian-schulze/mnemosyne-docker"
LABEL org.opencontainers.image.licenses="MIT"

# Repo build context (dockerignore keeps tests/scripts/docs/out)
COPY . /src

# Pin the mcp SDK (spike-proven) and install the fork package + extras
RUN pip install --no-cache-dir "mcp>=1.28,<2" "/src[mcp,embeddings]" && \
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
