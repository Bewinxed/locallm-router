# ═══════════════════════════════════════════════════════════════
# Stage 1: Build llama.cpp from source
# Only rebuilds when LLAMA_CPP_VERSION changes.
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG LLAMA_CPP_VERSION=b8180
RUN mkdir -p /out && \
    git clone --depth 1 --branch ${LLAMA_CPP_VERSION} https://github.com/ggml-org/llama.cpp /tmp/llama.cpp && \
    cmake /tmp/llama.cpp -B /tmp/llama.cpp/build \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES="89" \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/llama.cpp/build --config Release -j$(nproc) \
      --target llama-server llama-cli llama-gguf-split && \
    cp /tmp/llama.cpp/build/bin/llama-server /out/ && \
    cp /tmp/llama.cpp/build/bin/llama-cli /out/ && \
    cp /tmp/llama.cpp/build/bin/llama-gguf-split /out/ && \
    chmod +x /out/*

# ═══════════════════════════════════════════════════════════════
# Stage 2: Python deps (vLLM + torch ~15GB)
# Only rebuilds when VLLM_VERSION pin changes.
# Separated so app code changes never re-download torch.
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS python-deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy UV_BREAK_SYSTEM_PACKAGES=1

# Bump this to force re-download of vLLM nightly (otherwise layer stays cached)
ARG VLLM_CACHE_BUST=2026-02-28
# All deps pre-downloaded locally (WSL2 mirrored networking drops sustained downloads)
COPY wheels/ /tmp/wheels/
RUN uv pip install --no-index --find-links /tmp/wheels vllm hf_transfer aiohttp Pillow && rm -rf /tmp/wheels
RUN uv pip install git+https://github.com/vllm-project/vllm-omni.git --upgrade

# Bypass vLLM P2P check for consumer GPUs (WSL2)
RUN CUDA_PY=$(find /usr/local/lib -path "*/vllm/platforms/cuda.py" 2>/dev/null | head -1) && \
    if [ -n "$CUDA_PY" ]; then \
      sed -i 's/handles = \[pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids\]/return True/g' "$CUDA_PY"; \
    fi

# ═══════════════════════════════════════════════════════════════
# Stage 3: Final runtime image
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl unzip git libnuma-dev pciutils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp binaries (cached in stage 1)
COPY --from=llama-builder /out/llama-server /usr/local/bin/
COPY --from=llama-builder /out/llama-cli /usr/local/bin/
COPY --from=llama-builder /out/llama-gguf-split /usr/local/bin/

# Python site-packages with vLLM + torch (cached in stage 2)
COPY --from=python-deps /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# uv (for any future runtime pip needs)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy UV_BREAK_SYSTEM_PACKAGES=1

# Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

WORKDIR /app

# ─── Dashboard deps (cached until lockfile changes) ───────────
COPY dashboard/package.json dashboard/bun.lock ./dashboard/
WORKDIR /app/dashboard
RUN bun install --frozen-lockfile
WORKDIR /app

# ─── Dashboard config + source + build ────────────────────────
COPY dashboard/svelte.config.js dashboard/vite.config.ts dashboard/tsconfig.json dashboard/components.json ./dashboard/
COPY dashboard/src ./dashboard/src
COPY dashboard/static ./dashboard/static
WORKDIR /app/dashboard
RUN bun run build
WORKDIR /app

# ─── App code (changes most often — last layer) ──────────────
COPY src/ ./src/

ENV LLM_ROUTER_CONFIG="/app/config.json"
ENV HF_HOME="/models"
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

EXPOSE 30000 30001 30099

CMD ["bun", "run", "src/index.ts"]
