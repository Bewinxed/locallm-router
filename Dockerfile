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
# Only rebuilds when VLLM_VERSION pin or wheels/ changes.
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS python-deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"

ARG VLLM_CACHE_BUST=2026-02-28
RUN uv pip install "setuptools>=77.0.3,<81.0.0" wheel "setuptools-scm>=8.0"
COPY wheels/ /tmp/wheels/
RUN uv pip install --no-build-isolation --no-index --find-links /tmp/wheels \
      vllm vllm-omni hf_transfer aiohttp Pillow && rm -rf /tmp/wheels

RUN CUDA_PY=$(find /opt/venv -path "*/vllm/platforms/cuda.py" 2>/dev/null | head -1) && \
    if [ -n "$CUDA_PY" ]; then \
      sed -i 's/handles = \[pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids\]/return True/g' "$CUDA_PY"; \
    fi

# ═══════════════════════════════════════════════════════════════
# Stage 3: Dashboard build (SvelteKit)
# Only rebuilds when dashboard/ source changes.
# Completely independent of Python/wheels.
# ═══════════════════════════════════════════════════════════════
FROM oven/bun:latest AS dashboard-builder

WORKDIR /app/dashboard

# Deps first (cached until lockfile changes)
COPY dashboard/package.json dashboard/bun.lock ./
RUN bun install --frozen-lockfile

# Source + build
COPY dashboard/svelte.config.js dashboard/vite.config.ts dashboard/tsconfig.json dashboard/components.json ./
COPY dashboard/src ./src
COPY dashboard/static ./static
RUN bun run build

# ═══════════════════════════════════════════════════════════════
# Stage 4: Final runtime image
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl unzip git libnuma-dev pciutils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp binaries
COPY --from=llama-builder /out/llama-server /usr/local/bin/
COPY --from=llama-builder /out/llama-cli /usr/local/bin/
COPY --from=llama-builder /out/llama-gguf-split /usr/local/bin/

# Python venv with vLLM + torch + CLI entrypoints
COPY --from=python-deps /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

WORKDIR /app

# Pre-built dashboard (just the output, no node_modules)
COPY --from=dashboard-builder /app/dashboard/build ./dashboard/build
COPY --from=dashboard-builder /app/dashboard/package.json ./dashboard/

# App code (changes most often — last layer)
COPY src/ ./src/

ENV LLM_ROUTER_CONFIG="/app/config.json"
ENV HF_HOME="/models"
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

EXPOSE 30000 30001 30099

CMD ["bun", "run", "src/index.ts"]
