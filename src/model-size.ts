/**
 * Model Size Estimation for GPU Memory Planning
 *
 * Estimates are produced in trust order (see resolveEstimate): the configured
 * vramGB, then the GGUF header — read from disk, or over HTTP range requests
 * for a model that has not been downloaded yet — then a parameter-count
 * heuristic. Sizes are GiB throughout, matching what nvidia-smi reports.
 */

import { statSync } from "fs";
import { gguf, type GGUFParseOutput } from "@huggingface/gguf";
import type { ModelConfig } from "./config";
import { hfFileUrl, isLocalModel, resolveModelPath } from "./download";

export interface ModelSizeEstimate {
  /** Estimated parameter count in billions */
  parametersBillions: number;
  /** Estimated model weights size in GB */
  weightsGB: number;
  /** Estimated KV cache size in GB (at max context) */
  kvCacheGB: number;
  /** Total estimated VRAM in GB */
  totalVRAM: number;
  /** Which rung of the ladder supplied the number, most trusted first. */
  confidence: "manual" | "gguf-local" | "gguf-remote" | "exact" | "inferred" | "unknown";
}

const GIB = 1024 ** 3;

/** Bytes per parameter for different quantization types */
const BYTES_PER_PARAM: Record<string, number> = {
  fp32: 4,
  float32: 4,
  fp16: 2,
  float16: 2,
  bf16: 2,
  bfloat16: 2,
  fp8: 1,
  fp8_e4m3fn: 1,
  fp8_e5m2: 1,
  int8: 1,
  w8a8: 1,
  int4: 0.5,
  w4a16: 0.5,
  gptq: 0.5, // Typically 4-bit
  awq: 0.5, // Typically 4-bit
  gguf: 0.5, // Varies, assume 4-bit average
  "compressed-tensors": 0.5, // compressed-tensors: typically 4-bit
  bitsandbytes: 0.5, // bnb 4-bit
};

/** KV cache size factor per 1K context tokens (GB) - rough estimates by model size */
const KV_CACHE_FACTOR: Record<string, number> = {
  tiny: 0.02, // <3B params
  small: 0.05, // 3-8B params
  medium: 0.1, // 8-20B params
  large: 0.15, // 20-40B params
  xlarge: 0.25, // >40B params
};

/** Fixed CUDA/framework overhead in GB */
const CUDA_OVERHEAD_GB = 1.5;

/**
 * Compute buffers. Near-constant per backend, then a slice that follows the
 * context window — unlike the weights, which they don't scale with at all.
 */
const ACTIVATION_BASE_GB = 0.5;
const ACTIVATION_PER_1K_CTX_GB = 0.02;

/**
 * Bytes per element for the quantizations llama.cpp accepts in
 * --cache-type-k/v: ggml's type_size / block_size, scales and all. A quantized
 * KV cache is never as small as its name suggests — q8_0 stores a 16-bit scale
 * per 32-element block, so it costs 1.0625 bytes per element, not 1.
 */
const KV_CACHE_TYPE_BYTES: Record<string, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  q8_0: 1.0625,
  q5_1: 0.75,
  q5_0: 0.6875,
  q4_1: 0.625,
  q4_0: 0.5625,
  iq4_nl: 0.5625,
};

/**
 * Bytes per element for every ggml tensor type, by its GGUF type id:
 * type_size / block_size from ggml-common.h. K-quants pack 256 elements per
 * block, the rest 32.
 */
const GGML_TYPE_BYTES: Record<number, number> = {
  0: 4, // F32
  1: 2, // F16
  2: 0.5625, // Q4_0
  3: 0.625, // Q4_1
  6: 0.6875, // Q5_0
  7: 0.75, // Q5_1
  8: 1.0625, // Q8_0
  9: 1.125, // Q8_1
  10: 0.328125, // Q2_K
  11: 0.4296875, // Q3_K
  12: 0.5625, // Q4_K
  13: 0.6875, // Q5_K
  14: 0.8203125, // Q6_K
  15: 1.140625, // Q8_K
  16: 0.2578125, // IQ2_XXS
  17: 0.2890625, // IQ2_XS
  18: 0.3828125, // IQ3_XXS
  19: 0.1953125, // IQ1_S
  20: 0.5625, // IQ4_NL
  21: 0.4296875, // IQ3_S
  22: 0.3203125, // IQ2_S
  23: 0.53125, // IQ4_XS
  24: 1, // I8
  25: 2, // I16
  26: 4, // I32
  27: 8, // I64
  28: 8, // F64
  29: 0.21875, // IQ1_M
  30: 2, // BF16
  34: 0.2109375, // TQ1_0
  35: 0.2578125, // TQ2_0
  39: 0.53125, // MXFP4
};

/**
 * Infer parameter count from model path/name.
 * Handles common patterns like:
 * - "Qwen2.5-14B-Instruct" -> 14
 * - "Mistral-Small-3.1-24B" -> 24
 * - "Ministral-3-14B" -> 14
 * - "llama-7b" -> 7
 * - "phi-3-mini-4k" -> 3.8 (mini = 3.8B for phi)
 */
export function inferParameterCount(modelPath: string): number | null {
  const name = modelPath.toLowerCase();

  // Direct patterns: "14B", "7b", "24B", etc.
  const directMatch = name.match(/(\d+(?:\.\d+)?)\s*b(?:illion)?(?:-|_|$)/i);
  if (directMatch) {
    return parseFloat(directMatch[1]);
  }

  // Patterns like "14b-instruct"
  const suffixMatch = name.match(/[_-](\d+(?:\.\d+)?)b[_-]/i);
  if (suffixMatch) {
    return parseFloat(suffixMatch[1]);
  }

  // Known model families with size conventions
  const knownSizes: Record<string, number> = {
    "phi-3-mini": 3.8,
    "phi-3-small": 7,
    "phi-3-medium": 14,
    "gemma-2b": 2,
    "gemma-7b": 7,
    "mistral-7b": 7,
    "mixtral-8x7b": 47, // MoE, effective params
    "mixtral-8x22b": 141,
    "llama-2-7b": 7,
    "llama-2-13b": 13,
    "llama-2-70b": 70,
    "llama-3-8b": 8,
    "llama-3-70b": 70,
    "qwen-vl-3b": 3,
    "qwen-vl-7b": 7,
    translategemma: 4, // TranslateGemma is 4B
  };

  for (const [pattern, size] of Object.entries(knownSizes)) {
    if (name.includes(pattern.toLowerCase())) {
      return size;
    }
  }

  return null;
}

/**
 * Get the quantization type from extraArgs or dtype.
 */
export function getQuantizationType(
  dtype: string | undefined,
  extraArgs: string[] | undefined
): string {
  // Check extraArgs for --quantization flag
  if (extraArgs) {
    const quantIdx = extraArgs.indexOf("--quantization");
    if (quantIdx !== -1 && extraArgs[quantIdx + 1]) {
      return extraArgs[quantIdx + 1].toLowerCase();
    }
  }

  // Fall back to dtype
  return (dtype || "bfloat16").toLowerCase();
}

/** Read the value that follows a flag in extraArgs. */
export function argValue(extraArgs: string[] | undefined, ...flags: string[]): string | null {
  if (!extraArgs) return null;
  for (const flag of flags) {
    const idx = extraArgs.indexOf(flag);
    if (idx !== -1 && extraArgs[idx + 1]) return extraArgs[idx + 1];
  }
  return null;
}

/**
 * Get max model length from extraArgs. `--max-model-len` is vLLM/SGLang,
 * `-c`/`--ctx-size` is llama.cpp.
 */
export function getMaxModelLen(extraArgs: string[] | undefined): number {
  const value = argValue(extraArgs, "--max-model-len", "-c", "--ctx-size");
  const parsed = value ? parseInt(value) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 4096;
}

/** Average bytes per KV cache element, from llama.cpp's --cache-type-k/v. */
function getKvCacheTypeBytes(extraArgs: string[] | undefined): number {
  const k = argValue(extraArgs, "--cache-type-k", "-ctk");
  const v = argValue(extraArgs, "--cache-type-v", "-ctv");
  const bytes = (type: string | null) =>
    (type && KV_CACHE_TYPE_BYTES[type.toLowerCase()]) || KV_CACHE_TYPE_BYTES.f16;
  return (bytes(k) + bytes(v)) / 2;
}

/**
 * Weights size for a GGUF model: the file on disk is the answer, no
 * quantization guessing needed. Null when the file isn't downloaded yet.
 */
function ggufWeightsGB(config: { modelPath: string; modelFile?: string }): number | null {
  try {
    return statSync(resolveModelPath(config)).size / GIB;
  } catch {
    return null;
  }
}

/** Positive integer value of a flag, or `fallback` when absent or unparseable. */
function intArg(extraArgs: string[] | undefined, fallback: number, ...flags: string[]): number {
  const parsed = parseInt(argValue(extraArgs, ...flags) ?? "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

/** FreeToken's own default for --memory-ratio. */
const FREETOKEN_MEMORY_RATIO = 0.9;

/** Card size assumed when the driver won't name one, as gpu-lock assumes too. */
const UNKNOWN_CARD_TOTAL_GB = 24;

/** Share of a card freetoken will fill, from its --memory-ratio flag. */
function getMemoryRatio(extraArgs: string[] | undefined): number {
  const ratio = parseFloat(argValue(extraArgs, "--memory-ratio") ?? "");
  return Number.isFinite(ratio) && ratio > 0 ? ratio : FREETOKEN_MEMORY_RATIO;
}

/** What a freetoken model occupies on one card: --memory-ratio of it, no more. */
function freetokenVRAM(extraArgs: string[] | undefined, cardTotalGB: number): number {
  return Math.round(cardTotalGB * getMemoryRatio(extraArgs) * 100) / 100;
}

/**
 * Get GPU memory utilization from extraArgs.
 */
export function getGpuMemoryUtil(extraArgs: string[] | undefined): number {
  if (extraArgs) {
    const utilIdx = extraArgs.indexOf("--gpu-memory-utilization");
    if (utilIdx !== -1 && extraArgs[utilIdx + 1]) {
      return parseFloat(extraArgs[utilIdx + 1]);
    }
  }
  return 0.9; // Default
}

/**
 * Get KV cache size class based on parameter count.
 */
function getKvCacheClass(paramsBillions: number): keyof typeof KV_CACHE_FACTOR {
  if (paramsBillions < 3) return "tiny";
  if (paramsBillions < 8) return "small";
  if (paramsBillions < 20) return "medium";
  if (paramsBillions < 40) return "large";
  return "xlarge";
}

/**
 * Estimate VRAM usage for a model configuration.
 *
 * For MoE models, supply `activatedParameterCount` to get accurate activation
 * memory estimates. All weights still load into VRAM, but activation overhead
 * is computed from active params only.
 */
export function estimateVRAM(config: {
  modelPath: string;
  modelFile?: string;
  backend?: string;
  parameterCount?: number;
  activatedParameterCount?: number;
  dtype?: string;
  extraArgs?: string[];
}): ModelSizeEstimate {
  // Determine parameter count (total, for weights)
  let paramsBillions: number;
  let confidence: ModelSizeEstimate["confidence"];

  if (config.parameterCount) {
    paramsBillions = config.parameterCount;
    confidence = "exact";
  } else {
    const inferred = inferParameterCount(config.modelPath);
    if (inferred) {
      paramsBillions = inferred;
      confidence = "inferred";
    } else {
      // Unknown - assume medium size (14B) as safe default
      paramsBillions = 14;
      confidence = "unknown";
    }
  }

  // FreeToken's weights never reach the card: an offloaded MoE keeps them in
  // host RAM and caches only the hot experts here, up to --memory-ratio of the
  // GPU. Sizing it from its parameters the way everything below does would put
  // a 284B model's 140GB on a 24GB card's books and leave it unschedulable for
  // good. resolveEstimate refines the card size; this is the blind guess.
  if (config.backend === "freetoken") {
    return {
      parametersBillions: paramsBillions,
      weightsGB: 0,
      kvCacheGB: 0,
      totalVRAM: freetokenVRAM(config.extraArgs, UNKNOWN_CARD_TOTAL_GB),
      confidence: "inferred",
    };
  }

  // For MoE: activated params drive activation overhead; total params drive weights
  const activeParamsBillions = config.activatedParameterCount ?? paramsBillions;

  // Get quantization and calculate bytes per param
  const quantType = getQuantizationType(config.dtype, config.extraArgs);
  const bytesPerParam = BYTES_PER_PARAM[quantType] || BYTES_PER_PARAM.bfloat16;

  // Weights: the GGUF on disk beats any parameter-count guess.
  const ggufWeights = config.backend === "llama-server" ? ggufWeightsGB(config) : null;
  const weightsGB = ggufWeights ?? (paramsBillions * 1e9 * bytesPerParam) / 1e9;
  if (ggufWeights !== null) confidence = "exact";

  // KV cache scales with the context window and how it's quantized
  const contextTokens = getMaxModelLen(config.extraArgs);
  const kvClass = getKvCacheClass(activeParamsBillions);
  const kvBytesFactor = getKvCacheTypeBytes(config.extraArgs) / KV_CACHE_TYPE_BYTES.f16;
  const kvCacheGB = (contextTokens / 1024) * KV_CACHE_FACTOR[kvClass] * kvBytesFactor;

  // Compute buffers: roughly constant, plus a context-driven slice
  const activationGB = ACTIVATION_BASE_GB + (contextTokens / 1024) * ACTIVATION_PER_1K_CTX_GB;

  // Total
  const totalVRAM = weightsGB + kvCacheGB + activationGB + CUDA_OVERHEAD_GB;

  return {
    parametersBillions: paramsBillions,
    weightsGB: Math.round(weightsGB * 100) / 100,
    kvCacheGB: Math.round(kvCacheGB * 100) / 100,
    totalVRAM: Math.round(totalVRAM * 100) / 100,
    confidence,
  };
}

// ─── GGUF header estimation ─────────────────────────────────────

/** Everything the GGUF header tells us that bears on VRAM, backend flags aside. */
interface GgufShape {
  weightsBytes: number;
  /** KV elements per token, summed over the layers that actually have a cache. */
  kvElementsPerToken: number;
  /** Recurrent (SSM) state per sequence — constant, unlike the KV cache. */
  recurrentBytesPerSeq: number;
  embeddingLength: number;
  parametersBillions: number;
}

const GGUF_HTTP_TIMEOUT_MS = 20_000;

/** Element size of the safetensors dtypes the HuggingFace API reports. */
const SAFETENSORS_DTYPE_BYTES: Record<string, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  F8_E4M3: 1,
  F8_E5M2: 1,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  BOOL: 1,
};
/** Constant slice of the compute buffers: graph scratch, CUDA/Vulkan context. */
const COMPUTE_BUFFER_BASE_GB = 0.5;
/** Activation tensors of ubatch × embedding_length in flight at once, in f32. */
const COMPUTE_BUFFER_TENSORS = 8;

/** Parsed headers, keyed by file path + mtime (local) or URL (remote). */
const ggufShapeCache = new Map<string, Promise<GgufShape | null>>();

function ggufNumber(md: Record<string, unknown>, key: string): number | undefined {
  const value = md[key];
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  return undefined;
}

function ggufShape(parsed: GGUFParseOutput): GgufShape {
  const md = parsed.metadata as unknown as Record<string, unknown>;
  const arch = String(md["general.architecture"] ?? "");
  const num = (key: string) => ggufNumber(md, `${arch}.${key}`);

  const blocks = num("block_count") ?? 0;
  const embed = num("embedding_length") ?? 0;
  const heads = num("attention.head_count") ?? 0;
  const headDim = num("attention.key_length") ?? (heads > 0 ? embed / heads : 0);
  const valueDim = num("attention.value_length") ?? headDim;

  // Hybrid stacks give a KV cache only to their full-attention layers; the rest
  // carry a recurrent state instead. Counting all of them as attention layers
  // overstates the cache by the ratio between the two (4x on Qwen3.5-style
  // models). GGUFs express the split either per layer or as an interval.
  const perLayerKvHeads = md[`${arch}.attention.head_count_kv`];
  const interval = num("full_attention_interval") ?? 0;
  const attentionLayers = interval > 1 ? Math.floor(blocks / interval) : blocks;
  const kvHeads = Array.isArray(perLayerKvHeads)
    ? perLayerKvHeads.reduce((sum: number, h) => sum + Number(h), 0)
    : attentionLayers * (num("attention.head_count_kv") ?? heads);

  const convKernel = num("ssm.conv_kernel") ?? 0;
  const innerSize = num("ssm.inner_size") ?? 0;
  const stateSize = num("ssm.state_size") ?? 0;
  const groups = num("ssm.group_count") ?? 1;
  const recurrentLayers = blocks - attentionLayers;
  const recurrentBytesPerSeq =
    recurrentLayers > 0 && innerSize > 0
      ? recurrentLayers *
        ((convKernel - 1) * (innerSize + 2 * groups * stateSize) + innerSize * stateSize) *
        4
      : 0;

  let weightsBytes = 0;
  let elements = 0;
  for (const tensor of parsed.tensorInfos) {
    const numel = Number(tensor.shape.reduce((a, b) => a * b, 1n));
    elements += numel;
    // An unknown type is a new quantization we have no size for; a byte per
    // element keeps the total in the right order of magnitude.
    weightsBytes += numel * (GGML_TYPE_BYTES[tensor.dtype] ?? 1);
  }

  return {
    weightsBytes,
    kvElementsPerToken: kvHeads * (headDim + valueDim),
    recurrentBytesPerSeq,
    embeddingLength: embed,
    parametersBillions: elements / 1e9,
  };
}

/** fetch with a deadline, so a stalled range request can't hold up a start. */
const timedFetch = Object.assign(
  (input: RequestInfo | URL, init?: RequestInit) =>
    fetch(input, { ...init, signal: AbortSignal.timeout(GGUF_HTTP_TIMEOUT_MS) }),
  { preconnect: fetch.preconnect }
);

/**
 * Read a GGUF header. Local files are read directly; remote ones cost a few
 * range requests against the repo instead of the whole multi-GB download.
 */
function readGgufShape(uri: string, allowLocalFile: boolean): Promise<GgufShape | null> {
  const token = process.env.HF_TOKEN;
  return gguf(uri, {
    allowLocalFile,
    additionalFetchHeaders: token ? { Authorization: `Bearer ${token}` } : {},
    fetch: timedFetch,
  })
    .then(ggufShape)
    .catch((err) => {
      console.warn(`[model-size] Could not parse GGUF header at ${uri}: ${err}`);
      return null;
    });
}

function cachedGgufShape(uri: string, allowLocalFile: boolean): Promise<GgufShape | null> {
  let key = uri;
  if (allowLocalFile) {
    try {
      key = `${uri}:${statSync(uri).mtimeMs}`;
    } catch {
      return Promise.resolve(null);
    }
  }
  let shape = ggufShapeCache.get(key);
  if (!shape) {
    shape = readGgufShape(uri, allowLocalFile);
    ggufShapeCache.set(key, shape);
  }
  return shape;
}

/** Turn a parsed header into an estimate for the flags this model runs with. */
function ggufEstimate(
  shape: GgufShape,
  config: ModelConfig,
  confidence: ModelSizeEstimate["confidence"]
): ModelSizeEstimate {
  const contextTokens = getMaxModelLen(config.extraArgs);
  const slots = intArg(config.extraArgs, 1, "--parallel", "-np");
  const ubatch = intArg(config.extraArgs, 512, "--ubatch-size", "-ub");

  const weightsGB = shape.weightsBytes / GIB;
  const kvCacheGB =
    (shape.kvElementsPerToken * getKvCacheTypeBytes(config.extraArgs) * contextTokens * slots +
      shape.recurrentBytesPerSeq * slots) /
    GIB;
  const computeGB =
    COMPUTE_BUFFER_BASE_GB + (ubatch * shape.embeddingLength * 4 * COMPUTE_BUFFER_TENSORS) / GIB;

  return {
    parametersBillions: Math.round(shape.parametersBillions * 100) / 100,
    weightsGB: Math.round(weightsGB * 100) / 100,
    kvCacheGB: Math.round(kvCacheGB * 100) / 100,
    totalVRAM: Math.round((weightsGB + kvCacheGB + computeGB) * 100) / 100,
    confidence,
  };
}

/**
 * VRAM this model needs, from the best source available: the configured
 * vramGB, then the GGUF header on disk, then the same header fetched from
 * HuggingFace when the file isn't downloaded yet, and finally the
 * parameter-count heuristic. The remote step is what lets admission run before
 * a download. A freetoken model skips the ladder below vramGB entirely: what it
 * holds follows the card it runs on, not the weights it serves.
 *
 * vramGB outranks the parsers because it is a human overruling them: it gets
 * set precisely when an automatic estimate came out wrong, so letting an
 * estimate win would ignore the correction it exists to apply. The one thing
 * that does outrank it is a measurement, and that is applied a rung above this
 * function — in the scheduler's footprint(), which prefers what the driver saw
 * this model take over anything predicted here.
 */
export async function resolveEstimate(config: ModelConfig): Promise<ModelSizeEstimate> {
  if (config.vramGB !== undefined) {
    return { ...estimateVRAM(config), totalVRAM: config.vramGB, confidence: "manual" };
  }

  // A freetoken model is sized by the card it runs on rather than by its own
  // weights (see estimateVRAM); the driver is the only place that card's size
  // comes from. When it can't say, the 24GB assumption made there stands.
  if (config.backend === "freetoken") {
    const gpu = config.gpus?.[0] ?? config.gpu;
    const info = (await getGpuInfo()).find((g) => g.index === gpu);
    if (!info) return estimateVRAM(config);
    return {
      ...estimateVRAM(config),
      totalVRAM: freetokenVRAM(config.extraArgs, info.totalMemoryMB / 1024),
    };
  }

  if (config.backend === "llama-server") {
    const localPath = resolveModelPath(config);
    const local = await cachedGgufShape(localPath, true);
    if (local) return ggufEstimate(local, config, "gguf-local");

    if (!isLocalModel(config.modelPath) && config.modelFile) {
      const remote = await cachedGgufShape(hfFileUrl(config.modelPath, config.modelFile), false);
      if (remote) return ggufEstimate(remote, config, "gguf-remote");
    }
  }

  return estimateVRAM(config);
}

/** Repo weight sizes already fetched, keyed by repo id. */
const repoWeightsCache = new Map<string, Promise<number | null>>();

/**
 * Weights alone for a HuggingFace repo, from the safetensors index the API
 * already publishes. Null when the repo doesn't report one.
 */
export function fetchRepoWeightsGB(config: ModelConfig): Promise<number | null> {
  let pending = repoWeightsCache.get(config.modelPath);
  if (!pending) {
    pending = readRepoWeightsGB(config);
    repoWeightsCache.set(config.modelPath, pending);
  }
  return pending;
}

async function readRepoWeightsGB(config: ModelConfig): Promise<number | null> {
  if (isLocalModel(config.modelPath)) return null;
  try {
    const token = process.env.HF_TOKEN;
    const r = await fetch(`https://huggingface.co/api/models/${config.modelPath}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      signal: AbortSignal.timeout(GGUF_HTTP_TIMEOUT_MS),
    });
    if (!r.ok) return null;
    const info = (await r.json()) as {
      safetensors?: { parameters?: Record<string, number> };
      config?: { quantization_config?: { bits?: number } };
    };
    const parameters = info.safetensors?.parameters;
    if (!parameters) return null;

    // Quantized checkpoints report their packed weights under an integer
    // dtype; what those cost is the quantization's bit width, not the
    // container's. Float buckets (norms, embeddings, unquantized modules) are
    // stored at face value.
    const quantBits = info.config?.quantization_config?.bits;
    let bytes = 0;
    for (const [dtype, count] of Object.entries(parameters)) {
      const elementBytes = SAFETENSORS_DTYPE_BYTES[dtype.toUpperCase()] ?? 2;
      const isFloat = /^(F|BF)\d/.test(dtype.toUpperCase());
      bytes += count * (!isFloat && quantBits ? quantBits / 8 : elementBytes);
    }
    return bytes / GIB;
  } catch {
    return null;
  }
}

/**
 * GPU memory info from nvidia-smi.
 */
export interface GpuInfo {
  index: number;
  name: string;
  totalMemoryMB: number;
  usedMemoryMB: number;
  freeMemoryMB: number;
}

/**
 * nvidia-smi normally answers in milliseconds, but a contended driver can leave
 * it blocked in the kernel for as long as it likes. This call sits inside the
 * scheduler's admission path, where a read that never returns is a start that
 * spawns nothing, logs nothing and never fails — so it gets a deadline.
 */
const NVIDIA_SMI_TIMEOUT_MS = 10_000;

/** Run an nvidia-smi query, returning its CSV rows already split by column. */
async function nvidiaSmi(query: string): Promise<string[][]> {
  try {
    const proc = Bun.spawn({
      cmd: ["nvidia-smi", query, "--format=csv,noheader,nounits"],
      stdout: "pipe",
      stderr: "ignore",
    });

    let deadline: Timer | undefined;
    const answered = await Promise.race([
      Promise.all([new Response(proc.stdout).text(), proc.exited]),
      new Promise<null>((resolve) => {
        deadline = setTimeout(resolve, NVIDIA_SMI_TIMEOUT_MS, null);
      }),
    ]).finally(() => clearTimeout(deadline));

    if (!answered) {
      // Best effort: something stuck in the driver sits in uninterruptible
      // sleep and leaves only when the kernel lets it, SIGKILL or not.
      proc.kill("SIGKILL");
      console.error(
        `[model-size] nvidia-smi ${query} did not answer within ` +
          `${NVIDIA_SMI_TIMEOUT_MS / 1000}s — giving up on this reading`
      );
      return [];
    }

    const [stdout, exitCode] = answered;
    if (exitCode !== 0) {
      console.error(`[model-size] nvidia-smi ${query} failed`);
      return [];
    }

    return stdout
      .trim()
      .split("\n")
      .filter((line) => line.trim().length > 0)
      .map((line) => line.split(",").map((cell) => cell.trim()));
  } catch (err) {
    console.error("[model-size] Failed to run nvidia-smi:", err);
    return [];
  }
}

/**
 * Query GPU memory status via nvidia-smi.
 */
export async function getGpuInfo(): Promise<GpuInfo[]> {
  const rows = await nvidiaSmi("--query-gpu=index,name,memory.total,memory.used,memory.free");
  return rows.map(([index, name, total, used, free]) => ({
    index: parseInt(index),
    name,
    totalMemoryMB: parseInt(total),
    usedMemoryMB: parseInt(used),
    freeMemoryMB: parseInt(free),
  }));
}

/** One process's memory on one GPU, as the driver accounts for it. */
export interface ComputeApp {
  gpu: number;
  pid: number;
  usedMemoryMB: number;
}

/** GPU index per UUID. Fixed for the life of the driver, so read once. */
let gpuIndexByUuid: Promise<Map<string, number>> | null = null;

function readGpuIndexByUuid(): Promise<Map<string, number>> {
  if (!gpuIndexByUuid) {
    gpuIndexByUuid = nvidiaSmi("--query-gpu=index,uuid").then(
      (rows) => new Map(rows.map(([index, uuid]) => [uuid, parseInt(index)]))
    );
  }
  return gpuIndexByUuid;
}

/**
 * Per-process GPU memory. This is CUDA accounting: processes using the GPU
 * through another API — a Vulkan build of llama.cpp, say — hold real memory
 * without ever showing up here.
 */
export async function getComputeApps(): Promise<ComputeApp[]> {
  const [rows, indexByUuid] = await Promise.all([
    nvidiaSmi("--query-compute-apps=gpu_uuid,pid,used_memory"),
    readGpuIndexByUuid(),
  ]);
  const apps: ComputeApp[] = [];
  for (const [uuid, pid, used] of rows) {
    const gpu = indexByUuid.get(uuid);
    if (gpu === undefined) continue;
    apps.push({ gpu, pid: parseInt(pid), usedMemoryMB: parseInt(used) });
  }
  return apps;
}

/**
 * Format a size estimate for logging.
 */
export function formatEstimate(estimate: ModelSizeEstimate): string {
  return `${estimate.parametersBillions}B params, ~${estimate.totalVRAM}GB VRAM (${estimate.confidence})`;
}
