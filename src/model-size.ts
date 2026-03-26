/**
 * Model Size Estimation for GPU Memory Planning
 *
 * Estimates VRAM usage based on:
 * - Parameter count (from config or inferred from model name)
 * - Quantization type (fp16, fp8, int8, int4, etc.)
 * - Context length (affects KV cache size)
 * - GPU memory utilization setting
 */

export interface ModelSizeEstimate {
  /** Estimated parameter count in billions */
  parametersBillions: number;
  /** Estimated model weights size in GB */
  weightsGB: number;
  /** Estimated KV cache size in GB (at max context) */
  kvCacheGB: number;
  /** Total estimated VRAM in GB */
  totalVRAM: number;
  /** Confidence: 'exact' if params specified, 'inferred' if from name */
  confidence: "exact" | "inferred" | "unknown";
}

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

/** Activation memory overhead multiplier */
const ACTIVATION_OVERHEAD = 1.15;

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

/**
 * Get max model length from extraArgs.
 */
export function getMaxModelLen(extraArgs: string[] | undefined): number {
  if (extraArgs) {
    const lenIdx = extraArgs.indexOf("--max-model-len");
    if (lenIdx !== -1 && extraArgs[lenIdx + 1]) {
      return parseInt(extraArgs[lenIdx + 1]);
    }
  }
  return 4096; // Default
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

  // For MoE: activated params drive activation overhead; total params drive weights
  const activeParamsBillions = config.activatedParameterCount ?? paramsBillions;

  // Get quantization and calculate bytes per param
  const quantType = getQuantizationType(config.dtype, config.extraArgs);
  const bytesPerParam = BYTES_PER_PARAM[quantType] || BYTES_PER_PARAM.bfloat16;

  // Calculate weights size (all params must fit in VRAM)
  const weightsGB = (paramsBillions * 1e9 * bytesPerParam) / 1e9;

  // Calculate KV cache size (based on active params for MoE)
  const maxModelLen = getMaxModelLen(config.extraArgs);
  const kvClass = getKvCacheClass(activeParamsBillions);
  const kvCacheGB = (maxModelLen / 1024) * KV_CACHE_FACTOR[kvClass];

  // Activation overhead based on active params only (MoE benefit)
  const activationGB = (activeParamsBillions * 1e9 * bytesPerParam) / 1e9 * (ACTIVATION_OVERHEAD - 1);

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

/**
 * Check if a model can fit in available GPU memory.
 */
export function canFitInMemory(
  estimatedVRAM: number,
  availableVRAM: number,
  gpuMemoryUtil: number = 0.9
): boolean {
  const usableVRAM = availableVRAM * gpuMemoryUtil;
  return estimatedVRAM <= usableVRAM;
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
 * Query GPU memory status via nvidia-smi.
 */
export async function getGpuInfo(): Promise<GpuInfo[]> {
  try {
    const result = Bun.spawnSync({
      cmd: [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
      ],
    });

    if (result.exitCode !== 0) {
      console.error("[model-size] nvidia-smi failed");
      return [];
    }

    const output = result.stdout.toString().trim();
    const gpus: GpuInfo[] = [];

    for (const line of output.split("\n")) {
      const [index, name, total, used, free] = line.split(", ").map((s) => s.trim());
      gpus.push({
        index: parseInt(index),
        name,
        totalMemoryMB: parseInt(total),
        usedMemoryMB: parseInt(used),
        freeMemoryMB: parseInt(free),
      });
    }

    return gpus;
  } catch (err) {
    console.error("[model-size] Failed to query GPU info:", err);
    return [];
  }
}

/**
 * Format a size estimate for logging.
 */
export function formatEstimate(estimate: ModelSizeEstimate): string {
  return `${estimate.parametersBillions}B params, ~${estimate.totalVRAM}GB VRAM (${estimate.confidence})`;
}
