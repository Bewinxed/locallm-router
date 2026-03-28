/**
 * locallm-router Configuration
 *
 * Single-port architecture: one proxy serves all models,
 * routing by the `model` field in the request payload.
 */

export interface ModelConfig {
  /** Unique name for this model (used as the model ID in API requests) */
  name: string;
  /** HuggingFace repo ID (for vllm/sglang) or absolute path to GGUF file (for llama-server) */
  modelPath: string;
  /** Backend: "vllm" (default), "llama-server", "sglang", or "kestrel" */
  backend?: "vllm" | "llama-server" | "sglang" | "kestrel";
  /** GPU device index (single GPU) */
  gpu: number;
  /** GPU device indices for multi-GPU (tensor parallel). Overrides gpu if set. */
  gpus?: number[];
  /** Idle timeout in ms before killing process */
  idleTimeout: number;
  /** Additional args passed to the backend */
  extraArgs?: string[];
  /** Dtype (default: bfloat16, vllm/sglang only) */
  dtype?: string;
  /** Startup timeout in ms (default from manager config) */
  startupTimeout?: number;
  /** Model parameter count in billions (optional, for VRAM estimation) */
  parameterCount?: number;
  /** For MoE models: activated parameter count in billions per forward pass */
  activatedParameterCount?: number;
  /** Manual VRAM override in GB (total across all GPUs). Bypasses automatic estimation. */
  vramGB?: number;
  /** Eviction priority (default: 0). Higher = harder to evict. Models with equal priority use LRU. */
  priority?: number;
  /** Delay in ms after idle timeout before sleeping/stopping. 0 = immediate. Default: 0 */
  sleepDelay?: number;
  /** Aliases: additional model names that route to this model */
  aliases?: string[];
  /**
   * Balance mode: list of GPU indices to spawn independent instances on.
   * Enables `:balance` (auto-route to idle), `:gpu0`/`:gpu1` (pin to GPU).
   * Each GPU gets its own llama-server process. The primary `gpu` field is
   * still the default when no modifier is used.
   */
  balanceGpus?: number[];
  /**
   * Per-GPU extra arg overrides for balance mode.
   * Keys are GPU indices (as strings). Values are extra args arrays that
   * REPLACE the base extraArgs for that GPU's instance.
   * If not specified, the base extraArgs are used for all instances.
   */
  balanceGpuArgs?: Record<string, string[]>;
}

export interface ManagerConfig {
  models: ModelConfig[];
  /** Single external port for the unified OpenAI-compatible proxy */
  port: number;
  /** Management API port */
  managementPort: number;
  /** Health check interval in ms */
  healthCheckInterval: number;
  /** Max startup wait time in ms */
  startupTimeout: number;
}

const INTERNAL_PORT_BASE = 40000;

/** Get the auto-assigned internal port for a model by its index */
export function getInternalPort(modelIndex: number): number {
  return INTERNAL_PORT_BASE + modelIndex;
}

export const defaultConfig: ManagerConfig = {
  models: [],
  port: 30000,
  managementPort: 30099,
  healthCheckInterval: 5000,
  startupTimeout: 1800000,
};

import { readFileSync } from "fs";

/** Resolve the config file path from environment */
function getConfigPath(): string | null {
  return process.env.LLM_ROUTER_CONFIG || null;
}

function readConfigFile(configPath: string): ManagerConfig {
  const raw = readFileSync(configPath, "utf-8");
  const loaded = JSON.parse(raw) as Partial<ManagerConfig>;
  return { ...defaultConfig, ...loaded };
}

/** Apply post-load defaults (startup timeout, idle timeout overrides) */
function applyDefaults(config: ManagerConfig): ManagerConfig {
  // Apply manager-level startupTimeout to models that don't have their own
  for (const model of config.models) {
    if (!model.startupTimeout) {
      model.startupTimeout = config.startupTimeout;
    }
  }

  // Override idle timeout from env var (in minutes)
  const envIdleTimeout = process.env.LLM_IDLE_TIMEOUT_MINUTES;
  if (envIdleTimeout) {
    const timeoutMs = parseInt(envIdleTimeout) * 60 * 1000;
    for (const model of config.models) {
      model.idleTimeout = timeoutMs;
    }
  }

  return config;
}

export function loadConfig(): ManagerConfig {
  let config = { ...defaultConfig };
  const configPath = getConfigPath();
  if (configPath) {
    try {
      config = readConfigFile(configPath);
    } catch (err) {
      console.warn(`Failed to load config from ${configPath}, using defaults`, err);
    }
  }

  return applyDefaults(config);
}

/**
 * Re-read config from disk, bypassing require cache.
 * Returns fresh ManagerConfig with all defaults applied.
 */
export function reloadConfig(): ManagerConfig {
  let config = { ...defaultConfig };
  const configPath = getConfigPath();
  if (configPath) {
    try {
      config = readConfigFile(configPath);
      console.log(`[config] Reloaded config from ${configPath}`);
    } catch (err) {
      console.warn(`[config] Failed to reload from ${configPath}:`, err);
    }
  }

  return applyDefaults(config);
}

/**
 * Re-read config and return the ModelConfig for a specific model name.
 * Returns null if the model is not found in the reloaded config.
 */
export function reloadModelConfig(modelName: string): ModelConfig | null {
  const config = reloadConfig();
  return config.models.find((m) => m.name === modelName) || null;
}

/** Build a lookup map: model name/alias -> ModelConfig */
export function buildModelLookup(models: ModelConfig[]): Map<string, ModelConfig> {
  const map = new Map<string, ModelConfig>();
  for (const model of models) {
    map.set(model.name, model);
    if (model.aliases) {
      for (const alias of model.aliases) {
        map.set(alias, model);
      }
    }
  }
  return map;
}
