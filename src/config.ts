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
  /**
   * Specific file to download from the HF repo (llama-server only).
   * Use for GGUF repos that ship many quantizations — downloads just this
   * file instead of the whole repo, e.g. "Qwythos-27B-Q4_K_M.gguf".
   */
  modelFile?: string;
  /** Backend: "vllm" (default), "llama-server", or "sglang" */
  backend?: "vllm" | "llama-server" | "sglang";
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

const DEFAULT_IDLE_TIMEOUT_MS = 300000;

function readConfigFile(configPath: string): ManagerConfig {
  const raw = readFileSync(configPath, "utf-8");
  const loaded = JSON.parse(raw) as Partial<ManagerConfig>;
  return { ...defaultConfig, ...loaded };
}

/**
 * Reject a config we can't serve, loudly. Half-loading it means the router
 * comes up healthy while quietly serving fewer models than the user wrote.
 */
function validateConfig(config: ManagerConfig): void {
  const errors: string[] = [];

  if (!Array.isArray(config.models)) {
    throw new Error("config.models must be an array");
  }

  const names = new Set<string>();
  for (const [i, model] of config.models.entries()) {
    const label = model?.name ? `models[${i}] (${model.name})` : `models[${i}]`;

    if (!model || typeof model !== "object") {
      errors.push(`${label}: must be an object`);
      continue;
    }
    if (!model.name || typeof model.name !== "string") {
      errors.push(`${label}: "name" is required`);
    } else if (names.has(model.name)) {
      errors.push(`${label}: duplicate model name "${model.name}"`);
    } else {
      names.add(model.name);
    }

    if (!model.modelPath || typeof model.modelPath !== "string") {
      errors.push(`${label}: "modelPath" is required`);
    }
    if (typeof model.gpu !== "number" || !Number.isInteger(model.gpu) || model.gpu < 0) {
      errors.push(`${label}: "gpu" must be a non-negative integer`);
    }
    if (model.gpus && model.gpus.some((g) => typeof g !== "number")) {
      errors.push(`${label}: "gpus" must be a list of numbers`);
    }
    if (model.idleTimeout === undefined) {
      model.idleTimeout = DEFAULT_IDLE_TIMEOUT_MS;
    } else if (typeof model.idleTimeout !== "number" || !(model.idleTimeout > 0)) {
      errors.push(`${label}: "idleTimeout" must be a positive number of milliseconds`);
    }
  }

  // Aliases are looked up in the same namespace as names, so a collision
  // silently steals another model's traffic.
  for (const model of config.models) {
    for (const alias of model?.aliases ?? []) {
      if (names.has(alias) && alias !== model.name) {
        errors.push(`models (${model.name}): alias "${alias}" collides with another model's name`);
      }
    }
  }

  if (errors.length > 0) {
    throw new Error(`Invalid configuration:\n  - ${errors.join("\n  - ")}`);
  }
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
  const configPath = getConfigPath();
  if (!configPath) return applyDefaults({ ...defaultConfig });

  let config: ManagerConfig;
  try {
    config = readConfigFile(configPath);
    validateConfig(config);
  } catch (err) {
    console.error(`✗ Cannot load config from ${configPath}:`);
    console.error(`  ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }

  if (config.models.length === 0) {
    console.warn(`⚠ ${configPath} declares no models — the router will serve nothing.`);
  }

  return applyDefaults(config);
}

/**
 * Re-read config from disk, bypassing require cache.
 * Returns fresh ManagerConfig with all defaults applied. Throws on a bad file:
 * unlike boot, a bad reload must not take the running router down.
 */
export function reloadConfig(): ManagerConfig {
  const configPath = getConfigPath();
  if (!configPath) return applyDefaults({ ...defaultConfig });

  const config = readConfigFile(configPath);
  validateConfig(config);
  console.log(`[config] Reloaded config from ${configPath}`);
  return applyDefaults(config);
}

/**
 * Re-read config and return the ModelConfig for a specific model name.
 * Returns null if the model is not found or the file is unusable.
 */
export function reloadModelConfig(modelName: string): ModelConfig | null {
  try {
    const config = reloadConfig();
    return config.models.find((m) => m.name === modelName) || null;
  } catch (err) {
    console.warn(`[config] Failed to reload config:`, err instanceof Error ? err.message : err);
    return null;
  }
}
