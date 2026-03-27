/**
 * Process Manager - handles starting/stopping vLLM, llama-server, and sglang processes.
 *
 * Each backend implements the same interface so the proxy can treat them uniformly.
 */

import { spawn, type ChildProcess } from "child_process";
import type { ModelConfig } from "./config";

type StartupLogMatcher = (text: string) => boolean;

async function waitForProcessReady(
  config: ModelConfig,
  internalPort: number,
  state: ProcessState,
  spawnProcess: () => ChildProcess,
  readyLogMatcher: StartupLogMatcher,
  startupLabel: string
): Promise<void> {
  if (state.status === "running") return;
  if (state.status === "starting") {
    await waitForStateTransition(state, config.startupTimeout || 600000);
    return;
  }

  state.status = "starting";
  state.error = null;

  await new Promise<void>((resolve, reject) => {
    const proc = spawnProcess();
    state.process = proc;

    let settled = false;
    let probeTimer: Timer | null = null;
    let deadlineTimer: Timer | null = null;

    const cleanup = () => {
      if (probeTimer) clearInterval(probeTimer);
      if (deadlineTimer) clearTimeout(deadlineTimer);
      proc.stdout?.off("data", onData);
      proc.stderr?.off("data", onData);
      proc.off("error", onError);
      proc.off("exit", onExit);
    };

    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };

    const succeed = () => {
      if (settled) return;
      settled = true;
      state.status = "running";
      state.startedAt = Date.now();
      state.lastActivity = Date.now();
      cleanup();
      resolve();
    };

    const onData = (data: Buffer) => {
      const text = data.toString();
      process.stdout.write(`[${config.name}] ${text}`);
      if (readyLogMatcher(text)) {
        void probeHealth().then((ready) => {
          if (ready) {
            console.log(`[${config.name}] ${startupLabel} ready on port ${internalPort}`);
            succeed();
          }
        }).catch(() => {});
      }
    };

    const onError = (err: Error) => {
      state.status = "stopped";
      state.error = err.message;
      state.process = null;
      fail(err);
    };

    const onExit = (code: number | null) => {
      const wasStarting = state.status === "starting";
      state.status = "stopped";
      state.process = null;
      if (wasStarting) {
        fail(new Error(`Process exited with code ${code}`));
      }
    };

    const probeHealth = async (): Promise<boolean> => {
      if (state.process !== proc || state.status === "stopped") return false;
      try {
        const response = await fetch(`http://127.0.0.1:${internalPort}/health`, {
          signal: AbortSignal.timeout(3000),
        });
        return response.ok;
      } catch {
        return false;
      }
    };

    proc.stdout?.on("data", onData);
    proc.stderr?.on("data", onData);
    proc.on("error", onError);
    proc.on("exit", onExit);

    probeTimer = setInterval(() => {
      void probeHealth().then((ready) => {
        if (ready) {
          console.log(`[${config.name}] ${startupLabel} ready on port ${internalPort}`);
          succeed();
        }
      }).catch(() => {});
    }, 1000);

    deadlineTimer = setTimeout(() => {
      if (settled || state.status !== "starting") return;
      state.error = "Startup timeout";
      proc.kill("SIGTERM");
      fail(new Error(`${startupLabel} startup timeout`));
    }, config.startupTimeout || 600000);
  });
}

async function waitForStateTransition(state: ProcessState, timeoutMs: number): Promise<void> {
  const started = Date.now();
  while (state.status === "starting") {
    if (Date.now() - started > timeoutMs) {
      throw new Error(state.error || "Startup timeout");
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }

  if (state.status !== "running") {
    throw new Error(state.error || "Startup failed");
  }
}

export interface ProcessState {
  process: ChildProcess | null;
  status: "stopped" | "starting" | "running" | "stopping" | "sleeping";
  lastActivity: number;
  startedAt: number | null;
  error: string | null;
}

/** Common interface for all backend processes */
export interface BackendProcess {
  readonly status: string;
  readonly lastActivity: number;
  touchActivity(): void;
  start(): Promise<void>;
  stop(): Promise<void>;
  healthCheck(): Promise<boolean>;
  getInternalUrl(): string;
  getState(): ProcessState & { name: string };
  /** Put the model to sleep (offload weights to CPU). Only supported by vLLM. */
  sleep(): Promise<boolean>;
  /** Wake a sleeping model. Only supported by vLLM. */
  wake(): Promise<boolean>;
  /** Whether this backend supports sleep/wake */
  readonly supportsSleep: boolean;
  /** Swap in a new ModelConfig. Takes effect on next start(). Process must be stopped first. */
  updateConfig(config: ModelConfig): void;
  /** Get the current config */
  getConfig(): ModelConfig;
}

/** Factory: create the right process class based on config.backend */
export function createProcess(config: ModelConfig, internalPort: number): BackendProcess {
  switch (config.backend) {
    case "llama-server":
      return new LlamaServerProcess(config, internalPort);
    case "sglang":
      return new SGLangProcess(config, internalPort);
    default:
      return new VLLMProcess(config, internalPort);
  }
}

// ─── Base helpers ───────────────────────────────────────────────

function cudaDevices(config: ModelConfig): string {
  return config.gpus ? config.gpus.join(",") : config.gpu.toString();
}

function isMultiGpu(config: ModelConfig): boolean {
  return (config.gpus?.length ?? 1) > 1;
}

function gpuLabel(config: ModelConfig): string {
  return config.gpus ? `GPUs ${cudaDevices(config)}` : `GPU ${config.gpu}`;
}

function ncclEnv(config: ModelConfig): Record<string, string> {
  if (!isMultiGpu(config)) return {};
  return {
    NCCL_P2P_DISABLE: "1",
    NCCL_IB_DISABLE: "1",
    NCCL_SHM_DISABLE: "1",
    NCCL_DEBUG: "WARN",
  };
}

async function waitForGpuReclaim(config: ModelConfig): Promise<void> {
  console.log(`[${config.name}] Waiting for GPU memory reclaim...`);
  await new Promise((r) => setTimeout(r, 5000));

  const cleanupDevices = cudaDevices(config);
  const cleanup = Bun.spawnSync({
    cmd: ["python3", "-c", "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"],
    env: { ...process.env, CUDA_VISIBLE_DEVICES: cleanupDevices },
    stderr: "pipe",
  });
  if (cleanup.exitCode === 0) {
    console.log(`[${config.name}] GPU memory reclaimed`);
  }
}

// ─── vLLM Process ───────────────────────────────────────────────

export class VLLMProcess implements BackendProcess {
  private config: ModelConfig;
  private internalPort: number;
  private state: ProcessState;

  constructor(config: ModelConfig, internalPort: number) {
    this.config = config;
    this.internalPort = internalPort;
    this.state = {
      process: null,
      status: "stopped",
      lastActivity: Date.now(),
      startedAt: null,
      error: null,
    };
  }

  get status() { return this.state.status; }
  get lastActivity() { return this.state.lastActivity; }
  get supportsSleep() { return true; }
  touchActivity() { this.state.lastActivity = Date.now(); }
  getConfig(): ModelConfig { return this.config; }
  updateConfig(config: ModelConfig): void {
    if (this.state.status !== "stopped") {
      throw new Error(`Cannot update config while process is ${this.state.status}`);
    }
    this.config = config;
  }

  async start(): Promise<void> {
    if (this.state.status === "sleeping") {
      await this.wake();
      return;
    }

    const args = [
      "serve", this.config.modelPath,
      "--port", this.internalPort.toString(),
      "--host", "127.0.0.1",
      "--dtype", this.config.dtype || "bfloat16",
      "--served-model-name", this.config.name,
      "--enable-sleep-mode",
      ...(this.config.extraArgs || []),
    ];

    const env = {
      ...process.env,
      CUDA_VISIBLE_DEVICES: cudaDevices(this.config),
      ...ncclEnv(this.config),
      VLLM_SERVER_DEV_MODE: "1",
      FLASH_ATTN: "1",
      OMP_NUM_THREADS: "6",
      VLLM_CPU_KVCACHE_SPACE: "8",
      VLLM_FLASHINFER_MOE_BACKEND: "throughput",
      VLLM_MARLIN_USE_ATOMIC_ADD: "1",
      VLLM_TARGET_DEVICE: "cuda",
      CUDA_DEVICE_ORDER: "FASTEST_FIRST",
    };

    console.log(`[${this.config.name}] Starting vLLM on ${gpuLabel(this.config)}: vllm ${args.join(" ")}`);
    console.log(`[${this.config.name}] CUDA_VISIBLE_DEVICES=${env.CUDA_VISIBLE_DEVICES}`);

    await waitForProcessReady(
      this.config,
      this.internalPort,
      this.state,
      () => spawn("vllm", args, { env, stdio: ["pipe", "pipe", "pipe"] }),
      (text) => text.includes("Uvicorn running") || text.includes("Application startup complete"),
      "vLLM"
    );
  }

  async sleep(): Promise<boolean> {
    if (this.state.status !== "running") return false;
    try {
      const r = await fetch(`http://127.0.0.1:${this.internalPort}/sleep?level=1`, {
        method: "POST",
        signal: AbortSignal.timeout(30000),
      });
      if (r.ok) {
        this.state.status = "sleeping";
        console.log(`[${this.config.name}] Entered sleep mode (weights offloaded to CPU)`);
        return true;
      }
      console.warn(`[${this.config.name}] Sleep request failed: ${r.status}`);
      return false;
    } catch (err) {
      console.warn(`[${this.config.name}] Sleep request error:`, err);
      return false;
    }
  }

  async wake(): Promise<boolean> {
    if (this.state.status !== "sleeping") return false;
    try {
      const r = await fetch(`http://127.0.0.1:${this.internalPort}/wake_up`, {
        method: "POST",
        signal: AbortSignal.timeout(60000),
      });
      if (r.ok) {
        this.state.status = "running";
        this.state.lastActivity = Date.now();
        console.log(`[${this.config.name}] Woke up from sleep mode`);
        return true;
      }
      console.warn(`[${this.config.name}] Wake request failed: ${r.status}`);
      return false;
    } catch (err) {
      console.warn(`[${this.config.name}] Wake request error:`, err);
      return false;
    }
  }

  async stop(): Promise<void> {
    if (!this.state.process || this.state.status === "stopped") return;
    this.state.status = "stopping";
    await new Promise<void>((resolve) => {
      const proc = this.state.process!;
      const forceKill = setTimeout(() => proc.kill("SIGKILL"), 10000);
      proc.on("exit", () => { clearTimeout(forceKill); this.state.process = null; resolve(); });
      proc.kill("SIGTERM");
    });
    await waitForGpuReclaim(this.config);
    this.state.status = "stopped";
  }

  async healthCheck(): Promise<boolean> {
    if (this.state.status !== "running") return false;
    try {
      const r = await fetch(`http://127.0.0.1:${this.internalPort}/health`, { signal: AbortSignal.timeout(5000) });
      return r.ok;
    } catch { return false; }
  }

  getInternalUrl(): string { return `http://127.0.0.1:${this.internalPort}`; }

  getState(): ProcessState & { name: string } {
    return { ...this.state, name: this.config.name };
  }
}

// ─── llama-server Process ───────────────────────────────────────

export class LlamaServerProcess implements BackendProcess {
  private config: ModelConfig;
  private internalPort: number;
  private state: ProcessState;

  constructor(config: ModelConfig, internalPort: number) {
    this.config = config;
    this.internalPort = internalPort;
    this.state = {
      process: null,
      status: "stopped",
      lastActivity: Date.now(),
      startedAt: null,
      error: null,
    };
  }

  get status() { return this.state.status; }
  get lastActivity() { return this.state.lastActivity; }
  get supportsSleep() { return false; }
  touchActivity() { this.state.lastActivity = Date.now(); }
  getConfig(): ModelConfig { return this.config; }
  updateConfig(config: ModelConfig): void {
    if (this.state.status !== "stopped") {
      throw new Error(`Cannot update config while process is ${this.state.status}`);
    }
    this.config = config;
  }
  async sleep(): Promise<boolean> { return false; }
  async wake(): Promise<boolean> { return false; }

  async start(): Promise<void> {
    const args = [
      "-m", this.config.modelPath,
      "--port", this.internalPort.toString(),
      "--host", "127.0.0.1",
      "-a", this.config.name,
      ...(this.config.extraArgs || []),
    ];

    const env = {
      ...process.env,
      CUDA_VISIBLE_DEVICES: cudaDevices(this.config),
      GGML_CUDA_GRAPH_OPT: "1",
    };

    console.log(`[${this.config.name}] Starting llama-server on ${gpuLabel(this.config)}: llama-server ${args.join(" ")}`);

    await waitForProcessReady(
      this.config,
      this.internalPort,
      this.state,
      () => spawn("llama-server", args, { env, stdio: ["pipe", "pipe", "pipe"] }),
      (text) =>
        text.includes("server is listening") ||
        text.includes("HTTP server listening") ||
        text.includes("server listening") ||
        text.includes("llama server listening"),
      "llama-server"
    );
  }

  async stop(): Promise<void> {
    if (!this.state.process || this.state.status === "stopped") return;
    this.state.status = "stopping";
    await new Promise<void>((resolve) => {
      const proc = this.state.process!;
      const forceKill = setTimeout(() => proc.kill("SIGKILL"), 10000);
      proc.on("exit", () => { clearTimeout(forceKill); this.state.process = null; resolve(); });
      proc.kill("SIGTERM");
    });
    await new Promise((r) => setTimeout(r, 5000));
    this.state.status = "stopped";
  }

  async healthCheck(): Promise<boolean> {
    if (this.state.status !== "running") return false;
    try {
      const r = await fetch(`http://127.0.0.1:${this.internalPort}/health`, { signal: AbortSignal.timeout(5000) });
      return r.ok;
    } catch { return false; }
  }

  getInternalUrl(): string { return `http://127.0.0.1:${this.internalPort}`; }

  getState(): ProcessState & { name: string } {
    return { ...this.state, name: this.config.name };
  }
}

// ─── SGLang Process ─────────────────────────────────────────────

export class SGLangProcess implements BackendProcess {
  private config: ModelConfig;
  private internalPort: number;
  private state: ProcessState;

  constructor(config: ModelConfig, internalPort: number) {
    this.config = config;
    this.internalPort = internalPort;
    this.state = {
      process: null,
      status: "stopped",
      lastActivity: Date.now(),
      startedAt: null,
      error: null,
    };
  }

  get status() { return this.state.status; }
  get lastActivity() { return this.state.lastActivity; }
  get supportsSleep() { return false; }
  touchActivity() { this.state.lastActivity = Date.now(); }
  getConfig(): ModelConfig { return this.config; }
  updateConfig(config: ModelConfig): void {
    if (this.state.status !== "stopped") {
      throw new Error(`Cannot update config while process is ${this.state.status}`);
    }
    this.config = config;
  }
  async sleep(): Promise<boolean> { return false; }
  async wake(): Promise<boolean> { return false; }

  async start(): Promise<void> {
    const args = [
      "-m", "sglang.launch_server",
      "--model", this.config.modelPath,
      "--port", this.internalPort.toString(),
      "--host", "127.0.0.1",
      "--served-model-name", this.config.name,
      ...(this.config.extraArgs || []),
    ];

    const env = {
      ...process.env,
      CUDA_VISIBLE_DEVICES: cudaDevices(this.config),
      ...ncclEnv(this.config),
    };

    console.log(`[${this.config.name}] Starting SGLang on ${gpuLabel(this.config)}: python3 ${args.join(" ")}`);

    await waitForProcessReady(
      this.config,
      this.internalPort,
      this.state,
      () => spawn("python3", args, { env, stdio: ["pipe", "pipe", "pipe"] }),
      (text) =>
        text.includes("The server is fired up") ||
        text.includes("Uvicorn running") ||
        text.includes("Application startup complete"),
      "SGLang"
    );
  }

  async stop(): Promise<void> {
    if (!this.state.process || this.state.status === "stopped") return;
    this.state.status = "stopping";
    await new Promise<void>((resolve) => {
      const proc = this.state.process!;
      const forceKill = setTimeout(() => proc.kill("SIGKILL"), 10000);
      proc.on("exit", () => { clearTimeout(forceKill); this.state.process = null; resolve(); });
      proc.kill("SIGTERM");
    });
    await waitForGpuReclaim(this.config);
    this.state.status = "stopped";
  }

  async healthCheck(): Promise<boolean> {
    if (this.state.status !== "running") return false;
    try {
      const r = await fetch(`http://127.0.0.1:${this.internalPort}/health`, { signal: AbortSignal.timeout(5000) });
      return r.ok;
    } catch { return false; }
  }

  getInternalUrl(): string { return `http://127.0.0.1:${this.internalPort}`; }

  getState(): ProcessState & { name: string } {
    return { ...this.state, name: this.config.name };
  }
}
