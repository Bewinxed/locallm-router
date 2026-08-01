/**
 * Process Manager — starts, supervises and stops backend inference servers.
 *
 * One class drives every backend; what differs between vLLM, llama-server and
 * SGLang (binary, args, env, ready log, health path) lives in BACKENDS.
 *
 * Two invariants the rest of the router depends on:
 *   - stdout/stderr are drained for the whole life of the child. A pipe nobody
 *     reads fills at ~64KB and the backend blocks forever on its next log line.
 *   - every exit, expected or not, lands in handleExit(): status goes back to
 *     "stopped", the handle is dropped and the owner is notified so the GPU
 *     reservation is released.
 */

import type { Subprocess } from "bun";
import { readdirSync, readFileSync } from "fs";
import type { ModelConfig } from "./config";
import { resolveModelPath } from "./download";
import { argValue, getGpuInfo } from "./model-size";

export type ProcessStatus = "stopped" | "starting" | "running" | "stopping" | "sleeping";

/** Snapshot of a process for the management API (no live handles). */
export interface ProcessSnapshot {
  name: string;
  status: ProcessStatus;
  lastActivity: number;
  startedAt: number | null;
  error: string | null;
  pid: number | null;
}

export interface ExitInfo {
  exitCode: number | null;
  signalCode: number | null;
  /** True when the exit came from stop() rather than a crash. */
  expected: boolean;
}

const STOP_GRACE_MS = 15_000;
const SIGKILL_GRACE_MS = 2_000;
const HEALTH_PROBE_INTERVAL_MS = 1_000;
const HEALTH_PROBE_TIMEOUT_MS = 3_000;
const GPU_RECLAIM_TIMEOUT_MS = 10_000;
const GPU_RECLAIM_THRESHOLD_MB = 512;
const GROUP_REAP_ATTEMPTS = 5;
const GROUP_REAP_INTERVAL_MS = 1_000;

/**
 * Spawning through setsid puts the backend in its own process group, so
 * SIGTERM to -pid also reaches vLLM's forked workers. Without it we may only
 * signal the direct child (never our own group — see signal()).
 */
const SETSID_BIN = Bun.which("setsid");

/**
 * Environment handed to backends. An explicit allowlist rather than the
 * router's whole environment: the child gets what it needs to find its
 * libraries, cache and GPUs, and nothing else.
 */
const ENV_ALLOWLIST = ["PATH", "HOME", "USER", "HF_TOKEN", "HF_HOME", "LD_LIBRARY_PATH", "DISPLAY"];
const ENV_ALLOWED_PREFIXES = ["CUDA_", "GGML_", "VK_", "XDG_"];

function allowedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
    if (ENV_ALLOWLIST.includes(key) || ENV_ALLOWED_PREFIXES.some((p) => key.startsWith(p))) {
      env[key] = value;
    }
  }
  return env;
}

// ─── Backend descriptors ────────────────────────────────────────

interface BackendDescriptor {
  label: string;
  binary: string;
  healthPath: string;
  supportsSleep: boolean;
  buildArgs(config: ModelConfig, port: number): string[];
  buildEnv(config: ModelConfig): Record<string, string>;
  isReadyLog(text: string): boolean;
  /** Feature detection this backend needs before its args can be built. */
  prepare?(): Promise<void>;
}

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
    // Tensor parallelism here is single-node, so the bootstrap never has to
    // leave the box. Left to itself NCCL picks the first routable interface —
    // on this machine the WiFi adapter — and a collective that can't connect
    // there wedges the whole start for minutes before it gives up.
    NCCL_SOCKET_IFNAME: "lo",
    NCCL_DEBUG: "WARN",
  };
}

/**
 * Whether the installed llama-server understands --fit-ctx. Probed once, on
 * the first launch: an older build errors out on an unknown flag, and a
 * working setup must not break because we added a guard to it.
 */
let fitCtxSupported: boolean | null = null;

async function probeFitCtxSupport(): Promise<void> {
  if (fitCtxSupported !== null) return;
  try {
    const proc = Bun.spawn({
      cmd: [BACKENDS["llama-server"].binary, "--help"],
      stdout: "pipe",
      stderr: "ignore",
    });
    const [stdout] = await Promise.all([new Response(proc.stdout).text(), proc.exited]);
    fitCtxSupported = stdout.includes("--fit-ctx");
  } catch {
    fitCtxSupported = false;
  }
  console.log(
    `[process] llama-server ${fitCtxSupported ? "supports" : "does not support"} --fit-ctx`
  );
}

/**
 * llama.cpp's --fit is on by default in recent builds: rather than failing to
 * fit, it quietly shrinks the context and serves less than was asked for.
 * Pinning the floor to the requested context turns that back into a startup
 * error we can see.
 */
function fitCtxArgs(extraArgs: string[]): string[] {
  if (!fitCtxSupported) return [];
  if (extraArgs.some((arg) => ["--fit", "-fit", "--fit-ctx", "-fitc"].includes(arg))) return [];
  const ctx = argValue(extraArgs, "-c", "--ctx-size");
  return ctx ? ["--fit-ctx", ctx] : [];
}

const BACKENDS: Record<string, BackendDescriptor> = {
  vllm: {
    label: "vLLM",
    binary: "vllm",
    healthPath: "/health",
    supportsSleep: true,
    buildArgs: (config, port) => [
      "serve", config.modelPath,
      "--port", port.toString(),
      "--host", "127.0.0.1",
      "--dtype", config.dtype || "bfloat16",
      "--served-model-name", config.name,
      "--enable-sleep-mode",
      ...(config.extraArgs || []),
    ],
    buildEnv: (config) => ({
      CUDA_VISIBLE_DEVICES: cudaDevices(config),
      ...ncclEnv(config),
      VLLM_SERVER_DEV_MODE: "1",
      FLASH_ATTN: "1",
      OMP_NUM_THREADS: "6",
      VLLM_CPU_KVCACHE_SPACE: "8",
      VLLM_FLASHINFER_MOE_BACKEND: "throughput",
      VLLM_MARLIN_USE_ATOMIC_ADD: "1",
      VLLM_TARGET_DEVICE: "cuda",
      CUDA_DEVICE_ORDER: "FASTEST_FIRST",
    }),
    isReadyLog: (text) =>
      text.includes("Uvicorn running") || text.includes("Application startup complete"),
  },

  "llama-server": {
    label: "llama-server",
    binary: "llama-server",
    healthPath: "/health",
    supportsSleep: false,
    buildArgs: (config, port) => [
      "-m", resolveModelPath(config),
      "--port", port.toString(),
      "--host", "127.0.0.1",
      "-a", config.name,
      ...(config.extraArgs || []),
      ...fitCtxArgs(config.extraArgs || []),
    ],
    prepare: probeFitCtxSupport,
    buildEnv: (config) => ({
      CUDA_VISIBLE_DEVICES: cudaDevices(config),
      // Vulkan builds of llama.cpp ignore CUDA_VISIBLE_DEVICES; this is the
      // equivalent device filter for them. Harmless on CUDA builds.
      GGML_VK_VISIBLE_DEVICES: cudaDevices(config),
      GGML_CUDA_GRAPH_OPT: "1",
    }),
    isReadyLog: (text) =>
      text.includes("server is listening") ||
      text.includes("HTTP server listening") ||
      text.includes("server listening") ||
      text.includes("llama server listening"),
  },

  sglang: {
    label: "SGLang",
    binary: "python3",
    healthPath: "/health",
    supportsSleep: false,
    buildArgs: (config, port) => [
      "-m", "sglang.launch_server",
      "--model", config.modelPath,
      "--port", port.toString(),
      "--host", "127.0.0.1",
      "--served-model-name", config.name,
      ...(config.extraArgs || []),
    ],
    buildEnv: (config) => ({
      CUDA_VISIBLE_DEVICES: cudaDevices(config),
      ...ncclEnv(config),
    }),
    isReadyLog: (text) =>
      text.includes("The server is fired up") ||
      text.includes("Uvicorn running") ||
      text.includes("Application startup complete"),
  },
};

/** Create a backend process. Throws on an unknown backend rather than guessing. */
export function createProcess(
  config: ModelConfig,
  internalPort: number,
  onExit?: (info: ExitInfo) => void
): BackendProcess {
  const name = config.backend || "vllm";
  const descriptor = BACKENDS[name];
  if (!descriptor) {
    throw new Error(
      `Model ${config.name}: unknown backend "${name}" (expected ${Object.keys(BACKENDS).join(", ")})`
    );
  }
  return new BackendProcess(config, internalPort, descriptor, onExit);
}

// ─── Backend process ────────────────────────────────────────────

export class BackendProcess {
  private config: ModelConfig;
  private readonly internalPort: number;
  private readonly backend: BackendDescriptor;
  private readonly onExit?: (info: ExitInfo) => void;

  private proc: Subprocess | null = null;
  private state: ProcessStatus = "stopped";
  private startedAt: number | null = null;
  private lastActivityAt = Date.now();
  private lastError: string | null = null;

  private startPromise: Promise<void> | null = null;
  private stopPromise: Promise<void> | null = null;
  /**
   * Set the moment a stop is asked for. A model evicted between start() and
   * the spawn would otherwise finish launching into VRAM that has already been
   * promised to someone else.
   */
  private stopRequested = false;
  /** Set while a startup is in flight: fed by the log drain, settled by exits. */
  private onLogLine: ((line: string) => void) | null = null;
  private failStartup: ((err: Error) => void) | null = null;

  constructor(
    config: ModelConfig,
    internalPort: number,
    backend: BackendDescriptor,
    onExit?: (info: ExitInfo) => void
  ) {
    this.config = config;
    this.internalPort = internalPort;
    this.backend = backend;
    this.onExit = onExit;
  }

  get status(): ProcessStatus { return this.state; }
  get lastActivity(): number { return this.lastActivityAt; }
  get supportsSleep(): boolean { return this.backend.supportsSleep; }

  touchActivity(): void { this.lastActivityAt = Date.now(); }
  getConfig(): ModelConfig { return this.config; }
  getInternalUrl(): string { return `http://127.0.0.1:${this.internalPort}`; }

  updateConfig(config: ModelConfig): void {
    if (this.state !== "stopped") {
      throw new Error(`Cannot update config while process is ${this.state}`);
    }
    this.config = config;
  }

  getState(): ProcessSnapshot {
    return {
      name: this.config.name,
      status: this.state,
      lastActivity: this.lastActivityAt,
      startedAt: this.startedAt,
      error: this.lastError,
      pid: this.proc?.pid ?? null,
    };
  }

  async start(): Promise<void> {
    // A stop already in flight — an eviction, say — owns the port until it
    // finishes; spawning into it would leave two backends fighting over it.
    if (this.stopPromise) await this.stopPromise.catch(() => {});
    this.stopRequested = false;
    if (this.state === "sleeping") {
      if (await this.wake()) return;
      await this.stop();
    }
    if (this.state === "running") return;
    if (!this.startPromise) {
      this.startPromise = this.launch().finally(() => {
        this.startPromise = null;
      });
    }
    return this.startPromise;
  }

  async stop(): Promise<void> {
    this.stopRequested = true;
    if (!this.stopPromise) {
      this.stopPromise = this.terminate().finally(() => {
        this.stopPromise = null;
      });
    }
    return this.stopPromise;
  }

  async healthCheck(): Promise<boolean> {
    if (this.state !== "running") return false;
    return this.probeHealth();
  }

  /** Synchronous best-effort kill, for the router's own exit handler. */
  killNow(): void {
    if (this.proc) this.signal(this.proc, "SIGKILL");
  }

  /**
   * Every pid holding this backend's GPU memory: the child we spawned plus the
   * workers it forked, which are its process group (see SETSID_BIN).
   */
  groupPids(): number[] {
    const pid = this.proc?.pid;
    if (pid === undefined) return [];
    return SETSID_BIN ? processGroupMembers(pid) : [pid];
  }

  /** Abort a launch that a stop overtook, rather than starting it anyway. */
  private assertNotStopping(): void {
    if (this.stopRequested) {
      throw new Error(`${this.config.name} was stopped during startup`);
    }
  }

  /** Put the model to sleep (offload weights to CPU). vLLM only. */
  async sleep(): Promise<boolean> {
    if (!this.backend.supportsSleep || this.state !== "running") return false;
    try {
      const r = await fetch(`${this.getInternalUrl()}/sleep`, {
        method: "POST",
        signal: AbortSignal.timeout(120000),
      });
      if (r.ok) {
        this.state = "sleeping";
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

  /** Wake a sleeping model. vLLM only. */
  async wake(): Promise<boolean> {
    if (this.state !== "sleeping") return false;
    try {
      const r = await fetch(`${this.getInternalUrl()}/wake_up`, {
        method: "POST",
        signal: AbortSignal.timeout(60000),
      });
      if (r.ok) {
        this.state = "running";
        this.lastActivityAt = Date.now();
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

  // ─── Startup ──────────────────────────────────────────────

  private async launch(): Promise<void> {
    await this.claimPort();
    await this.backend.prepare?.();

    const args = this.backend.buildArgs(this.config, this.internalPort);
    const env = { ...allowedEnv(), ...this.backend.buildEnv(this.config) };

    // Last check before the spawn: everything above here can await, and a stop
    // that lands in one of those gaps has nothing to kill yet.
    this.assertNotStopping();

    this.state = "starting";
    this.lastError = null;

    console.log(
      `[${this.config.name}] Starting ${this.backend.label} on ${gpuLabel(this.config)}: ` +
        `${this.backend.binary} ${args.join(" ")}`
    );

    const cmd = SETSID_BIN
      ? [SETSID_BIN, this.backend.binary, ...args]
      : [this.backend.binary, ...args];

    let proc: Subprocess;
    try {
      proc = Bun.spawn({
        cmd,
        env,
        stdin: "ignore",
        stdout: "pipe",
        stderr: "pipe",
        onExit: (child, exitCode, signalCode) => this.handleExit(child, exitCode, signalCode),
      });
    } catch (err) {
      this.state = "stopped";
      this.lastError = String(err);
      throw new Error(`Failed to spawn ${this.backend.binary}: ${err}`);
    }

    this.proc = proc;
    this.drain(proc.stdout);
    this.drain(proc.stderr);

    try {
      await this.waitForReady(proc);
    } catch (err) {
      this.lastError = err instanceof Error ? err.message : String(err);
      throw err;
    }
  }

  /**
   * A backend must own its port outright. Binding blind is how a restart that
   * raced a lazy start ends up proxying to an orphan of the previous run: the
   * spawn fails on the taken port while requests keep reaching the stranger.
   */
  private async claimPort(): Promise<void> {
    let holder = await portListener(this.internalPort);
    if (holder === null) return;

    const tracked = this.proc?.pid ?? null;
    if (tracked === null || !isOurs(holder, tracked)) {
      throw new Error(
        `Port ${this.internalPort} is held by pid ${holder}, which is not ${this.config.name}'s ` +
          `backend (tracked pid ${tracked ?? "none"}) — refusing to start`
      );
    }

    console.warn(
      `[${this.config.name}] Port ${this.internalPort} still held by our pid ${holder}, stopping it first`
    );
    await this.stop();

    holder = await portListener(this.internalPort);
    if (holder !== null) {
      throw new Error(
        `Port ${this.internalPort} is still held by pid ${holder} after stopping ${this.config.name}`
      );
    }
  }

  private waitForReady(proc: Subprocess): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      let probeTimer: Timer | null = null;
      let deadlineTimer: Timer | null = null;

      const settle = (err?: Error) => {
        if (settled) return;
        settled = true;
        if (probeTimer) clearInterval(probeTimer);
        if (deadlineTimer) clearTimeout(deadlineTimer);
        this.onLogLine = null;
        this.failStartup = null;
        if (err) reject(err);
        else resolve();
      };

      const probe = () => {
        if (this.stopRequested) {
          settle(new Error(`${this.backend.label} was stopped during startup`));
          return;
        }
        void this.probeHealth(proc).then((ready) => {
          if (!ready || settled) return;
          this.state = "running";
          this.startedAt = Date.now();
          this.lastActivityAt = Date.now();
          console.log(
            `[${this.config.name}] ${this.backend.label} ready on port ${this.internalPort}`
          );
          settle();
        });
      };

      // Any exit before we are ready fails the startup, whatever the status
      // says — an eviction flips it to "stopping" before the process is gone.
      this.failStartup = (err) => settle(err);
      this.onLogLine = (line) => {
        if (this.backend.isReadyLog(line)) probe();
      };

      probeTimer = setInterval(probe, HEALTH_PROBE_INTERVAL_MS);

      const timeoutMs = this.config.startupTimeout || 600000;
      deadlineTimer = setTimeout(() => {
        this.lastError = "Startup timeout";
        void this.stop();
        settle(new Error(`${this.backend.label} startup timeout after ${timeoutMs / 1000}s`));
      }, timeoutMs);
    });
  }

  private async probeHealth(proc?: Subprocess): Promise<boolean> {
    if (proc && this.proc !== proc) return false;
    try {
      const r = await fetch(`${this.getInternalUrl()}${this.backend.healthPath}`, {
        signal: AbortSignal.timeout(HEALTH_PROBE_TIMEOUT_MS),
      });
      return r.ok;
    } catch {
      return false;
    }
  }

  // ─── Log drain ────────────────────────────────────────────

  private drain(stream: Subprocess["stdout"]): void {
    if (!(stream instanceof ReadableStream)) return;
    void this.pump(stream);
  }

  private async pump(stream: ReadableStream<Uint8Array>): Promise<void> {
    const decoder = new TextDecoder();
    let buffered = "";
    try {
      for await (const chunk of stream) {
        buffered += decoder.decode(chunk, { stream: true });
        // llama.cpp writes progress with \r, so both count as line breaks.
        let idx = buffered.search(/[\r\n]/);
        while (idx !== -1) {
          this.emitLine(buffered.slice(0, idx));
          buffered = buffered.slice(idx + 1);
          idx = buffered.search(/[\r\n]/);
        }
      }
    } catch {
      // Stream torn down with the process — nothing left to read.
    }
    if (buffered) this.emitLine(buffered);
  }

  private emitLine(line: string): void {
    if (!line.trim()) return;
    console.log(`[${this.config.name}] ${line}`);
    this.onLogLine?.(line);
  }

  // ─── Exit + shutdown ──────────────────────────────────────

  private handleExit(proc: Subprocess, exitCode: number | null, signalCode: number | null): void {
    // Only the tracked process gets to reset this entry. A stale child dying
    // late must not clear state — or release a GPU reservation — that now
    // belongs to a newer spawn. A null handle means terminate() already reset
    // us on the way out, so there is nothing to report.
    if (this.proc !== proc) {
      if (this.proc) {
        console.log(
          `[${this.config.name}] Ignoring exit of superseded pid ${proc.pid} ` +
            `(tracked pid ${this.proc.pid})`
        );
      }
      return;
    }

    const expected = this.state === "stopping";
    const how = signalCode ? `signal ${signalCode}` : `code ${exitCode}`;

    this.proc = null;
    this.state = "stopped";
    this.startedAt = null;
    if (!expected) this.lastError = `Backend exited with ${how}`;

    console.log(`[${this.config.name}] ${this.backend.label} exited (${how})`);

    this.failStartup?.(new Error(`${this.backend.label} exited with ${how} during startup`));

    try {
      this.onExit?.({ exitCode, signalCode, expected });
    } catch (err) {
      console.error(`[${this.config.name}] Exit handler failed:`, err);
    }
  }

  private async terminate(): Promise<void> {
    const proc = this.proc;
    if (!proc) {
      this.state = "stopped";
      return;
    }

    this.state = "stopping";
    const exited = proc.exited.then(
      () => undefined,
      () => undefined
    );

    // Baseline before the kill: afterwards there is nothing left to compare to.
    const vramBeforeMB = await gpuUsedMB(this.config);

    this.signal(proc, "SIGTERM");
    if (!(await settlesWithin(exited, STOP_GRACE_MS))) {
      console.warn(
        `[${this.config.name}] Still alive ${STOP_GRACE_MS / 1000}s after SIGTERM, sending SIGKILL`
      );
      this.signal(proc, "SIGKILL");
      await settlesWithin(exited, SIGKILL_GRACE_MS);
    }

    await this.reapGroup(proc.pid);

    // Same guard as handleExit: if a newer spawn took over while we were
    // killing this one, its handle is not ours to drop.
    if (this.proc === proc) {
      this.proc = null;
      this.state = "stopped";
    }
    await waitForGpuRelease(this.config, vramBeforeMB);
  }

  /**
   * The backend's children can outlive it: a vLLM worker wedged in a CUDA or
   * NCCL call never gets round to the SIGTERM that took its parent down, and
   * goes on holding its share of the GPU. The group is ours, so whatever is
   * still in it after the leader is gone gets SIGKILL.
   */
  private async reapGroup(pgid: number): Promise<void> {
    if (!SETSID_BIN) return;

    for (let attempt = 0; attempt < GROUP_REAP_ATTEMPTS; attempt++) {
      const survivors = processGroupMembers(pgid);
      if (survivors.length === 0) return;

      console.warn(
        `[${this.config.name}] ${survivors.length} process(es) outlived the backend ` +
          `(pids ${survivors.join(", ")} in group ${pgid}) — killing`
      );
      try {
        process.kill(-pgid, "SIGKILL");
      } catch {
        return; // Group went away between the scan and the signal.
      }
      await Bun.sleep(GROUP_REAP_INTERVAL_MS);
    }

    const remaining = processGroupMembers(pgid);
    if (remaining.length > 0) {
      console.error(
        `[${this.config.name}] pids ${remaining.join(", ")} survived SIGKILL and may still hold VRAM`
      );
    }
  }

  /**
   * Signal the backend's process group so forked workers go with it. Only safe
   * when we spawned through setsid — otherwise the child shares our group and
   * -pid would signal the router itself.
   */
  private signal(proc: Subprocess, sig: NodeJS.Signals): void {
    if (proc.exitCode !== null || proc.signalCode !== null) return;
    if (SETSID_BIN) {
      try {
        process.kill(-proc.pid, sig);
        return;
      } catch {
        // Group already gone, or the child never became a group leader.
      }
    }
    try {
      proc.kill(sig);
    } catch {
      // Already reaped.
    }
  }
}

// ─── Helpers ────────────────────────────────────────────────────

/** PID listening on a loopback port, or null when nothing is bound to it. */
async function portListener(port: number): Promise<number | null> {
  try {
    const proc = Bun.spawn({
      cmd: ["ss", "-ltnpH", `sport = :${port}`],
      stdout: "pipe",
      stderr: "ignore",
    });
    const [stdout] = await Promise.all([new Response(proc.stdout).text(), proc.exited]);
    const match = stdout.match(/pid=(\d+)/);
    return match ? parseInt(match[1], 10) : null;
  } catch {
    return null;
  }
}

/** Process group of a pid, or null if it is gone. */
function processGroupOf(pid: number): number | null {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, "utf-8");
    // comm may contain spaces and parentheses; the numeric fields resume
    // after the last ')': state, ppid, pgrp, ...
    const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    const pgrp = parseInt(fields[2], 10);
    return Number.isFinite(pgrp) ? pgrp : null;
  } catch {
    return null;
  }
}

/** Every live pid in a process group, the group leader included. */
function processGroupMembers(pgid: number): number[] {
  const members: number[] = [];
  for (const entry of readdirSync("/proc")) {
    const pid = parseInt(entry, 10);
    if (!Number.isFinite(pid) || String(pid) !== entry) continue;
    if (processGroupOf(pid) === pgid) members.push(pid);
  }
  return members;
}

/**
 * True when `pid` is the backend we spawned or one of its workers. Backends
 * are their own process group (see SETSID_BIN), so the group led by the child
 * we track is exactly "ours"; without setsid only the pid itself qualifies.
 */
function isOurs(pid: number, trackedPid: number): boolean {
  if (pid === trackedPid) return true;
  return SETSID_BIN ? processGroupOf(pid) === trackedPid : false;
}

/** True if the promise settles before the deadline. */
async function settlesWithin(promise: Promise<unknown>, ms: number): Promise<boolean> {
  let timer: Timer | undefined;
  const deadline = new Promise<boolean>((resolve) => {
    timer = setTimeout(() => resolve(false), ms);
  });
  try {
    return await Promise.race([promise.then(() => true), deadline]);
  } finally {
    clearTimeout(timer);
  }
}

/** Memory in use on a model's primary GPU, or null without nvidia-smi. */
async function gpuUsedMB(config: ModelConfig): Promise<number | null> {
  const gpu = config.gpus?.[0] ?? config.gpu;
  const info = (await getGpuInfo()).find((g) => g.index === gpu);
  return info ? info.usedMemoryMB : null;
}

/**
 * Bounded wait for the driver to hand VRAM back after a backend exits, so the
 * next model doesn't start into memory that is still being freed. Gives up
 * early once usage has clearly dropped or simply stopped moving; does nothing
 * at all when nvidia-smi can't see the GPU.
 */
async function waitForGpuRelease(config: ModelConfig, beforeMB: number | null): Promise<void> {
  if (beforeMB === null) return;

  const deadline = Date.now() + GPU_RECLAIM_TIMEOUT_MS;
  let previous = beforeMB;
  let stableMs = 0;

  while (Date.now() < deadline) {
    await Bun.sleep(500);
    const now = await gpuUsedMB(config);
    if (now === null || now <= beforeMB - GPU_RECLAIM_THRESHOLD_MB) return;

    if (now === previous) {
      stableMs += 500;
      if (stableMs >= 1500) return;
    } else {
      stableMs = 0;
      previous = now;
    }
  }
  console.log(
    `[${config.name}] GPU memory not reclaimed within ${GPU_RECLAIM_TIMEOUT_MS / 1000}s`
  );
}
