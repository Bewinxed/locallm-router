/**
 * Unified Proxy — single port, model routing via request payload.
 *
 * Intercepts incoming OpenAI-compatible requests, extracts the `model` field,
 * ensures the backend is downloaded + running, then proxies the request.
 *
 * Supports load-balanced models via `:balance`, `:gpu0`, `:gpu1` modifiers:
 *   - `model:balance`  → auto-route to the idle GPU instance
 *   - `model:gpu0`     → pin to GPU 0 instance
 *   - `model:gpu1`     → pin to GPU 1 instance
 *   - `model` (plain)  → use the primary GPU (config.gpu)
 *
 * Uses GpuLock to coordinate models sharing GPUs.
 */

import type { ModelConfig } from "./config";
import { getInternalPort, reloadModelConfig as reloadModelConfigFromDisk } from "./config";
import { createProcess, type BackendProcess, type ExitInfo } from "./process";
import { downloadModel, isLocalModel } from "./download";
import { EvictionCooldownError, type GpuLock, type GpuOwner } from "./gpu-lock";

// ─── Types ──────────────────────────────────────────────────────

export interface ModelEntry {
  config: ModelConfig;
  process: BackendProcess;
  internalPort: number;
  downloading: boolean;
  downloadError: string | null;
  /** Requests currently proxied to this entry, streams included. */
  inflight: number;
}

/** Parsed model name with optional modifier */
interface ParsedModelName {
  /** Base model name (without modifier) */
  baseName: string;
  /** Routing modifier: null (use primary), "balance", or "gpuN" */
  modifier: string | null;
  /** For gpuN modifiers, the target GPU index */
  targetGpu: number | null;
}

// ─── Balance Group ──────────────────────────────────────────────

/**
 * BalancedModelGroup wraps multiple ModelEntry instances (one per GPU)
 * for the same model, providing smart routing based on load.
 */
class BalancedModelGroup {
  readonly baseName: string;
  readonly primaryGpu: number;
  /** Map of GPU index -> ModelEntry for that GPU's instance */
  private instances: Map<number, ModelEntry> = new Map();
  /** Track in-flight request count per GPU */
  private inflight: Map<number, number> = new Map();

  constructor(baseName: string, primaryGpu: number) {
    this.baseName = baseName;
    this.primaryGpu = primaryGpu;
  }

  addInstance(gpu: number, entry: ModelEntry): void {
    this.instances.set(gpu, entry);
    this.inflight.set(gpu, 0);
  }

  getInstance(gpu: number): ModelEntry | undefined {
    return this.instances.get(gpu);
  }

  getPrimaryInstance(): ModelEntry | undefined {
    return this.instances.get(this.primaryGpu);
  }

  getAllInstances(): [number, ModelEntry][] {
    return [...this.instances.entries()];
  }

  getGpus(): number[] {
    return [...this.instances.keys()];
  }

  /**
   * Pick the best instance for a :balance request.
   * Priority:
   *   1. Running instance with 0 in-flight requests (truly idle)
   *   2. Running instance with fewer in-flight requests
   *   3. Stopped instance (will cold-start — prefer primary GPU)
   */
  pickIdleInstance(): { gpu: number; entry: ModelEntry } | undefined {
    let bestGpu: number | undefined;
    let bestInflight = Infinity;
    let bestIsRunning = false;

    for (const [gpu, entry] of this.instances) {
      const running = entry.process.status === "running";
      const count = this.inflight.get(gpu) || 0;

      // Prefer running instances over stopped ones
      if (running && !bestIsRunning) {
        bestGpu = gpu;
        bestInflight = count;
        bestIsRunning = true;
        continue;
      }

      // Among same category (both running or both stopped), pick fewer inflight
      if (running === bestIsRunning) {
        if (count < bestInflight) {
          bestGpu = gpu;
          bestInflight = count;
        } else if (count === bestInflight && gpu === this.primaryGpu) {
          // Tie-break: prefer primary GPU
          bestGpu = gpu;
        }
      }
    }

    if (bestGpu !== undefined) {
      return { gpu: bestGpu, entry: this.instances.get(bestGpu)! };
    }
    return undefined;
  }

  /** Increment in-flight count for a GPU */
  startRequest(gpu: number): void {
    this.inflight.set(gpu, (this.inflight.get(gpu) || 0) + 1);
  }

  /** Decrement in-flight count for a GPU */
  endRequest(gpu: number): void {
    const current = this.inflight.get(gpu) || 0;
    this.inflight.set(gpu, Math.max(0, current - 1));
  }

  /** Get status for management API */
  getStatus(): object {
    return {
      baseName: this.baseName,
      primaryGpu: this.primaryGpu,
      instances: [...this.instances.entries()].map(([gpu, entry]) => ({
        gpu,
        status: entry.process.status,
        inflight: this.inflight.get(gpu) || 0,
        internalPort: entry.internalPort,
        lastActivity: entry.process.lastActivity,
      })),
    };
  }
}

// ─── Helpers ────────────────────────────────────────────────────

/** Body framing + hop-by-hop headers. fetch() re-frames and decodes for us. */
const STRIPPED_REQUEST_HEADERS = [
  "host",
  "content-length",
  "content-encoding",
  "connection",
  "transfer-encoding",
];
const STRIPPED_RESPONSE_HEADERS = ["content-encoding", "content-length", "transfer-encoding"];

/** Runtime arg overrides accepted by /restart, and the shape of their values. */
const RESTART_OVERRIDE_FLAGS = [
  "-c",
  "--ctx-size",
  "--max-model-len",
  "--gpu-memory-utilization",
  "-ngl",
];
const RESTART_OVERRIDE_VALUE = /^[0-9.]+$/;

/** Consecutive failed probes before a running backend is declared dead. */
const HEALTH_FAILURES_BEFORE_STOP = 3;

function stripHeaders(headers: Headers, names: string[]): Headers {
  const out = new Headers(headers);
  for (const name of names) out.delete(name);
  return out;
}

/**
 * Runtime overrides for /restart, from `?context=N` and/or a JSON body.
 * Only known numeric flags are accepted — these end up on a command line.
 */
async function parseRestartOverrides(req: Request, url: URL): Promise<Record<string, string>> {
  const overrides: Record<string, string> = {};

  const context = url.searchParams.get("context");
  if (context) overrides["-c"] = context;

  const body = await req.json().catch(() => null);
  if (body?.overrides && typeof body.overrides === "object") {
    Object.assign(overrides, body.overrides);
  }

  for (const [flag, value] of Object.entries(overrides)) {
    if (!RESTART_OVERRIDE_FLAGS.includes(flag)) {
      throw new Error(
        `Unsupported override flag "${flag}" (allowed: ${RESTART_OVERRIDE_FLAGS.join(", ")})`
      );
    }
    if (typeof value !== "string" || !RESTART_OVERRIDE_VALUE.test(value)) {
      throw new Error(`Override "${flag}" must be a numeric string`);
    }
  }

  return overrides;
}

function isStreamingResponse(response: Response): boolean {
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("text/event-stream") || contentType.includes("stream");
}

/**
 * Wrap a streamed body so `onDone` runs when it finishes, errors, or the
 * client cancels. Bun never calls a TransformStream's `cancel`, so the hook
 * hangs off a reader instead.
 */
function trackStreamEnd(
  body: ReadableStream<Uint8Array>,
  onDone: () => void
): ReadableStream<Uint8Array> {
  const reader = body.getReader();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          controller.close();
          onDone();
          return;
        }
        controller.enqueue(value);
      } catch (err) {
        onDone();
        controller.error(err);
      }
    },
    cancel(reason) {
      onDone();
      return reader.cancel(reason);
    },
  });
}

function makeGpuOwner(entry: ModelEntry): GpuOwner {
  return {
    name: entry.config.name,
    gpu: entry.config.gpu,
    gpus: entry.config.gpus,
    config: entry.config,
    isRunning: () => entry.process.status === "running",
    stopProcess: () => entry.process.stop(),
  };
}

function isEntryRunning(entry: ModelEntry): boolean {
  return entry.process.status === "running" || entry.process.status === "starting";
}

/**
 * Parse a model name that may include a routing modifier.
 * Examples:
 *   "qwen3.5-35b-a3b"          → { baseName: "qwen3.5-35b-a3b", modifier: null, targetGpu: null }
 *   "qwen3.5-35b-a3b:balance"  → { baseName: "qwen3.5-35b-a3b", modifier: "balance", targetGpu: null }
 *   "qwen3.5-35b-a3b:gpu0"    → { baseName: "qwen3.5-35b-a3b", modifier: "gpu0", targetGpu: 0 }
 *   "qwen3.5-35b-a3b:gpu1"    → { baseName: "qwen3.5-35b-a3b", modifier: "gpu1", targetGpu: 1 }
 */
function parseModelName(name: string): ParsedModelName {
  // Match :balance or :gpuN at the end
  const match = name.match(/^(.+):(balance|gpu(\d+))$/);
  if (!match) {
    return { baseName: name, modifier: null, targetGpu: null };
  }

  const baseName = match[1];
  const modifier = match[2];
  const targetGpu = match[3] !== undefined ? parseInt(match[3], 10) : null;

  return { baseName, modifier, targetGpu };
}

/** Next available port for balance instances (starts after normal model ports) */
const BALANCE_PORT_BASE = 41000;
let nextBalancePort = BALANCE_PORT_BASE;

function allocateBalancePort(): number {
  return nextBalancePort++;
}

// ─── Unified Proxy ──────────────────────────────────────────────

export class UnifiedProxy {
  private models: Map<string, ModelEntry> = new Map();
  private gpuLock: GpuLock;
  private server: ReturnType<typeof Bun.serve> | null = null;
  private idleTimers: Map<string, Timer> = new Map();
  private idleSweepTimer: Timer | null = null;
  private healthTimer: Timer | null = null;
  /** Consecutive failed health probes per model name */
  private healthFailures: Map<string, number> = new Map();

  /** Serializes lifecycle transitions per model — see withLifecycle(). */
  private lifecycleQueues: Map<ModelEntry, Promise<unknown>> = new Map();

  /** Balance groups: base model name -> BalancedModelGroup */
  private balanceGroups: Map<string, BalancedModelGroup> = new Map();
  /** Set of base model names that have balance mode enabled */
  private balancedModels: Set<string> = new Set();

  constructor(configs: ModelConfig[], gpuLock: GpuLock) {
    this.gpuLock = gpuLock;

    // Create entries for each model
    for (let i = 0; i < configs.length; i++) {
      const config = configs[i];
      const internalPort = getInternalPort(i);
      const entry = this.createEntry(config, internalPort);
      this.models.set(config.name, entry);

      // Register aliases
      if (config.aliases) {
        for (const alias of config.aliases) {
          this.models.set(alias, entry);
        }
      }

      // Register with GPU lock
      this.gpuLock.register(makeGpuOwner(entry));

      // ─── Balance mode setup ─────────────────────────────
      if (config.balanceGpus && config.balanceGpus.length > 1) {
        this.balancedModels.add(config.name);
        if (config.aliases) {
          for (const alias of config.aliases) {
            this.balancedModels.add(alias);
          }
        }

        const group = new BalancedModelGroup(config.name, config.gpu);

        // The primary GPU uses the existing entry
        group.addInstance(config.gpu, entry);

        // Create additional entries for non-primary GPUs
        for (const gpu of config.balanceGpus) {
          if (gpu === config.gpu) continue; // Already added primary

          // Build GPU-specific config
          const gpuConfig: ModelConfig = {
            ...config,
            // Internal name to distinguish instances (not exposed to API)
            name: `${config.name}__gpu${gpu}`,
            gpu,
            gpus: undefined, // Single GPU per balance instance
          };

          // Apply per-GPU arg overrides if specified
          if (config.balanceGpuArgs && config.balanceGpuArgs[gpu.toString()]) {
            gpuConfig.extraArgs = config.balanceGpuArgs[gpu.toString()];
          }

          const balancePort = allocateBalancePort();
          const balanceEntry = this.createEntry(gpuConfig, balancePort);

          group.addInstance(gpu, balanceEntry);

          // Register with GPU lock under internal name
          this.gpuLock.register(makeGpuOwner(balanceEntry));

          console.log(
            `[balance] Created instance ${gpuConfig.name} on GPU ${gpu}, port ${balancePort}`
          );
        }

        this.balanceGroups.set(config.name, group);

        // Also map aliases to the same group
        if (config.aliases) {
          for (const alias of config.aliases) {
            this.balanceGroups.set(alias, group);
          }
        }

        console.log(
          `[balance] Model ${config.name}: balance group on GPUs [${config.balanceGpus.join(", ")}], primary GPU ${config.gpu}`
        );
      }
    }
  }

  private createEntry(config: ModelConfig, internalPort: number): ModelEntry {
    let entry: ModelEntry;
    const proc = createProcess(config, internalPort, (info) => this.handleBackendExit(entry, info));
    entry = {
      config,
      process: proc,
      internalPort,
      downloading: false,
      downloadError: null,
      inflight: 0,
    };
    return entry;
  }

  /**
   * Every backend exit lands here, crash or not: the GPU reservation goes back
   * to the lock so the next request can cold-start instead of waiting on VRAM
   * that nothing is using.
   */
  private handleBackendExit(entry: ModelEntry, info: ExitInfo): void {
    this.clearIdleTimer(entry.config.name);
    this.healthFailures.delete(entry.config.name);
    entry.inflight = 0;
    this.gpuLock.notifyStopped(makeGpuOwner(entry));
    if (!info.expected) {
      console.warn(
        `[${entry.config.name}] Backend exited unexpectedly ` +
          `(${info.signalCode ?? `code ${info.exitCode}`}) — GPU reservation released`
      );
    }
  }

  /**
   * One lifecycle transition at a time per model: lazy start, management
   * start/stop, restart, config reload, idle unload and the health sweep all
   * queue here. Two of them driving the same BackendProcess is what leaves an
   * orphaned backend holding the internal port. The per-GPU lock doesn't cover
   * this — it arbitrates VRAM between models, not transitions within one.
   *
   * Eviction (GpuOwner.stopProcess) deliberately stays outside: it runs while
   * the incoming model holds its own slot, and taking the victim's lock there
   * would let two models on the same card wait on each other.
   */
  private withLifecycle<T>(entry: ModelEntry, transition: () => Promise<T>): Promise<T> {
    const previous = this.lifecycleQueues.get(entry) ?? Promise.resolve();
    const run = previous.then(transition, transition);
    this.lifecycleQueues.set(entry, run.catch(() => {}));
    return run;
  }

  private getManagedEntries(): ModelEntry[] {
    const entries = new Map<string, ModelEntry>();

    for (const entry of this.models.values()) {
      entries.set(entry.config.name, entry);
    }

    for (const group of this.balanceGroups.values()) {
      for (const [, entry] of group.getAllInstances()) {
        entries.set(entry.config.name, entry);
      }
    }

    return [...entries.values()];
  }

  private stopEntry(entry: ModelEntry): Promise<void> {
    return this.withLifecycle(entry, () => this.stopEntryLocked(entry));
  }

  private async stopEntryLocked(entry: ModelEntry): Promise<void> {
    this.clearIdleTimer(entry.config.name);
    if (!isEntryRunning(entry) && entry.process.status !== "sleeping") return;
    await entry.process.stop();
    this.gpuLock.notifyStopped(makeGpuOwner(entry));
  }

  /** Try to sleep instead of kill (vLLM only). Falls back to full stop. */
  private sleepOrStop(entry: ModelEntry): Promise<void> {
    return this.withLifecycle(entry, async () => {
      if (entry.process.supportsSleep && entry.process.status === "running") {
        const slept = await entry.process.sleep();
        if (slept) {
          this.clearIdleTimer(entry.config.name);
          this.gpuLock.notifyStopped(makeGpuOwner(entry));
          return;
        }
      }
      await this.stopEntryLocked(entry);
    });
  }

  private async stopBalanceShadows(baseName: string, primaryEntry?: ModelEntry): Promise<void> {
    const group = this.balanceGroups.get(baseName);
    if (!group) return;

    for (const [, entry] of group.getAllInstances()) {
      if (primaryEntry && entry.config.name === primaryEntry.config.name) continue;
      await this.stopEntry(entry);
    }
  }

  /** Start the unified HTTP server */
  start(port: number) {
    this.server = Bun.serve({
      port,
      idleTimeout: 255,
      maxRequestBodySize: 64 * 1024 * 1024,
      fetch: (req) => this.handleRequest(req),
      error: (err) => {
        console.error("[proxy] Unhandled request error:", err);
        return Response.json({ error: "Internal proxy error", details: String(err) }, { status: 500 });
      },
    });
    console.log(`Unified proxy listening on port ${port}`);
    return this;
  }

  /** Get a model entry, resolving aliases (no modifier parsing) */
  private getEntry(modelName: string): ModelEntry | undefined {
    return this.models.get(modelName);
  }

  private async handleRequest(req: Request): Promise<Response> {
    const url = new URL(req.url);

    // ─── OpenAI /v1/models endpoint ─────────────────────────
    if (url.pathname === "/v1/models" && req.method === "GET") {
      return this.handleListModels();
    }

    // ─── Health / management endpoints ──────────────────────
    if (url.pathname === "/health") {
      return Response.json({ status: "ok", models: this.getModelsStatus() });
    }

    if (url.pathname === "/manager" || url.pathname.startsWith("/manager/")) {
      return this.handleManagement(req);
    }

    // ─── API requests — route by model field ────────────────

    // Extract `model` from the request body based on content type.
    // Only JSON bodies are re-encoded (model name normalization); anything
    // else — multipart audio uploads especially — is forwarded byte for byte.
    let body: any = null;
    let rawBody: ArrayBuffer | null = null;
    let jsonBody = false;
    if (req.method !== "GET" && req.method !== "HEAD") {
      const contentType = req.headers.get("content-type") || "";
      rawBody = await req.arrayBuffer();

      if (contentType.includes("multipart/form-data")) {
        try {
          const formData = await new Response(rawBody, {
            headers: { "content-type": contentType },
          }).formData();
          const model = formData.get("model");
          if (model && typeof model === "string") {
            body = { model };
          }
        } catch {
          // Malformed multipart — pass through, backend will reject
        }
      } else {
        try {
          body = JSON.parse(new TextDecoder().decode(rawBody));
          jsonBody = contentType.includes("json");
        } catch {
          // Not JSON — pass through
        }
      }
    }

    const modelName: string | null = body?.model ?? null;
    if (!modelName) {
      return Response.json(
        {
          error: "Missing 'model' field in request body",
          available_models: this.getModelNames(),
          hint: "Set the 'model' field in your request payload to one of the available models. " +
            "Append :balance for auto load-balancing, :gpu0/:gpu1 to pin to a specific GPU.",
        },
        { status: 400 }
      );
    }

    // ─── Parse modifier and route ───────────────────────────
    const parsed = parseModelName(modelName);
    const baseEntry = this.getEntry(parsed.baseName);

    if (!baseEntry) {
      return Response.json(
        {
          error: `Unknown model: ${parsed.baseName}`,
          available_models: this.getModelNames(),
        },
        { status: 404 }
      );
    }

    // Check if this model has a balance group
    const group = this.balanceGroups.get(parsed.baseName);

    // ─── Balance routing ────────────────────────────────────
    if (parsed.modifier && group) {
      return this.handleBalancedRequest(req, body, rawBody, jsonBody, parsed, group, baseEntry);
    }

    // ─── Modifier used but model doesn't support it ─────────
    if (parsed.modifier && !group) {
      return Response.json(
        {
          error: `Model ${parsed.baseName} does not support :${parsed.modifier} modifier (no balanceGpus configured)`,
          available_models: this.getModelNames(),
        },
        { status: 400 }
      );
    }

    // ─── Standard single-instance routing ───────────────────
    // For balanced models without a modifier, route to primary GPU
    // but also track inflight so :balance requests see the load.
    if (group) {
      // Treat plain model name as an implicit :gpuN for the primary GPU
      const implicitParsed: ParsedModelName = {
        baseName: parsed.baseName,
        modifier: `gpu${baseEntry.config.gpu}`,
        targetGpu: baseEntry.config.gpu,
      };
      return this.handleBalancedRequest(req, body, rawBody, jsonBody, implicitParsed, group, baseEntry);
    }

    try {
      await this.ensureRunning(baseEntry);
    } catch (err) {
      return this.startFailureResponse(err);
    }

    // Rewrite model name in body to match what the backend expects
    if (jsonBody && body && body.model !== baseEntry.config.name) {
      body.model = baseEntry.config.name;
      rawBody = new TextEncoder().encode(JSON.stringify(body)).buffer as ArrayBuffer;
    }

    return this.proxyRequest(req, baseEntry, rawBody);
  }

  /** 503 for a model that could not start, with Retry-After during eviction cooldown. */
  private startFailureResponse(err: unknown): Response {
    if (err instanceof EvictionCooldownError) {
      return Response.json(
        { error: err.message, retry_after_seconds: err.retryAfterSeconds },
        { status: 503, headers: { "Retry-After": String(err.retryAfterSeconds) } }
      );
    }
    return Response.json(
      { error: "Model failed to start", details: String(err) },
      { status: 503 }
    );
  }

  /**
   * Handle a request with a balance modifier (:balance, :gpuN).
   * Selects the right instance, ensures it's running, tracks inflight, proxies.
   */
  private async handleBalancedRequest(
    req: Request,
    body: any,
    rawBody: ArrayBuffer | null,
    jsonBody: boolean,
    parsed: ParsedModelName,
    group: BalancedModelGroup,
    baseEntry: ModelEntry
  ): Promise<Response> {
    let targetEntry: ModelEntry;
    let targetGpu: number;

    if (parsed.modifier === "balance") {
      // Auto-route to idle instance
      const pick = group.pickIdleInstance();
      if (!pick) {
        return Response.json(
          { error: "No available instances in balance group" },
          { status: 503 }
        );
      }
      targetEntry = pick.entry;
      targetGpu = pick.gpu;
      console.log(
        `[balance] ${parsed.baseName}:balance → GPU ${targetGpu} ` +
          `(status: ${targetEntry.process.status})`
      );
    } else if (parsed.targetGpu !== null) {
      // Pin to specific GPU
      const instance = group.getInstance(parsed.targetGpu);
      if (!instance) {
        return Response.json(
          {
            error: `Model ${parsed.baseName} has no instance on GPU ${parsed.targetGpu}`,
            available_gpus: group.getGpus(),
          },
          { status: 400 }
        );
      }
      targetEntry = instance;
      targetGpu = parsed.targetGpu;
      console.log(
        `[balance] ${parsed.baseName}:gpu${targetGpu} → pinned`
      );
    } else {
      // Shouldn't happen (modifier is set but not balance or gpuN)
      targetEntry = baseEntry;
      targetGpu = baseEntry.config.gpu;
    }

    // Ensure the target instance is running
    try {
      await this.ensureRunning(targetEntry);
    } catch (err) {
      // If the target fails, try fallback to another instance (for :balance only)
      if (parsed.modifier === "balance") {
        console.warn(
          `[balance] GPU ${targetGpu} failed to start, trying fallback...`
        );
        for (const [gpu, entry] of group.getAllInstances()) {
          if (gpu === targetGpu) continue;
          try {
            await this.ensureRunning(entry);
            targetEntry = entry;
            targetGpu = gpu;
            console.log(`[balance] Fallback to GPU ${gpu} succeeded`);
            break;
          } catch {
            continue;
          }
        }
        if (targetEntry.process.status !== "running") {
          return Response.json(
            { error: "All balance instances failed to start", details: String(err) },
            { status: 503 }
          );
        }
      } else {
        return this.startFailureResponse(err);
      }
    }

    // Rewrite model name to the base name (strip modifier) so the backend recognizes it.
    // The backend's -a (alias) flag is set to the base model name for the primary,
    // and the internal name for secondary instances. We need to match what llama-server expects.
    if (jsonBody && body) {
      // llama-server uses -a flag to set served model name.
      // Primary instance: served as config.name (e.g. "qwen3.5-35b-a3b")
      // Secondary instances: served as internal name (e.g. "qwen3.5-35b-a3b__gpu0")
      // We rewrite to whatever the target's config.name is.
      body.model = targetEntry.config.name;
      rawBody = new TextEncoder().encode(JSON.stringify(body)).buffer as ArrayBuffer;
    }

    // Track inflight and proxy
    group.startRequest(targetGpu);
    let counted = true;
    const endRequest = () => {
      if (!counted) return;
      counted = false;
      group.endRequest(targetGpu);
    };
    req.signal.addEventListener("abort", endRequest);

    try {
      const response = await this.proxyRequest(req, targetEntry, rawBody);

      // For streaming responses, wrap the body so the count drops when the
      // stream ends — or when the client walks away mid-stream.
      if (response.body && isStreamingResponse(response)) {
        return new Response(trackStreamEnd(response.body, endRequest), {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }

      // Non-streaming: already fully consumed, decrement now
      endRequest();
      return response;
    } catch (err) {
      endRequest();
      throw err;
    }
  }

  /** Public wrapper for management API — starts a model with GPU lock + idle timer */
  async startModel(entry: ModelEntry): Promise<void> {
    await this.ensureRunning(entry);
  }

  /** Public wrapper for management API — stops a model with GPU lock release + balance cleanup */
  async stopModel(entry: ModelEntry): Promise<void> {
    await this.stopEntry(entry);
    await this.stopBalanceShadows(entry.config.name, entry);
  }

  /** Ensure a model is downloaded and its backend process is running */
  private ensureRunning(entry: ModelEntry): Promise<void> {
    return this.withLifecycle(entry, () => this.ensureRunningLocked(entry));
  }

  private async ensureRunningLocked(entry: ModelEntry): Promise<void> {
    // Step 1: Download if needed (HF models only)
    if (!isLocalModel(entry.config.modelPath)) {
      if (entry.downloading) {
        // Wait for existing download to complete
        await new Promise<void>((resolve, reject) => {
          const check = setInterval(() => {
            if (!entry.downloading) {
              clearInterval(check);
              if (entry.downloadError) reject(new Error(entry.downloadError));
              else resolve();
            }
          }, 1000);
        });
      } else {
        entry.downloading = true;
        entry.downloadError = null;
        try {
          await downloadModel(
            entry.config.modelPath,
            (line) => {
              console.log(`[${entry.config.name}] ${line}`);
            },
            entry.config.modelFile
          );
        } catch (err) {
          entry.downloadError = String(err);
          entry.downloading = false;
          throw err;
        }
        entry.downloading = false;
      }
    }

    // Step 2: Start backend if not running
    if (entry.process.status !== "running") {
      const owner = makeGpuOwner(entry);
      const isSleeping = entry.process.status === "sleeping";

      this.gpuLock.assertNotCoolingDown(owner);
      await this.gpuLock.acquire(owner);

      if (isSleeping) {
        console.log(`[${entry.config.name}] Waking from sleep mode...`);
      } else {
        const gpuStr = entry.config.gpus
          ? `GPUs ${entry.config.gpus.join(",")}`
          : `GPU ${entry.config.gpu}`;
        console.log(`[${entry.config.name}] Cold start — launching ${entry.config.backend || "vllm"} on ${gpuStr}...`);
      }

      // Snapshot after acquire: any eviction has already released its VRAM,
      // so the post-start delta is this model's alone.
      const preVRAM = isSleeping ? 0 : await this.gpuLock.snapshotVRAM(owner);
      await entry.process.start();

      if (!isSleeping) {
        this.gpuLock.recordObservedVRAM(owner, preVRAM).catch(() => {});
      }
    }

    entry.process.touchActivity();
    this.gpuLock.touchActivity(makeGpuOwner(entry));
    this.resetIdleTimer(entry);
  }

  /** Proxy a request to the backend process */
  private async proxyRequest(
    req: Request,
    entry: ModelEntry,
    rawBody: ArrayBuffer | null
  ): Promise<Response> {
    const url = new URL(req.url);
    const targetUrl = `${entry.process.getInternalUrl()}${url.pathname}${url.search}`;

    const proxyReq: RequestInit = {
      method: req.method,
      headers: stripHeaders(req.headers, STRIPPED_REQUEST_HEADERS),
      // Client aborts have to reach the backend, otherwise it keeps decoding
      // tokens for a response nobody will read.
      signal: req.signal,
    };

    if (rawBody) {
      proxyReq.body = rawBody;
    }

    entry.inflight++;
    let counted = true;
    const done = () => {
      if (!counted) return;
      counted = false;
      entry.inflight = Math.max(0, entry.inflight - 1);
      entry.process.touchActivity();
      this.gpuLock.touchActivity(makeGpuOwner(entry));
      this.resetIdleTimer(entry);
    };
    // A client that hangs up mid-stream never reaches the stream's flush.
    req.signal.addEventListener("abort", done);

    let response: Response;
    try {
      response = await fetch(targetUrl, proxyReq);
    } catch (err) {
      done();
      if (req.signal.aborted) {
        return Response.json({ error: "Client aborted the request" }, { status: 499 });
      }
      console.error(`[${entry.config.name}] Upstream request failed:`, err);
      return Response.json(
        {
          error: "Backend request failed",
          model: entry.config.name,
          details: err instanceof Error ? err.message : String(err),
        },
        { status: 502 }
      );
    }

    const headers = stripHeaders(response.headers, STRIPPED_RESPONSE_HEADERS);

    if (response.body && isStreamingResponse(response)) {
      // The idle timer must not fire mid-stream, so activity is touched when
      // the stream finishes or the client cancels — not when it started.
      return new Response(trackStreamEnd(response.body, done), {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }

    try {
      const buffer = await response.arrayBuffer();
      return new Response(buffer, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    } catch (err) {
      if (req.signal.aborted) {
        return Response.json({ error: "Client aborted the request" }, { status: 499 });
      }
      throw err;
    } finally {
      done();
    }
  }

  /** OpenAI /v1/models response listing all configured models + balance modifiers */
  private handleListModels(): Response {
    const now = Math.floor(Date.now() / 1000);
    const data = [];

    for (const config of this.getUniqueConfigs()) {
      const entry = this.models.get(config.name);
      const group = this.balanceGroups.get(config.name);

      const modelData: any = {
        id: config.name,
        object: "model",
        created: now,
        owned_by: config.backend || "vllm",
        root: config.modelPath,
        status: entry?.process.status || "stopped",
        gpu: config.gpus ? config.gpus : [config.gpu],
        aliases: config.aliases || [],
      };

      // Advertise balance modifiers if available
      if (group) {
        const gpus = group.getGpus();
        modelData.balance = {
          enabled: true,
          gpus,
          modifiers: [
            `${config.name}:balance`,
            ...gpus.map((g) => `${config.name}:gpu${g}`),
          ],
          instances: group.getAllInstances().map(([gpu, e]) => ({
            gpu,
            status: e.process.status,
          })),
        };
      }

      data.push(modelData);
    }

    return Response.json({ object: "list", data });
  }

  /** Get unique model configs (deduplicated from aliases) */
  private getUniqueConfigs(): ModelConfig[] {
    const seen = new Set<string>();
    const configs: ModelConfig[] = [];
    for (const [, entry] of this.models) {
      if (!seen.has(entry.config.name)) {
        seen.add(entry.config.name);
        configs.push(entry.config);
      }
    }
    return configs;
  }

  /** Get model names including balance modifiers (for error messages) */
  private getModelNames(): string[] {
    const names: string[] = [];
    for (const config of this.getUniqueConfigs()) {
      names.push(config.name);
      const group = this.balanceGroups.get(config.name);
      if (group) {
        names.push(`${config.name}:balance`);
        for (const gpu of group.getGpus()) {
          names.push(`${config.name}:gpu${gpu}`);
        }
      }
    }
    return names;
  }

  /** Get status of all models */
  private getModelsStatus() {
    return this.getUniqueConfigs().map((config) => {
      const entry = this.models.get(config.name)!;
      const group = this.balanceGroups.get(config.name);
      return {
        name: config.name,
        backend: config.backend || "vllm",
        modelPath: config.modelPath,
        status: entry.process.status,
        lastActivity: entry.process.lastActivity,
        inflight: entry.inflight,
        downloading: entry.downloading,
        downloadError: entry.downloadError,
        aliases: config.aliases || [],
        balance: group ? group.getStatus() : null,
        gpu: config.gpu,
        gpus: config.gpus,
        priority: config.priority ?? 0,
        sleepDelay: config.sleepDelay ?? 0,
      };
    });
  }

  /** Get balance groups status for management API */
  private getBalanceGroupsStatus(): object[] {
    const statuses: object[] = [];
    const seen = new Set<string>();
    for (const [name, group] of this.balanceGroups) {
      if (seen.has(group.baseName)) continue;
      seen.add(group.baseName);
      statuses.push(group.getStatus());
    }
    return statuses;
  }

  // ─── Management API ───────────────────────────────────────

  /**
   * The whole management surface. Mounted twice: under /manager/* on the proxy
   * port, and at the root of the management port. Same handler, same shapes.
   */
  async handleManagement(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const path = url.pathname.replace(/^\/manager/, "") || "/";
    const isPost = req.method === "POST";

    if (path === "/status" && !isPost) {
      return Response.json({
        models: this.getModelsStatus(),
        gpus: await this.gpuLock.status(),
        balanceGroups: this.getBalanceGroupsStatus(),
      });
    }

    if (path === "/stop-all" && isPost) {
      for (const entry of this.getManagedEntries()) {
        await this.stopEntry(entry);
      }
      return Response.json({ status: "all stopped" });
    }

    const match = path.match(/^\/(status|start|stop|restart|reload-config)\/(.+)$/);
    if (match) {
      const [, action, encodedName] = match;
      const modelName = decodeURIComponent(encodedName);
      const entry = this.getEntry(modelName);
      if (!entry) {
        return Response.json({ error: `Unknown model: ${modelName}` }, { status: 404 });
      }
      const name = entry.config.name;

      if (action === "status") return Response.json(entry.process.getState());
      if (!isPost) return Response.json({ error: `${action} requires POST` }, { status: 405 });

      if (action === "start") {
        try {
          await this.startModel(entry);
          return Response.json({ status: "running", model: name });
        } catch (err) {
          if (err instanceof EvictionCooldownError) return this.startFailureResponse(err);
          return Response.json({ error: String(err) }, { status: 500 });
        }
      }

      if (action === "stop") {
        await this.stopModel(entry);
        return Response.json({ status: "stopped", model: name });
      }

      if (action === "reload-config") {
        const newConfig = await this.reloadModelConfig(name);
        if (!newConfig) {
          return Response.json(
            { error: `Model ${name} not found in config file` },
            { status: 404 }
          );
        }
        return Response.json({ status: "reloaded", model: name, config: newConfig });
      }

      let overrides: Record<string, string>;
      try {
        overrides = await parseRestartOverrides(req, url);
      } catch (err) {
        return Response.json({ error: (err as Error).message }, { status: 400 });
      }
      const result = await this.restartModel(
        name,
        Object.keys(overrides).length > 0 ? overrides : undefined
      );
      return Response.json(result, { status: result.status === "error" ? 500 : 200 });
    }

    return Response.json({
      endpoints: {
        "/status": "GET — Status of all models + GPUs",
        "/status/<model>": "GET — Status of a single model",
        "/stop-all": "POST — Stop all running models",
        "/start/<model>": "POST — Start a specific model",
        "/stop/<model>": "POST — Stop a specific model",
        "/reload-config/<model>": "POST — Reload config from disk for a model",
        "/restart/<model>":
          "POST — Restart a model with fresh config. Query: ?context=N. " +
          `Body: { overrides: { "-c": "N" } }. Allowed flags: ${RESTART_OVERRIDE_FLAGS.join(", ")}`,
      },
      models: this.getManagedEntries().map((e) => ({
        name: e.config.name,
        status: e.process.status,
        backend: e.config.backend || "vllm",
      })),
    });
  }

  // ─── Idle timer management ────────────────────────────────

  private resetIdleTimer(entry: ModelEntry) {
    const timerKey = entry.config.name;
    this.clearIdleTimer(timerKey);

    const timer = setTimeout(async () => {
      try {
        // A long generation is activity, even though nothing has been touched
        // since the request started.
        if (entry.inflight > 0) {
          this.resetIdleTimer(entry);
          return;
        }
        const idleTime = Date.now() - entry.process.lastActivity;
        if (idleTime >= entry.config.idleTimeout && entry.process.status === "running") {
          const sleepDelay = entry.config.sleepDelay ?? 0;
          if (sleepDelay > 0) {
            console.log(
              `[${entry.config.name}] Idle for ${Math.round(idleTime / 1000 / 60)}min, sleep in ${sleepDelay / 1000}s...`
            );
            await new Promise((r) => setTimeout(r, sleepDelay));
            if (entry.process.status !== "running") return;
            const freshIdle = Date.now() - entry.process.lastActivity;
            if (freshIdle < entry.config.idleTimeout) return;
          } else {
            console.log(
              `[${entry.config.name}] Idle for ${Math.round(idleTime / 1000 / 60)}min, sleeping/stopping...`
            );
          }
          await this.sleepOrStop(entry);
          await this.stopBalanceShadows(entry.config.name, entry);
        }
      } catch (err) {
        console.error(`[${entry.config.name}] Error during idle unload:`, err);
        try {
          await this.stopEntry(entry);
        } catch { /* already logged */ }
      }
    }, entry.config.idleTimeout);

    this.idleTimers.set(timerKey, timer);
  }

  private clearIdleTimer(name: string) {
    const timer = this.idleTimers.get(name);
    if (timer) {
      clearTimeout(timer);
      this.idleTimers.delete(name);
    }
  }

  /**
   * Periodic sweep to catch any models that missed their idle timer.
   * Re-armed after each pass so a slow sweep can't overlap itself.
   */
  startIdleSweep(intervalMs = 60_000) {
    if (this.idleSweepTimer) clearTimeout(this.idleSweepTimer);

    const sweep = async () => {
      for (const entry of this.getManagedEntries()) {
        if (entry.process.status !== "running" || entry.inflight > 0) continue;
        const idleTime = Date.now() - entry.process.lastActivity;
        if (idleTime >= entry.config.idleTimeout) {
          console.log(
            `[idle-sweep] ${entry.config.name} idle for ${Math.round(idleTime / 1000 / 60)}min, sleeping/stopping...`
          );
          try {
            await this.sleepOrStop(entry);
            await this.stopBalanceShadows(entry.config.name, entry);
          } catch (err) {
            console.error(`[idle-sweep] Error stopping ${entry.config.name}:`, err);
          }
        }
      }
      this.idleSweepTimer = setTimeout(sweep, intervalMs);
    };

    this.idleSweepTimer = setTimeout(sweep, intervalMs);
  }

  /**
   * Periodically probe every running backend. A backend that fails three
   * probes in a row is gone as far as clients are concerned (OOM-killed,
   * wedged, crashed): force it stopped so the GPU reservation is released and
   * the next request cold-starts it cleanly.
   */
  startHealthChecks(intervalMs: number) {
    if (this.healthTimer) clearTimeout(this.healthTimer);
    if (!intervalMs || intervalMs <= 0) return;

    const check = async () => {
      for (const entry of this.getManagedEntries()) {
        const name = entry.config.name;
        if (entry.process.status !== "running") {
          this.healthFailures.delete(name);
          continue;
        }

        if (await entry.process.healthCheck()) {
          this.healthFailures.delete(name);
          continue;
        }

        const failures = (this.healthFailures.get(name) ?? 0) + 1;
        this.healthFailures.set(name, failures);
        console.warn(`[health] ${name} failed health check (${failures}/${HEALTH_FAILURES_BEFORE_STOP})`);

        if (failures >= HEALTH_FAILURES_BEFORE_STOP) {
          console.error(`[health] ${name} is unhealthy — stopping it`);
          this.healthFailures.delete(name);
          try {
            await this.stopEntry(entry);
          } catch (err) {
            console.error(`[health] Failed to stop ${name}:`, err);
          }
        }
      }
      this.healthTimer = setTimeout(check, intervalMs);
    };

    this.healthTimer = setTimeout(check, intervalMs);
  }

  /**
   * Reload a model's config from disk and update the in-memory entry.
   * The model must be stopped first (or will be stopped automatically).
   * Returns the new config, or null if model not found in config file.
   */
  async reloadModelConfig(modelName: string): Promise<ModelConfig | null> {
    const entry = this.models.get(modelName);
    if (!entry) {
      console.warn(`[reload] Model ${modelName} not found in proxy`);
      return null;
    }
    return this.withLifecycle(entry, () => this.reloadModelConfigLocked(entry));
  }

  private async reloadModelConfigLocked(entry: ModelEntry): Promise<ModelConfig | null> {
    const modelName = entry.config.name;

    // Re-read from disk
    const newConfig = reloadModelConfigFromDisk(modelName);
    if (!newConfig) {
      console.warn(`[reload] Model ${modelName} not found in config file`);
      return null;
    }

    // Stop if running — stopEntry also covers "sleeping", which updateConfig
    // would otherwise reject.
    if (entry.process.status !== "stopped") {
      console.log(`[reload] Stopping ${modelName} before config update...`);
      await this.stopEntryLocked(entry);
    }

    // Update in-memory config
    entry.config = newConfig;
    entry.process.updateConfig(newConfig);

    // Also update balance group instances if applicable
    const group = this.balanceGroups.get(modelName);
    if (group && newConfig.balanceGpus) {
      for (const [gpu, balanceEntry] of group.getAllInstances()) {
        if (gpu === newConfig.gpu) continue; // Primary already updated above

        // Build GPU-specific config
        const gpuConfig: ModelConfig = {
          ...newConfig,
          name: `${newConfig.name}__gpu${gpu}`,
          gpu,
          gpus: undefined,
        };

        // Apply per-GPU arg overrides if specified
        if (newConfig.balanceGpuArgs && newConfig.balanceGpuArgs[gpu.toString()]) {
          gpuConfig.extraArgs = newConfig.balanceGpuArgs[gpu.toString()];
        }

        // Stop balance instance if running
        if (balanceEntry.process.status !== "stopped") {
          await this.stopEntry(balanceEntry);
        }

        balanceEntry.config = gpuConfig;
        balanceEntry.process.updateConfig(gpuConfig);
      }
    }

    console.log(`[reload] Config updated for ${modelName}`);
    return newConfig;
  }

  /**
   * Restart a model: stop it, reload config from disk, then start it.
   * Accepts optional argOverrides to patch extraArgs at runtime (e.g., { "-c": "32768" }).
   * These overrides are NOT persisted to config.json — they only affect this run.
   */
  async restartModel(
    modelName: string,
    argOverrides?: Record<string, string>
  ): Promise<{ status: string; config?: ModelConfig; error?: string }> {
    const entry = this.models.get(modelName);
    if (!entry) {
      return { status: "error", error: `Unknown model: ${modelName}` };
    }
    // Stop and start are one transition: a lazy start slipping between them
    // would spawn a second backend onto the port this one is about to take.
    return this.withLifecycle(entry, () => this.restartModelLocked(entry, argOverrides));
  }

  private async restartModelLocked(
    entry: ModelEntry,
    argOverrides?: Record<string, string>
  ): Promise<{ status: string; config?: ModelConfig; error?: string }> {
    const modelName = entry.config.name;

    try {
      // Step 1: Reload config (this also stops the process)
      const newConfig = await this.reloadModelConfigLocked(entry);
      if (!newConfig) {
        return { status: "error", error: `Model ${modelName} not found in config file` };
      }

      // Step 2: Apply runtime arg overrides if provided
      if (argOverrides && Object.keys(argOverrides).length > 0) {
        const args = [...(newConfig.extraArgs ?? [])];
        for (const [flag, value] of Object.entries(argOverrides)) {
          // Find the flag in extraArgs and replace the value that follows it
          const idx = args.indexOf(flag);
          if (idx !== -1 && idx + 1 < args.length) {
            const oldValue = args[idx + 1];
            args[idx + 1] = value;
            console.log(`[restart] Override: ${flag} ${oldValue} → ${value}`);
          } else {
            // Flag not found — append it
            args.push(flag, value);
            console.log(`[restart] Override: adding ${flag} ${value}`);
          }
        }
        newConfig.extraArgs = args;
        entry.config = newConfig;
        entry.process.updateConfig(newConfig);
      }

      // Step 3: Start with new config
      console.log(`[restart] Starting ${modelName} with updated config...`);
      await this.ensureRunningLocked(entry);

      return { status: "running", config: newConfig };
    } catch (err) {
      return { status: "error", error: String(err) };
    }
  }

  /** Stop all models and the server */
  async stop() {
    if (this.idleSweepTimer) {
      clearTimeout(this.idleSweepTimer);
      this.idleSweepTimer = null;
    }
    if (this.healthTimer) {
      clearTimeout(this.healthTimer);
      this.healthTimer = null;
    }

    for (const entry of this.getManagedEntries()) {
      await this.stopEntry(entry);
    }

    this.server?.stop();
  }

  /** Best-effort synchronous kill of every backend, for process exit. */
  killAllNow(): void {
    for (const entry of this.getManagedEntries()) {
      entry.process.killNow();
    }
  }

}
