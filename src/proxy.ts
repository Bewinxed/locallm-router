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
import { getInternalPort, buildModelLookup, reloadModelConfig as reloadModelConfigFromDisk } from "./config";
import { createProcess, type BackendProcess } from "./process";
import { downloadModel, isLocalModel } from "./download";
import type { GpuLock } from "./gpu-lock";

// ─── Types ──────────────────────────────────────────────────────

export interface ModelEntry {
  config: ModelConfig;
  process: BackendProcess;
  internalPort: number;
  downloading: boolean;
  downloadError: string | null;
}

/** GpuOwner adapter so GpuLock can manage ModelEntry */
export interface GpuOwner {
  readonly name: string;
  readonly gpu: number;
  readonly gpus?: number[];
  readonly config: ModelConfig;
  isRunning(): boolean;
  stopProcess(): Promise<void>;
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
  private lookup: Map<string, ModelConfig>;
  private gpuLock: GpuLock;
  private server: ReturnType<typeof Bun.serve> | null = null;
  private idleTimers: Map<string, Timer> = new Map();
  private idleSweepTimer: Timer | null = null;

  /** Balance groups: base model name -> BalancedModelGroup */
  private balanceGroups: Map<string, BalancedModelGroup> = new Map();
  /** Set of base model names that have balance mode enabled */
  private balancedModels: Set<string> = new Set();

  constructor(configs: ModelConfig[], gpuLock: GpuLock) {
    this.lookup = buildModelLookup(configs);
    this.gpuLock = gpuLock;

    // Create entries for each model
    for (let i = 0; i < configs.length; i++) {
      const config = configs[i];
      const internalPort = getInternalPort(i);
      const proc = createProcess(config, internalPort);
      const entry: ModelEntry = {
        config,
        process: proc,
        internalPort,
        downloading: false,
        downloadError: null,
      };
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
          const balanceProc = createProcess(gpuConfig, balancePort);
          const balanceEntry: ModelEntry = {
            config: gpuConfig,
            process: balanceProc,
            internalPort: balancePort,
            downloading: false,
            downloadError: null,
          };

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

  private async stopEntry(entry: ModelEntry): Promise<void> {
    this.clearIdleTimer(entry.config.name);
    if (!isEntryRunning(entry)) return;
    await entry.process.stop();
    this.gpuLock.notifyStopped(makeGpuOwner(entry));
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
      fetch: (req) => this.handleRequest(req),
    });
    console.log(`Unified proxy listening on port ${port}`);
    return this;
  }

  /** Get a model entry, resolving aliases (no modifier parsing) */
  private getEntry(modelName: string): ModelEntry | undefined {
    return this.models.get(modelName);
  }

  /** Check if a base model name has balance mode enabled */
  private isBalanced(baseName: string): boolean {
    return this.balancedModels.has(baseName);
  }

  /** Extract model name from request (body or URL) */
  private async extractModelName(req: Request, body?: any): Promise<string | null> {
    if (body?.model) {
      return body.model;
    }
    return null;
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

    if (url.pathname === "/manager/status") {
      return Response.json({
        models: this.getModelsStatus(),
        gpus: await this.gpuLock.status(),
        balanceGroups: this.getBalanceGroupsStatus(),
      });
    }

    if (url.pathname === "/manager/stop-all" && req.method === "POST") {
      for (const entry of this.getManagedEntries()) {
        await this.stopEntry(entry);
      }
      return Response.json({ status: "all stopped" });
    }

    // ─── Model-specific management ──────────────────────────
    const mgmtMatch = url.pathname.match(/^\/manager\/(\w+)\/(.+)$/);
    if (mgmtMatch) {
      const [, action, modelName] = mgmtMatch;
      const entry = this.getEntry(modelName);
      if (!entry) return Response.json({ error: `Unknown model: ${modelName}` }, { status: 404 });

      if (action === "start" && req.method === "POST") {
        try {
          await this.ensureRunning(entry);
          return Response.json({ status: "running" });
        } catch (err) {
          return Response.json({ error: String(err) }, { status: 500 });
        }
      }
      if (action === "stop" && req.method === "POST") {
        await this.stopEntry(entry);
        await this.stopBalanceShadows(entry.config.name, entry);
        return Response.json({ status: "stopped" });
      }
      if (action === "status") {
        return Response.json(entry.process.getState());
      }
    }

    // ─── API requests — route by model field ────────────────

    // Parse body for model field (only for methods with body)
    let body: any = null;
    let rawBody: ArrayBuffer | null = null;
    if (req.method !== "GET" && req.method !== "HEAD") {
      rawBody = await req.arrayBuffer();
      try {
        body = JSON.parse(new TextDecoder().decode(rawBody));
      } catch {
        // Not JSON — pass through
      }
    }

    const modelName = await this.extractModelName(req, body);
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
      return this.handleBalancedRequest(req, body, rawBody, parsed, group, baseEntry);
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
      return this.handleBalancedRequest(req, body, rawBody, implicitParsed, group, baseEntry);
    }

    try {
      await this.ensureRunning(baseEntry);
    } catch (err) {
      return Response.json(
        { error: "Model failed to start", details: String(err) },
        { status: 503 }
      );
    }

    // Rewrite model name in body to match what the backend expects
    if (body && body.model !== baseEntry.config.name) {
      body.model = baseEntry.config.name;
      rawBody = new TextEncoder().encode(JSON.stringify(body)).buffer as ArrayBuffer;
    }

    return this.proxyRequest(req, baseEntry, rawBody);
  }

  /**
   * Handle a request with a balance modifier (:balance, :gpuN).
   * Selects the right instance, ensures it's running, tracks inflight, proxies.
   */
  private async handleBalancedRequest(
    req: Request,
    body: any,
    rawBody: ArrayBuffer | null,
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
        return Response.json(
          { error: "Model failed to start", details: String(err) },
          { status: 503 }
        );
      }
    }

    // Rewrite model name to the base name (strip modifier) so the backend recognizes it.
    // The backend's -a (alias) flag is set to the base model name for the primary,
    // and the internal name for secondary instances. We need to match what llama-server expects.
    if (body) {
      // llama-server uses -a flag to set served model name.
      // Primary instance: served as config.name (e.g. "qwen3.5-35b-a3b")
      // Secondary instances: served as internal name (e.g. "qwen3.5-35b-a3b__gpu0")
      // We rewrite to whatever the target's config.name is.
      body.model = targetEntry.config.name;
      rawBody = new TextEncoder().encode(JSON.stringify(body)).buffer as ArrayBuffer;
    }

    // Track inflight and proxy
    group.startRequest(targetGpu);
    try {
      const response = await this.proxyRequest(req, targetEntry, rawBody);
      const contentType = response.headers.get("content-type") || "";

      // For streaming responses, wrap the body to decrement inflight when stream ends
      if (
        response.body &&
        (contentType.includes("text/event-stream") || contentType.includes("stream"))
      ) {
        const originalBody = response.body;
        const trackingStream = new TransformStream({
          flush: () => {
            group.endRequest(targetGpu);
          },
        });
        const pipedBody = originalBody.pipeThrough(trackingStream);
        return new Response(pipedBody, {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }

      // Non-streaming: already fully consumed, decrement now
      group.endRequest(targetGpu);
      return response;
    } catch (err) {
      group.endRequest(targetGpu);
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
  private async ensureRunning(entry: ModelEntry): Promise<void> {
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
          await downloadModel(entry.config.modelPath, (line) => {
            console.log(`[${entry.config.name}] ${line}`);
          });
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
      await this.gpuLock.acquire(owner);
      const gpuStr = entry.config.gpus
        ? `GPUs ${entry.config.gpus.join(",")}`
        : `GPU ${entry.config.gpu}`;
      console.log(`[${entry.config.name}] Cold start — launching ${entry.config.backend || "vllm"} on ${gpuStr}...`);
      await entry.process.start();
    }

    entry.process.touchActivity();
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

    const headers = new Headers(req.headers);
    headers.delete("host");

    const proxyReq: RequestInit = {
      method: req.method,
      headers,
    };

    if (rawBody) {
      proxyReq.body = rawBody;
    }

    const response = await fetch(targetUrl, proxyReq);
    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("text/event-stream") || contentType.includes("stream")) {
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    return new Response(await response.arrayBuffer(), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
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
        downloading: entry.downloading,
        aliases: config.aliases || [],
        balance: group ? group.getStatus() : null,
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

  // ─── Idle timer management ────────────────────────────────

  private resetIdleTimer(entry: ModelEntry) {
    const timerKey = entry.config.name;
    this.clearIdleTimer(timerKey);

    const timer = setTimeout(async () => {
      try {
        const idleTime = Date.now() - entry.process.lastActivity;
        if (idleTime >= entry.config.idleTimeout && entry.process.status === "running") {
          console.log(
            `[${entry.config.name}] Idle for ${Math.round(idleTime / 1000 / 60)}min, stopping...`
          );
          await this.stopEntry(entry);
          await this.stopBalanceShadows(entry.config.name, entry);
        }
      } catch (err) {
        console.error(`[${entry.config.name}] Error during idle unload:`, err);
        // Force kill if stop() failed
        try {
          await entry.process.stop();
        } catch { /* already logged */ }
        try {
          this.gpuLock.notifyStopped(makeGpuOwner(entry));
        } catch { /* best effort */ }
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

  /** Periodic sweep to catch any models that missed their idle timer */
  startIdleSweep(intervalMs = 60_000) {
    if (this.idleSweepTimer) clearInterval(this.idleSweepTimer);
    this.idleSweepTimer = setInterval(async () => {
      for (const entry of this.getEntries()) {
        if (entry.process.status !== "running") continue;
        const idleTime = Date.now() - entry.process.lastActivity;
        if (idleTime >= entry.config.idleTimeout) {
          console.log(
            `[idle-sweep] ${entry.config.name} idle for ${Math.round(idleTime / 1000 / 60)}min, stopping...`
          );
          try {
            await this.stopEntry(entry);
            await this.stopBalanceShadows(entry.config.name, entry);
          } catch (err) {
            console.error(`[idle-sweep] Error stopping ${entry.config.name}:`, err);
          }
        }
      }
    }, intervalMs);
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

    // Re-read from disk
    const newConfig = reloadModelConfigFromDisk(modelName);
    if (!newConfig) {
      console.warn(`[reload] Model ${modelName} not found in config file`);
      return null;
    }

    // Stop if running
    if (entry.process.status === "running" || entry.process.status === "starting") {
      console.log(`[reload] Stopping ${modelName} before config update...`);
      await entry.process.stop();
      this.clearIdleTimer(modelName);
      this.gpuLock.notifyStopped(makeGpuOwner(entry));
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
        if (balanceEntry.process.status === "running" || balanceEntry.process.status === "starting") {
          await balanceEntry.process.stop();
          this.clearIdleTimer(gpuConfig.name);
          this.gpuLock.notifyStopped(makeGpuOwner(balanceEntry));
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

    try {
      // Step 1: Reload config (this also stops the process)
      const newConfig = await this.reloadModelConfig(modelName);
      if (!newConfig) {
        return { status: "error", error: `Model ${modelName} not found in config file` };
      }

      // Step 2: Apply runtime arg overrides if provided
      if (argOverrides && Object.keys(argOverrides).length > 0 && newConfig.extraArgs) {
        const args = [...newConfig.extraArgs];
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
      await this.ensureRunning(entry);

      return { status: "running", config: newConfig };
    } catch (err) {
      return { status: "error", error: String(err) };
    }
  }

  /** Stop all models and the server */
  async stop() {
    if (this.idleSweepTimer) {
      clearInterval(this.idleSweepTimer);
      this.idleSweepTimer = null;
    }

    for (const entry of this.getManagedEntries()) {
      await this.stopEntry(entry);
    }

    this.server?.stop();
  }

  /** Get entries for the management API */
  getEntries(): ModelEntry[] {
    const seen = new Set<string>();
    const entries: ModelEntry[] = [];
    for (const [, entry] of this.models) {
      if (!seen.has(entry.config.name)) {
        seen.add(entry.config.name);
        entries.push(entry);
      }
    }
    return entries;
  }

  /** Get balance groups for external inspection */
  getBalanceGroups(): Map<string, BalancedModelGroup> {
    return this.balanceGroups;
  }
}
