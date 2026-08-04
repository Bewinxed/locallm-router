/**
 * GPU Lock — VRAM-aware scheduler for model coexistence.
 *
 * Models that share a GPU can coexist if their combined VRAM fits.
 * If a new model doesn't fit, the scheduler evicts the least-recently-used
 * model(s) until there's enough room.
 */

import { readFileSync, writeFileSync } from "fs";
import {
  estimateVRAM,
  fetchRepoWeightsGB,
  getComputeApps,
  getGpuInfo,
  formatEstimate,
  getGpuMemoryUtil,
  resolveEstimate,
  type ComputeApp,
  type ModelSizeEstimate,
} from "./model-size";
import { MODELS_DIR } from "./download";
import type { ModelConfig } from "./config";

export interface GpuOwner {
  readonly name: string;
  readonly gpu: number;
  readonly gpus?: number[];
  readonly config: ModelConfig;
  /**
   * True while the backend process exists — starting, serving, asleep or on
   * its way out. All of those hold VRAM, so all of them keep their
   * reservation; only a process that is actually gone has a stale one.
   */
  isAlive(): boolean;
  stopProcess(): Promise<void>;
  /** Pids of the backend's process group; empty when it isn't running. */
  pids(): number[];
}

/** Everything the scheduler reads from the driver about one card, in one shot. */
export interface GpuReading {
  name: string;
  totalMB: number;
  usedMB: number;
  /** Per-process rows for this card only. */
  computeApps: ComputeApp[];
}

/** Null when the driver doesn't report the card at all. */
export type GpuProbe = (gpu: number) => Promise<GpuReading | null>;

const nvidiaProbe: GpuProbe = async (gpu) => {
  const [gpus, apps] = await Promise.all([getGpuInfo(), getComputeApps()]);
  const info = gpus.find((g) => g.index === gpu);
  if (!info) return null;
  return {
    name: info.name,
    totalMB: info.totalMemoryMB,
    usedMB: info.usedMemoryMB,
    computeApps: apps.filter((a) => a.gpu === gpu),
  };
};

interface LoadedModel {
  owner: GpuOwner;
  estimate: ModelSizeEstimate;
  observedVRAM: number | null;
  loadedAt: number;
  lastActivity: number;
  /** Asleep but still resident: the backend lives, its weights don't. */
  sleeping: boolean;
  /** What it still holds here while asleep; null when we couldn't measure. */
  sleepingVRAM: number | null;
}

/** Undoes one acquisition, for when a sibling card refuses the same model. */
type Rollback = () => void;

interface EvictionRecord {
  modelName: string;
  evictedAt: number;
}

/** Left free on every card, so the driver and the desktop aren't squeezed out. */
const VRAM_HEADROOM_GB = 1.0;
const ANTI_THRASH_COOLDOWN_MS = 30_000;
const VRAM_SNAPSHOT_SETTLE_MS = 3_000;

/** Thrown when a model cannot be made to fit, even by evicting everything else. */
export class InsufficientVRAMError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InsufficientVRAMError";
  }
}

/** Thrown when a just-evicted model is asked to reload into contended VRAM. */
export class EvictionCooldownError extends Error {
  readonly retryAfterSeconds: number;

  constructor(modelName: string, retryAfterSeconds: number) {
    super(
      `${modelName} was evicted less than ${ANTI_THRASH_COOLDOWN_MS / 1000}s ago and the GPU is ` +
        `still contended — retry in ${retryAfterSeconds}s`
    );
    this.name = "EvictionCooldownError";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

// ─── Observed VRAM persistence ──────────────────────────────────

/**
 * Measured VRAM per model, kept across restarts. Real numbers beat estimates,
 * and a router restart shouldn't throw away what we already learned.
 */
const OBSERVED_VRAM_PATH = `${MODELS_DIR}/.locallm-router-observed.json`;

interface ObservedVRAM {
  gb: number;
  /** True when the number came from per-process accounting rather than a delta. */
  perProcess: boolean;
}

/** Older files hold a bare number; those were all whole-GPU deltas. */
function parseObserved(value: unknown): ObservedVRAM | null {
  if (typeof value === "number" && value > 0) return { gb: value, perProcess: false };
  if (value && typeof value === "object") {
    const { gb, perProcess } = value as Partial<ObservedVRAM>;
    if (typeof gb === "number" && gb > 0) return { gb, perProcess: perProcess === true };
  }
  return null;
}

function loadObservedVRAM(): Map<string, ObservedVRAM> {
  try {
    const raw = JSON.parse(readFileSync(OBSERVED_VRAM_PATH, "utf-8")) as Record<string, unknown>;
    const entries: [string, ObservedVRAM][] = [];
    for (const [name, value] of Object.entries(raw)) {
      const observed = parseObserved(value);
      if (observed) entries.push([name, observed]);
    }
    if (entries.length > 0) {
      console.log(`[gpu-lock] Loaded observed VRAM for ${entries.length} model(s) from ${OBSERVED_VRAM_PATH}`);
    }
    return new Map(entries);
  } catch {
    return new Map();
  }
}

const observedVRAM = loadObservedVRAM();

/**
 * Record a measurement, unless it is a whole-GPU delta about to overwrite a
 * per-process number — the delta is the one that picks up a predecessor's
 * memory on its way out, so it doesn't get to win.
 */
function persistObservedVRAM(modelName: string, gb: number, perProcess: boolean): boolean {
  const prior = observedVRAM.get(modelName);
  if (!perProcess && prior?.perProcess) return false;

  observedVRAM.set(modelName, { gb: Math.round(gb * 100) / 100, perProcess });
  try {
    writeFileSync(OBSERVED_VRAM_PATH, JSON.stringify(Object.fromEntries(observedVRAM), null, 2));
  } catch (err) {
    console.warn(`[gpu-lock] Could not persist observed VRAM to ${OBSERVED_VRAM_PATH}:`, err);
  }
  return true;
}

export class GpuScheduler {
  readonly gpu: number;
  private readonly probe: GpuProbe;
  private totalVRAM: number = 0;
  private initialized = false;
  private initPromise: Promise<void> | null = null;
  private loaded = new Map<string, LoadedModel>();
  /** Every model configured for this card, resident or not — see otherUsage. */
  private owners = new Map<string, GpuOwner>();
  private estimates = new Map<string, ModelSizeEstimate>();
  private estimateRefresh = new Map<string, Promise<void>>();
  private recentEvictions: EvictionRecord[] = [];
  private lastReconcileTime = 0;
  /** VRAM held on this card by anything the router didn't start. */
  private otherUsageGB = 0;
  private otherUsageAt = 0;
  /** Serializes acquire() so two cold starts can't both claim the same VRAM. */
  private queue: Promise<unknown> = Promise.resolve();
  private static readonly RECONCILE_INTERVAL_MS = 5000;
  private static readonly OTHER_USAGE_TTL_MS = 5000;

  constructor(gpu: number, probe: GpuProbe = nvidiaProbe) {
    this.gpu = gpu;
    this.probe = probe;
  }

  initialize(): Promise<void> {
    if (!this.initPromise) this.initPromise = this.readTotalVRAM();
    return this.initPromise;
  }

  private async readTotalVRAM(): Promise<void> {
    const reading = await this.probe(this.gpu);
    if (reading) {
      this.totalVRAM = reading.totalMB / 1024;
      console.log(`[gpu-lock] GPU ${this.gpu}: ${reading.name}, ${this.totalVRAM.toFixed(1)}GB total`);
    } else {
      this.totalVRAM = 24;
      console.log(`[gpu-lock] GPU ${this.gpu}: Unknown (assuming 24GB)`);
    }
    this.initialized = true;
  }

  get isInitialized(): boolean {
    return this.initialized;
  }

  registerModel(owner: GpuOwner): void {
    this.owners.set(owner.name, owner);
    this.setEstimate(owner, estimateVRAM(owner.config));
    void this.ensureEstimate(owner);
    void this.warnIfWeightsExceedPrealloc(owner);

    const observed = observedVRAM.get(owner.name);
    if (observed !== undefined) {
      console.log(
        `[gpu-lock] GPU ${this.gpu}: ${owner.name} previously observed at ${observed.gb.toFixed(1)}GB` +
          (observed.perProcess ? "" : " (whole-GPU delta)")
      );
    }
  }

  /** A model's estimate is for one card; a tensor-parallel one splits across them. */
  private setEstimate(owner: GpuOwner, estimate: ModelSizeEstimate): void {
    const numGpus = owner.gpus ? owner.gpus.length : 1;
    const perGpuEstimate: ModelSizeEstimate =
      numGpus > 1
        ? {
            ...estimate,
            weightsGB: estimate.weightsGB / numGpus,
            kvCacheGB: estimate.kvCacheGB / numGpus,
            totalVRAM: estimate.totalVRAM / numGpus,
          }
        : estimate;
    this.estimates.set(owner.name, perGpuEstimate);
  }

  private logEstimate(owner: GpuOwner, estimate: ModelSizeEstimate): void {
    const numGpus = owner.gpus ? owner.gpus.length : 1;
    if (numGpus === 1) {
      console.log(`[gpu-lock] GPU ${this.gpu}: ${owner.name} -> ${formatEstimate(estimate)}`);
      return;
    }
    const perGpuGB = this.estimates.get(owner.name)?.totalVRAM ?? estimate.totalVRAM / numGpus;
    console.log(
      `[gpu-lock] GPU ${this.gpu}: ${owner.name} -> ${formatEstimate(estimate)} (${estimate.totalVRAM.toFixed(1)}GB total / ${numGpus} GPUs = ${perGpuGB.toFixed(1)}GB here)`
    );
  }

  /**
   * Replace the boot-time guess with what the model file actually says. The
   * header read is remote when the model isn't downloaded yet, so this is also
   * how a model gets a size before it costs us a download.
   */
  private ensureEstimate(owner: GpuOwner): Promise<void> {
    let pending = this.estimateRefresh.get(owner.name);
    if (!pending) {
      pending = resolveEstimate(owner.config)
        .then((estimate) => {
          this.setEstimate(owner, estimate);
          this.logEstimate(owner, estimate);
        })
        .catch((err) => {
          console.warn(`[gpu-lock] GPU ${this.gpu}: could not size ${owner.name}:`, err);
        })
        .finally(() => {
          // Anything short of a parsed header is worth retrying: the file may
          // be on disk, or the network back, by the next admission decision.
          if (!this.estimates.get(owner.name)?.confidence.startsWith("gguf")) {
            this.estimateRefresh.delete(owner.name);
          }
        });
      this.estimateRefresh.set(owner.name, pending);
    }
    return pending;
  }

  /**
   * vLLM is admitted on its prealloc budget rather than an estimate, but if its
   * weights alone don't fit in that budget it will OOM part-way through a
   * multi-minute load. Better to say so at startup than to find out then.
   */
  private async warnIfWeightsExceedPrealloc(owner: GpuOwner): Promise<void> {
    await this.initialize();
    const floor = this.preallocFloor(owner.config);
    if (floor <= 0) return;

    const weightsGB = await fetchRepoWeightsGB(owner.config);
    if (weightsGB === null) return;

    const share = weightsGB / (owner.gpus?.length ?? 1);
    if (share > floor) {
      console.warn(
        `[gpu-lock] GPU ${this.gpu}: ${owner.name} weights are ${share.toFixed(1)}GB per GPU but ` +
          `--gpu-memory-utilization only budgets ${floor.toFixed(1)}GB — it will not load`
      );
    }
  }

  /**
   * vLLM preallocates totalVRAM × its own --gpu-memory-utilization on every
   * card it runs on, so that is a floor on what it will take here. The flag
   * caps that model's own appetite; it says nothing about the rest of the card.
   */
  private preallocFloor(config: ModelConfig): number {
    if ((config.backend ?? "vllm") !== "vllm") return 0;
    return this.totalVRAM * getGpuMemoryUtil(config.extraArgs);
  }

  /**
   * Per-card footprint. A measurement beats a prediction, but never falls below
   * what vLLM is going to preallocate regardless: a snapshot taken while the
   * previous tenant was still handing memory back reads low, and admitting on
   * that number oversubscribes the card.
   */
  private footprint(config: ModelConfig, estimateGB: number, observedGB: number | null): number {
    return Math.max(observedGB ?? estimateGB, this.preallocFloor(config));
  }

  /**
   * A sleeping model is held to what it actually still occupies, not to its
   * running footprint or vLLM's prealloc floor — the weights are in host RAM
   * and that memory is genuinely available to everyone else.
   */
  private getModelVRAM(model: LoadedModel): number {
    if (model.sleeping && model.sleepingVRAM !== null) return model.sleepingVRAM;
    return this.footprint(model.owner.config, model.estimate.totalVRAM, model.observedVRAM);
  }

  /** What this model is expected to take here — measured if we've seen it run. */
  private expectedVRAM(owner: GpuOwner): number {
    const estimate = this.estimates.get(owner.name);
    if (!estimate) throw new Error(`Model ${owner.name} not registered`);
    return this.footprint(
      owner.config,
      estimate.totalVRAM,
      observedVRAM.get(owner.name)?.gb ?? null
    );
  }

  private getEstimatedVRAM(): number {
    let total = 0;
    for (const model of this.loaded.values()) total += this.getModelVRAM(model);
    return total;
  }

  async reconcileWithGpu(): Promise<void> {
    const now = Date.now();
    if (now - this.lastReconcileTime < GpuScheduler.RECONCILE_INTERVAL_MS) return;
    this.lastReconcileTime = now;

    const reading = await this.probe(this.gpu);
    if (!reading) return;

    const actualUsedGB = reading.usedMB / 1024;
    const estimatedUsedGB = this.getEstimatedVRAM();

    if (estimatedUsedGB > 2 && actualUsedGB < estimatedUsedGB - 2) {
      let removedAny = false;
      for (const [name, model] of this.loaded) {
        if (!model.owner.isAlive()) {
          console.log(`[gpu-lock] GPU ${this.gpu}: removing stale entry ${name} (est=${estimatedUsedGB.toFixed(1)}GB, actual=${actualUsedGB.toFixed(1)}GB)`);
          this.loaded.delete(name);
          removedAny = true;
        }
      }
      if (!removedAny && this.loaded.size > 0) {
        // VRAM mismatch but all entries claim to be running -- only log once
        console.log(`[gpu-lock] GPU ${this.gpu}: VRAM mismatch (est=${estimatedUsedGB.toFixed(1)}GB, actual=${actualUsedGB.toFixed(1)}GB) but all ${this.loaded.size} entries are running`);
        this.lastReconcileTime = now + 55_000; // suppress for 60s total
      }
    }
  }

  /**
   * VRAM held on this card by processes the router didn't start — a desktop
   * session is several GB on GPU 0 here. Accounting alone can't see it, so
   * without this the scheduler hands the same memory out twice.
   *
   * What is ours is decided by asking the driver which pids hold what, not by
   * our own bookkeeping: a backend the scheduler has stopped tracking is still
   * the router's memory, and taxing ourselves for it is how a sleeping model
   * ends up unable to afford its own wake.
   */
  private async refreshOtherUsage(): Promise<void> {
    const now = Date.now();
    if (now - this.otherUsageAt < GpuScheduler.OTHER_USAGE_TTL_MS) return;
    this.otherUsageAt = now;

    const reading = await this.probe(this.gpu);
    if (!reading) return;
    const oursGB = this.routerOwnedVRAM(reading.computeApps) ?? this.getEstimatedVRAM();
    this.otherUsageGB = Math.max(0, reading.usedMB / 1024 - oursGB);
  }

  /**
   * Per-process VRAM held here by backends the router started, awake or
   * asleep. Null when the driver accounts for none of them: a Vulkan build of
   * llama.cpp holds real memory and never appears in these rows, and reading
   * that as zero would declare the whole card foreign.
   */
  private routerOwnedVRAM(apps: ComputeApp[]): number | null {
    const pids = new Set<number>();
    for (const owner of this.owners.values()) {
      for (const pid of owner.pids()) pids.add(pid);
    }
    if (pids.size === 0) return null;

    const rows = apps.filter((a) => pids.has(a.pid));
    if (rows.length === 0) return null;
    return rows.reduce((sum, a) => sum + a.usedMemoryMB, 0) / 1024;
  }

  /** Room left on the card for a model that isn't resident yet. */
  private getAvailableVRAM(): number {
    return this.totalVRAM - VRAM_HEADROOM_GB - this.getEstimatedVRAM() - this.otherUsageGB;
  }

  /** Snapshot current GPU memory usage (call before model start) */
  async snapshotVRAM(): Promise<number> {
    const reading = await this.probe(this.gpu);
    return reading ? reading.usedMB / 1024 : 0;
  }

  /**
   * What this model's own processes hold on this card right now, per the
   * driver. Null when it has no pids here, or none the driver accounts for.
   */
  private async measureResident(owner: GpuOwner): Promise<number | null> {
    const pids = new Set(owner.pids());
    if (pids.size === 0) return null;

    const reading = await this.probe(this.gpu);
    if (!reading) return null;
    const rows = reading.computeApps.filter((a) => pids.has(a.pid));
    if (rows.length === 0) return null;
    return rows.reduce((sum, a) => sum + a.usedMemoryMB, 0) / 1024;
  }

  /**
   * What this model holds on this card. Per-process accounting is the honest
   * answer — a whole-GPU delta also counts whatever the model it replaced had
   * not finished handing back, which is how an 18GB model once measured
   * 11.6GB. But the driver only accounts for CUDA contexts: a Vulkan backend
   * holds real memory and reports no rows at all, so for those we fall back to
   * the delta and say that's what it is.
   */
  private async measureVRAM(
    owner: GpuOwner,
    preSnapshotGB: number
  ): Promise<{ gb: number; perProcess: boolean }> {
    const perProcessGB = await this.measureResident(owner);
    if (perProcessGB !== null) return { gb: perProcessGB, perProcess: true };

    const postGB = await this.snapshotVRAM();
    return { gb: Math.max(0, postGB - preSnapshotGB), perProcess: false };
  }

  /** Record observed VRAM after a model finishes loading */
  async recordObservedVRAM(owner: GpuOwner, preSnapshotGB: number): Promise<void> {
    await new Promise((r) => setTimeout(r, VRAM_SNAPSHOT_SETTLE_MS));
    const { gb, perProcess } = await this.measureVRAM(owner, preSnapshotGB);
    const model = this.loaded.get(owner.name);
    if (!model || gb < 0.5) return;

    if (!persistObservedVRAM(owner.name, gb, perProcess)) {
      console.log(
        `[gpu-lock] GPU ${this.gpu}: ${owner.name} measured ${gb.toFixed(1)}GB by whole-GPU delta — ` +
          `keeping the per-process ${observedVRAM.get(owner.name)!.gb.toFixed(1)}GB`
      );
      return;
    }

    model.observedVRAM = gb;
    const estGB = model.estimate.totalVRAM;
    const drift = Math.abs(gb - estGB);
    if (drift > 1) {
      console.log(
        `[gpu-lock] GPU ${this.gpu}: ${owner.name} observed ${gb.toFixed(1)}GB via ` +
          `${perProcess ? "per-process accounting" : "whole-GPU delta"} ` +
          `(estimated ${estGB.toFixed(1)}GB, drift ${drift.toFixed(1)}GB)`
      );
    }
  }

  /**
   * Adopt a per-card measurement taken on another card of the same model —
   * tensor-parallel residency is recorded on every GPU the model occupies.
   */
  refreshObservedVRAM(owner: GpuOwner): void {
    const model = this.loaded.get(owner.name);
    const observed = observedVRAM.get(owner.name);
    if (model && observed !== undefined) model.observedVRAM = observed.gb;
  }

  /** Milliseconds left on a model's anti-thrash cooldown, 0 when clear. */
  cooldownRemaining(modelName: string): number {
    const now = Date.now();
    this.recentEvictions = this.recentEvictions.filter(
      (e) => now - e.evictedAt < ANTI_THRASH_COOLDOWN_MS
    );
    const record = this.recentEvictions.find((e) => e.modelName === modelName);
    return record ? ANTI_THRASH_COOLDOWN_MS - (now - record.evictedAt) : 0;
  }

  /**
   * VRAM this model still has to find here. A sleeping model is already a
   * resident holding its CUDA context, so waking it only has to cover the
   * difference between that and its running footprint.
   */
  private additionalVRAM(owner: GpuOwner): number {
    const resident = this.loaded.get(owner.name);
    const heldGB = resident ? this.getModelVRAM(resident) : 0;
    return Math.max(0, this.expectedVRAM(owner) - heldGB);
  }

  /** True when this model can only be loaded by evicting something else. */
  needsEviction(owner: GpuOwner): boolean {
    const resident = this.loaded.get(owner.name);
    if (resident && !resident.sleeping) return false;
    return this.additionalVRAM(owner) > this.getAvailableVRAM();
  }

  private recordEviction(modelName: string): void {
    this.recentEvictions.push({ modelName, evictedAt: Date.now() });
  }

  private getEvictionCandidates(neededVRAM: number, excludeName: string): LoadedModel[] {
    const candidates = [...this.loaded.values()]
      .filter((m) => m.owner.name !== excludeName)
      .sort((a, b) => {
        const pa = a.owner.config.priority ?? 0;
        const pb = b.owner.config.priority ?? 0;
        if (pa !== pb) return pa - pb;
        return a.lastActivity - b.lastActivity;
      });
    const toEvict: LoadedModel[] = [];
    let freedVRAM = 0;
    for (const model of candidates) {
      if (freedVRAM >= neededVRAM) break;
      toEvict.push(model);
      freedVRAM += this.getModelVRAM(model);
    }
    return toEvict;
  }

  /** Fail loudly with the numbers, so "it didn't fit" is never a mystery. */
  private cannotFitError(owner: GpuOwner, needVRAM: number): InsufficientVRAMError {
    const residents = [...this.loaded.values()].map(
      (m) => `${m.owner.name} ${this.getModelVRAM(m).toFixed(1)}GB${m.sleeping ? " (sleeping)" : ""}`
    );
    return new InsufficientVRAMError(
      `Cannot fit ${owner.name} (${needVRAM.toFixed(1)}GB) on GPU ${this.gpu}: ` +
        `${this.totalVRAM.toFixed(1)}GB total, ${VRAM_HEADROOM_GB.toFixed(1)}GB headroom, ` +
        `${this.otherUsageGB.toFixed(1)}GB used outside the router, resident: ` +
        (residents.length > 0 ? residents.join(", ") : "none")
    );
  }

  /**
   * Would this model fit, evicting if it came to that? Reserves nothing and
   * evicts nothing — it is the question asked before a download, when the
   * answer decides whether the bytes are worth pulling at all.
   */
  async checkFit(owner: GpuOwner): Promise<void> {
    await this.initialize();
    await this.ensureEstimate(owner);
    await this.refreshOtherUsage();
    const resident = this.loaded.get(owner.name);
    if (resident && !resident.sleeping) return;

    const needVRAM = this.additionalVRAM(owner);
    const available = this.getAvailableVRAM();
    if (needVRAM <= available) return;

    const freeable = this.getEvictionCandidates(needVRAM - available, owner.name).reduce(
      (sum, m) => sum + this.getModelVRAM(m),
      0
    );
    if (needVRAM > available + freeable) throw this.cannotFitError(owner, needVRAM);
  }

  /**
   * Reserve VRAM for a model, evicting others if needed. Serialized: a second
   * caller waits rather than racing the first one's eviction. Resolves with an
   * undo for the reservation, which a multi-GPU acquire needs when a later
   * card turns the same model down.
   */
  acquire(owner: GpuOwner): Promise<Rollback> {
    const run = this.queue.then(
      () => this.acquireLocked(owner),
      () => this.acquireLocked(owner)
    );
    this.queue = run.catch(() => {});
    return run;
  }

  private async acquireLocked(owner: GpuOwner): Promise<Rollback> {
    await this.initialize();
    await this.reconcileWithGpu();

    const resident = this.loaded.get(owner.name);
    if (resident && !resident.sleeping) {
      resident.lastActivity = Date.now();
      return () => {};
    }

    await this.ensureEstimate(owner);
    await this.refreshOtherUsage();

    const estimate = this.estimates.get(owner.name);
    if (!estimate) throw new Error(`Model ${owner.name} not registered`);
    const needVRAM = this.additionalVRAM(owner);
    const available = this.getAvailableVRAM();

    // Coexistence first: only what the resident models actually hold is off
    // limits, and eviction is the fallback when that genuinely isn't enough.
    const toEvict =
      needVRAM <= available ? [] : this.getEvictionCandidates(needVRAM - available, owner.name);
    const freeable = toEvict.reduce((sum, m) => sum + this.getModelVRAM(m), 0);
    if (needVRAM > available + freeable) {
      throw this.cannotFitError(owner, needVRAM);
    }

    // Reserve before evicting: the slot is ours for the whole (slow) eviction,
    // and the victims can't pick it back up on their way out. Waking is a
    // reservation too — the model goes back to its full footprint here for the
    // whole wake, not once it finishes.
    const rollback = this.reserve(owner, estimate, resident);

    if (toEvict.length === 0) {
      console.log(
        `[gpu-lock] GPU ${this.gpu}: ${owner.name} ${resident ? "wakes" : "fits"} ` +
          `(need ${needVRAM.toFixed(1)}GB, have ${available.toFixed(1)}GB)`
      );
      return rollback;
    }

    console.log(
      `[gpu-lock] GPU ${this.gpu}: evicting ${toEvict.map((m) => m.owner.name).join(", ")} for ${owner.name}`
    );

    for (const victim of toEvict) {
      try {
        await victim.owner.stopProcess();
        this.recordEviction(victim.owner.name);
        this.loaded.delete(victim.owner.name);
      } catch (err) {
        console.error(`[gpu-lock] GPU ${this.gpu}: failed to evict ${victim.owner.name}:`, err);
      }
    }

    // An eviction that failed leaves the incoming model without the room it was
    // promised; starting it anyway just OOMs the card.
    if (this.getAvailableVRAM() < 0) {
      rollback();
      throw this.cannotFitError(owner, needVRAM);
    }

    return rollback;
  }

  /**
   * Take the model's full footprint here. `asleep` is its existing residency
   * when this is a wake rather than a cold start, and the rollback puts that
   * residency back — its processes are alive either way.
   */
  private reserve(
    owner: GpuOwner,
    estimate: ModelSizeEstimate,
    asleep?: LoadedModel
  ): Rollback {
    if (asleep) {
      const sleepingVRAM = asleep.sleepingVRAM;
      asleep.sleeping = false;
      asleep.sleepingVRAM = null;
      asleep.lastActivity = Date.now();
      return () => {
        asleep.sleeping = true;
        asleep.sleepingVRAM = sleepingVRAM;
      };
    }

    this.loaded.set(owner.name, {
      owner,
      estimate,
      observedVRAM: observedVRAM.get(owner.name)?.gb ?? null,
      loadedAt: Date.now(),
      lastActivity: Date.now(),
      sleeping: false,
      sleepingVRAM: null,
    });
    return () => this.loaded.delete(owner.name);
  }

  touchActivity(owner: GpuOwner): void {
    const model = this.loaded.get(owner.name);
    if (model) model.lastActivity = Date.now();
  }

  notifyStopped(owner: GpuOwner): void {
    this.loaded.delete(owner.name);
  }

  /**
   * A model that went to sleep stays resident. vLLM's sleep level 1 offloads
   * the weights to host RAM but keeps the process — and its CUDA context —
   * alive, so the card is still occupied; deregistering it here is what left
   * that memory looking like somebody else's.
   */
  async notifySleeping(owner: GpuOwner): Promise<void> {
    const model = this.loaded.get(owner.name);
    if (!model) return;

    const heldGB = await this.measureResident(owner);
    model.sleeping = true;
    model.sleepingVRAM = heldGB;
    // The card just changed underneath us; the cached split is stale.
    this.otherUsageAt = 0;

    console.log(
      `[gpu-lock] GPU ${this.gpu}: ${owner.name} asleep, still holding ` +
        (heldGB === null
          ? `an unmeasured share — keeping its ${this.getModelVRAM(model).toFixed(1)}GB reservation`
          : `${heldGB.toFixed(1)}GB here`)
    );
  }

  async getStatus() {
    await this.reconcileWithGpu();
    await this.refreshOtherUsage();
    return {
      totalVRAM: Math.round(this.totalVRAM * 100) / 100,
      usedVRAM: Math.round(this.getEstimatedVRAM() * 100) / 100,
      otherVRAM: Math.round(this.otherUsageGB * 100) / 100,
      availableVRAM: Math.round(this.getAvailableVRAM() * 100) / 100,
      loaded: [...this.loaded.entries()].map(([name, m]) => ({
        name,
        estimatedVRAM: m.estimate.totalVRAM,
        observedVRAM: m.observedVRAM,
        effectiveVRAM: this.getModelVRAM(m),
        priority: m.owner.config.priority ?? 0,
        lastActivity: m.lastActivity,
        sleeping: m.sleeping,
      })),
    };
  }
}

export class GpuLock {
  private schedulers = new Map<number, GpuScheduler>();
  private readonly probe: GpuProbe;

  constructor(probe: GpuProbe = nvidiaProbe) {
    this.probe = probe;
  }

  /** Number of GPUs the driver reported at boot. */
  get gpuCount(): number {
    return this.schedulers.size;
  }

  async initialize(): Promise<void> {
    const gpus = await getGpuInfo();
    console.log(`[gpu-lock] Detected ${gpus.length} GPU(s)`);
    for (const gpu of gpus) {
      if (this.schedulers.has(gpu.index)) continue;
      const scheduler = new GpuScheduler(gpu.index, this.probe);
      await scheduler.initialize();
      this.schedulers.set(gpu.index, scheduler);
    }
  }

  private gpuIndices(owner: GpuOwner): number[] {
    return owner.gpus && owner.gpus.length > 0 ? owner.gpus : [owner.gpu];
  }

  /**
   * A scheduler for a GPU the driver never reported still gets created — the
   * model is configured for it — but it only learns its size on first acquire.
   */
  private schedulerFor(gpuIdx: number): GpuScheduler {
    let scheduler = this.schedulers.get(gpuIdx);
    if (!scheduler) {
      scheduler = new GpuScheduler(gpuIdx, this.probe);
      this.schedulers.set(gpuIdx, scheduler);
    }
    return scheduler;
  }

  register(owner: GpuOwner): void {
    for (const gpuIdx of this.gpuIndices(owner)) {
      this.schedulerFor(gpuIdx).registerModel(owner);
    }
  }

  async acquire(owner: GpuOwner): Promise<void> {
    const rollbacks: Rollback[] = [];
    for (const gpuIdx of this.gpuIndices(owner)) {
      const scheduler = this.schedulerFor(gpuIdx);
      try {
        rollbacks.push(await scheduler.acquire(owner));
      } catch (err) {
        // Partial multi-GPU acquisition helps nobody: hand back what we took.
        for (const rollback of rollbacks) rollback();
        throw err;
      }
    }
  }

  /**
   * Check every card this model would land on, without reserving anything.
   * Throws InsufficientVRAMError when no amount of eviction would make room.
   */
  async assertCanFit(owner: GpuOwner): Promise<void> {
    for (const gpuIdx of this.gpuIndices(owner)) {
      await this.schedulerFor(gpuIdx).checkFit(owner);
    }
  }

  /**
   * Refuse to reload a model that was evicted moments ago into a GPU that is
   * still full — that's the thrash loop, and the client is better served by a
   * 503 telling it when to come back.
   */
  assertNotCoolingDown(owner: GpuOwner): void {
    for (const gpuIdx of this.gpuIndices(owner)) {
      const scheduler = this.schedulers.get(gpuIdx);
      if (!scheduler || !scheduler.isInitialized) continue;
      const remainingMs = scheduler.cooldownRemaining(owner.name);
      if (remainingMs > 0 && scheduler.needsEviction(owner)) {
        throw new EvictionCooldownError(owner.name, Math.ceil(remainingMs / 1000));
      }
    }
  }

  /** Snapshot VRAM on the primary GPU before a model starts */
  async snapshotVRAM(owner: GpuOwner): Promise<number> {
    const primaryGpu = this.gpuIndices(owner)[0];
    const scheduler = this.schedulers.get(primaryGpu);
    return scheduler ? scheduler.snapshotVRAM() : 0;
  }

  /** Record observed VRAM after a model finishes loading */
  async recordObservedVRAM(owner: GpuOwner, preSnapshotGB: number): Promise<void> {
    const [primaryGpu, ...otherGpus] = this.gpuIndices(owner);
    const scheduler = this.schedulers.get(primaryGpu);
    if (!scheduler) return;
    await scheduler.recordObservedVRAM(owner, preSnapshotGB);
    // The measurement is the model's share of one card, which is what a
    // tensor-parallel model holds on each of the others too.
    for (const gpuIdx of otherGpus) this.schedulers.get(gpuIdx)?.refreshObservedVRAM(owner);
  }

  /** Mark a model as used, so eviction order is real LRU */
  touchActivity(owner: GpuOwner): void {
    for (const gpuIdx of this.gpuIndices(owner)) {
      this.schedulers.get(gpuIdx)?.touchActivity(owner);
    }
  }

  notifyStopped(owner: GpuOwner): void {
    for (const gpuIdx of this.gpuIndices(owner)) {
      this.schedulers.get(gpuIdx)?.notifyStopped(owner);
    }
  }

  /** Keep a sleeping model resident, at what its backend still holds per card. */
  async notifySleeping(owner: GpuOwner): Promise<void> {
    for (const gpuIdx of this.gpuIndices(owner)) {
      await this.schedulers.get(gpuIdx)?.notifySleeping(owner);
    }
  }

  async status(): Promise<Record<number, any>> {
    const out: Record<number, any> = {};
    for (const [gpu, scheduler] of this.schedulers) {
      out[gpu] = await scheduler.getStatus();
    }
    return out;
  }
}
