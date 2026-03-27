/**
 * GPU Lock — VRAM-aware scheduler for model coexistence.
 *
 * Models that share a GPU can coexist if their combined VRAM fits.
 * If a new model doesn't fit, the scheduler evicts the least-recently-used
 * model(s) until there's enough room.
 */

import {
  estimateVRAM,
  getGpuInfo,
  formatEstimate,
  getGpuMemoryUtil,
  type ModelSizeEstimate,
} from "./model-size";
import type { ModelConfig } from "./config";

export interface GpuOwner {
  readonly name: string;
  readonly gpu: number;
  readonly gpus?: number[];
  readonly config: ModelConfig;
  isRunning(): boolean;
  stopProcess(): Promise<void>;
}

interface LoadedModel {
  owner: GpuOwner;
  estimate: ModelSizeEstimate;
  observedVRAM: number | null;
  loadedAt: number;
  lastActivity: number;
}

interface EvictionRecord {
  modelName: string;
  evictedAt: number;
}

const VRAM_SAFETY_MARGIN = 0.95;
const ANTI_THRASH_COOLDOWN_MS = 30_000;
const VRAM_SNAPSHOT_SETTLE_MS = 3_000;

class GpuScheduler {
  readonly gpu: number;
  private totalVRAM: number = 0;
  private loaded = new Map<string, LoadedModel>();
  private estimates = new Map<string, ModelSizeEstimate>();
  private recentEvictions: EvictionRecord[] = [];
  private lastReconcileTime = 0;
  private static readonly RECONCILE_INTERVAL_MS = 5000;

  constructor(gpu: number) {
    this.gpu = gpu;
  }

  async initialize(): Promise<void> {
    const gpus = await getGpuInfo();
    const gpuInfo = gpus.find((g) => g.index === this.gpu);
    if (gpuInfo) {
      this.totalVRAM = gpuInfo.totalMemoryMB / 1024;
      console.log(`[gpu-lock] GPU ${this.gpu}: ${gpuInfo.name}, ${this.totalVRAM.toFixed(1)}GB total`);
    } else {
      this.totalVRAM = 24;
      console.log(`[gpu-lock] GPU ${this.gpu}: Unknown (assuming 24GB)`);
    }
  }

  registerModel(owner: GpuOwner): void {
    let estimate = estimateVRAM(owner.config);
    if (owner.config.vramGB !== undefined) {
      estimate = { ...estimate, totalVRAM: owner.config.vramGB, confidence: "manual" as any };
    }
    const numGpus = owner.gpus ? owner.gpus.length : 1;
    const perGpuEstimate: typeof estimate =
      numGpus > 1
        ? {
            ...estimate,
            weightsGB: estimate.weightsGB / numGpus,
            kvCacheGB: estimate.kvCacheGB / numGpus,
            totalVRAM: estimate.totalVRAM / numGpus,
          }
        : estimate;
    this.estimates.set(owner.name, perGpuEstimate);
    if (numGpus > 1) {
      console.log(
        `[gpu-lock] GPU ${this.gpu}: ${owner.name} -> ${formatEstimate(estimate)} (${estimate.totalVRAM.toFixed(1)}GB total / ${numGpus} GPUs = ${perGpuEstimate.totalVRAM.toFixed(1)}GB here)`
      );
    } else {
      console.log(`[gpu-lock] GPU ${this.gpu}: ${owner.name} -> ${formatEstimate(estimate)}`);
    }
  }

  private getModelVRAM(model: LoadedModel): number {
    return model.observedVRAM ?? model.estimate.totalVRAM;
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

    const gpus = await getGpuInfo();
    const gpuInfo = gpus.find((g) => g.index === this.gpu);
    if (!gpuInfo) return;

    const actualUsedGB = gpuInfo.usedMemoryMB / 1024;
    const estimatedUsedGB = this.getEstimatedVRAM();

    if (estimatedUsedGB > 2 && actualUsedGB < estimatedUsedGB - 2) {
      let removedAny = false;
      for (const [name, model] of this.loaded) {
        if (!model.owner.isRunning()) {
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

  private getAvailableVRAM(): number {
    return this.totalVRAM * VRAM_SAFETY_MARGIN - this.getEstimatedVRAM();
  }

  canFit(owner: GpuOwner): boolean {
    const estimate = this.estimates.get(owner.name);
    if (!estimate) return false;
    return estimate.totalVRAM <= this.totalVRAM;
  }

  /** Snapshot current GPU memory usage (call before model start) */
  async snapshotVRAM(): Promise<number> {
    const gpus = await getGpuInfo();
    const gpuInfo = gpus.find((g) => g.index === this.gpu);
    return gpuInfo ? gpuInfo.usedMemoryMB / 1024 : 0;
  }

  /** Record observed VRAM after a model finishes loading */
  async recordObservedVRAM(owner: GpuOwner, preSnapshotGB: number): Promise<void> {
    await new Promise((r) => setTimeout(r, VRAM_SNAPSHOT_SETTLE_MS));
    const postGB = await this.snapshotVRAM();
    const delta = Math.max(0, postGB - preSnapshotGB);
    const model = this.loaded.get(owner.name);
    if (!model || delta < 0.5) return;

    model.observedVRAM = delta;
    const estGB = model.estimate.totalVRAM;
    const drift = Math.abs(delta - estGB);
    if (drift > 1) {
      console.log(
        `[gpu-lock] GPU ${this.gpu}: ${owner.name} observed ${delta.toFixed(1)}GB (estimated ${estGB.toFixed(1)}GB, drift ${drift.toFixed(1)}GB)`
      );
    }
  }

  /** Check if a model was recently evicted (anti-thrash) */
  isRecentlyEvicted(modelName: string): boolean {
    const now = Date.now();
    this.recentEvictions = this.recentEvictions.filter(
      (e) => now - e.evictedAt < ANTI_THRASH_COOLDOWN_MS
    );
    return this.recentEvictions.some((e) => e.modelName === modelName);
  }

  private recordEviction(modelName: string): void {
    this.recentEvictions.push({ modelName, evictedAt: Date.now() });
  }

  private getEvictionCandidates(neededVRAM: number): GpuOwner[] {
    const candidates = [...this.loaded.values()].sort((a, b) => {
      const pa = a.owner.config.priority ?? 0;
      const pb = b.owner.config.priority ?? 0;
      if (pa !== pb) return pa - pb;
      return a.lastActivity - b.lastActivity;
    });
    const toEvict: GpuOwner[] = [];
    let freedVRAM = 0;
    for (const model of candidates) {
      if (freedVRAM >= neededVRAM) break;
      toEvict.push(model.owner);
      freedVRAM += this.getModelVRAM(model);
    }
    return toEvict;
  }

  async acquire(owner: GpuOwner): Promise<void> {
    await this.reconcileWithGpu();

    if (this.loaded.has(owner.name)) {
      this.loaded.get(owner.name)!.lastActivity = Date.now();
      return;
    }

    const estimate = this.estimates.get(owner.name);
    if (!estimate) throw new Error(`Model ${owner.name} not registered`);

    const gpuUtil = getGpuMemoryUtil(owner.config.extraArgs);
    const hasOtherModels = this.loaded.size > 0;
    const margin = hasOtherModels ? VRAM_SAFETY_MARGIN : 1.0;
    const usableVRAM = this.totalVRAM * gpuUtil * margin;
    const available = usableVRAM - this.getEstimatedVRAM();

    if (estimate.totalVRAM <= available) {
      console.log(`[gpu-lock] GPU ${this.gpu}: ${owner.name} fits (need ${estimate.totalVRAM.toFixed(1)}GB, have ${available.toFixed(1)}GB)`);
      this.loaded.set(owner.name, { owner, estimate, observedVRAM: null, loadedAt: Date.now(), lastActivity: Date.now() });
      return;
    }

    const needsFullGpu = estimate.totalVRAM > this.totalVRAM * gpuUtil;
    const toEvict = needsFullGpu
      ? [...this.loaded.values()].map((m) => m.owner)
      : this.getEvictionCandidates(estimate.totalVRAM - available);

    if (toEvict.length === 0 && estimate.totalVRAM > this.totalVRAM) {
      throw new Error(`Cannot fit ${owner.name} on GPU ${this.gpu}`);
    }

    console.log(`[gpu-lock] GPU ${this.gpu}: evicting ${toEvict.map((o) => o.name).join(", ")} for ${owner.name}`);

    for (const victim of toEvict) {
      try {
        await victim.stopProcess();
        this.recordEviction(victim.name);
        this.loaded.delete(victim.name);
      } catch (err) {
        console.error(`[gpu-lock] GPU ${this.gpu}: failed to evict ${victim.name}:`, err);
      }
    }

    this.loaded.set(owner.name, { owner, estimate, observedVRAM: null, loadedAt: Date.now(), lastActivity: Date.now() });
  }

  touchActivity(owner: GpuOwner): void {
    const model = this.loaded.get(owner.name);
    if (model) model.lastActivity = Date.now();
  }

  release(owner: GpuOwner): void {
    this.loaded.delete(owner.name);
  }

  notifyStopped(owner: GpuOwner): void {
    this.loaded.delete(owner.name);
  }

  get currentModels(): string[] { return [...this.loaded.keys()]; }

  async getStatus() {
    await this.reconcileWithGpu();
    return {
      totalVRAM: Math.round(this.totalVRAM * 100) / 100,
      usedVRAM: Math.round(this.getEstimatedVRAM() * 100) / 100,
      availableVRAM: Math.round(this.getAvailableVRAM() * 100) / 100,
      loaded: [...this.loaded.entries()].map(([name, m]) => ({
        name,
        estimatedVRAM: m.estimate.totalVRAM,
        observedVRAM: m.observedVRAM,
        effectiveVRAM: this.getModelVRAM(m),
        priority: m.owner.config.priority ?? 0,
        lastActivity: m.lastActivity,
      })),
    };
  }
}

export class GpuLock {
  private schedulers = new Map<number, GpuScheduler>();
  private initialized = false;

  async initialize(): Promise<void> {
    if (this.initialized) return;
    const gpus = await getGpuInfo();
    console.log(`[gpu-lock] Detected ${gpus.length} GPU(s)`);
    for (const gpu of gpus) {
      const scheduler = new GpuScheduler(gpu.index);
      await scheduler.initialize();
      this.schedulers.set(gpu.index, scheduler);
    }
    this.initialized = true;
  }

  private gpuIndices(owner: GpuOwner): number[] {
    return owner.gpus && owner.gpus.length > 0 ? owner.gpus : [owner.gpu];
  }

  register(owner: GpuOwner): void {
    for (const gpuIdx of this.gpuIndices(owner)) {
      if (!this.schedulers.has(gpuIdx)) {
        const scheduler = new GpuScheduler(gpuIdx);
        this.schedulers.set(gpuIdx, scheduler);
      }
      this.schedulers.get(gpuIdx)!.registerModel(owner);
    }
  }

  async acquire(owner: GpuOwner): Promise<void> {
    for (const gpuIdx of this.gpuIndices(owner)) {
      const scheduler = this.schedulers.get(gpuIdx);
      if (!scheduler) throw new Error(`GPU ${gpuIdx} not registered`);
      await scheduler.acquire(owner);
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
    const primaryGpu = this.gpuIndices(owner)[0];
    const scheduler = this.schedulers.get(primaryGpu);
    if (scheduler) await scheduler.recordObservedVRAM(owner, preSnapshotGB);
  }

  /** Check if a model was recently evicted (anti-thrash cooldown) */
  isRecentlyEvicted(modelName: string): boolean {
    for (const scheduler of this.schedulers.values()) {
      if (scheduler.isRecentlyEvicted(modelName)) return true;
    }
    return false;
  }

  notifyStopped(owner: GpuOwner): void {
    for (const gpuIdx of this.gpuIndices(owner)) {
      this.schedulers.get(gpuIdx)?.notifyStopped(owner);
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
