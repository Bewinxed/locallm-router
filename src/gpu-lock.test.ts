/**
 * Scheduler admission tests, on a fake driver.
 *
 * The numbers are the ones from the bug this file exists for: a tensor-parallel
 * 27B on two 24GB cards, put to sleep, then refused its own wake with a 507
 * because the 1.4GB of CUDA context its workers never let go of had been
 * reclassified as somebody else's memory.
 */

import { afterEach, describe, expect, setSystemTime, test } from "bun:test";
import type { ModelConfig } from "./config";
import { GpuLock, InsufficientVRAMError, type GpuOwner, type GpuProbe } from "./gpu-lock";

const MB = 1024;
const CARD_TOTAL_MB = 24 * MB;
/** A desktop session, on both cards. Not the router's, and never will be. */
const FOREIGN_MB = Math.round(4.5 * MB);
const FOREIGN_PID = 900;
/** Per card, awake: vLLM's 0.70 prealloc plus its non-torch overhead. */
const AWAKE_MB = Math.round(17.9 * MB);
/** Per card, asleep: weights went to host RAM, the CUDA context did not. */
const ASLEEP_MB = Math.round(1.4 * MB);

/**
 * nvidia-smi, without the nvidia. Tracks who holds what on each card and
 * answers the two queries the scheduler asks: total/used, and the per-process
 * rows — for every process, not just the router's.
 */
class FakeDriver {
  /** gpu -> pid -> MB */
  private held = new Map<number, Map<number, number>>();

  constructor(private readonly gpus: number[]) {
    for (const gpu of gpus) this.hold(gpu, FOREIGN_PID, FOREIGN_MB);
  }

  hold(gpu: number, pid: number, mb: number): void {
    let card = this.held.get(gpu);
    if (!card) {
      card = new Map();
      this.held.set(gpu, card);
    }
    card.set(pid, mb);
  }

  release(gpu: number, pid: number): void {
    this.held.get(gpu)?.delete(pid);
  }

  readonly probe: GpuProbe = async (gpu) => {
    if (!this.gpus.includes(gpu)) return null;
    const card = this.held.get(gpu) ?? new Map<number, number>();
    let usedMB = 0;
    for (const mb of card.values()) usedMB += mb;
    return {
      name: "Fake RTX 4090",
      totalMB: CARD_TOTAL_MB,
      usedMB,
      computeApps: [...card].map(([pid, usedMemoryMB]) => ({ gpu, pid, usedMemoryMB })),
    };
  };
}

/**
 * A backend the router started. One pid per card it runs on — which is what a
 * tensor-parallel vLLM looks like to the driver: VLLM::Worker_TP0 accounted
 * for on GPU 0, TP1 on GPU 1, each holding that card's share and no more.
 * pids() returns the whole group, as the real one does.
 */
class FakeBackend {
  readonly owner: GpuOwner;
  /** The process exists — which is all the scheduler is entitled to know. */
  private live = false;

  constructor(
    readonly config: ModelConfig,
    private readonly driver: FakeDriver,
    private readonly pidBase: number
  ) {
    this.owner = {
      name: config.name,
      gpu: config.gpu,
      gpus: config.gpus,
      config,
      isAlive: () => this.live,
      stopProcess: async () => this.stop(),
      pids: () => (this.live ? this.gpus().map((gpu) => this.pidBase + gpu) : []),
    };
  }

  private gpus(): number[] {
    return this.config.gpus ?? [this.config.gpu];
  }

  /** Spawned and taking memory. Serving or still loading — same process. */
  holds(mbPerCard: number): void {
    for (const gpu of this.gpus()) this.driver.hold(gpu, this.pidBase + gpu, mbPerCard);
    this.live = true;
  }

  /** Running, but on the CPU: the driver has never heard of it. */
  runsOffCard(): void {
    this.live = true;
  }

  stop(): void {
    for (const gpu of this.gpus()) this.driver.release(gpu, this.pidBase + gpu);
    this.live = false;
  }
}

/** Model names are deliberately not real ones: observed VRAM is read from disk. */
function tpConfig(): ModelConfig {
  return {
    name: "tp-27b",
    modelPath: "/models/tp-27b-awq",
    backend: "vllm",
    gpu: 0,
    gpus: [0, 1],
    idleTimeout: 600_000,
    // 35.8GB over two cards is 17.9GB each, above the 16.8GB prealloc floor.
    vramGB: 35.8,
    extraArgs: ["--tensor-parallel-size", "2", "--gpu-memory-utilization", "0.70"],
  };
}

describe("GpuScheduler admission", () => {
  afterEach(() => setSystemTime());

  test("a sleeping tensor-parallel model wakes into the VRAM it still holds", async () => {
    const driver = new FakeDriver([0, 1]);
    const lock = new GpuLock(driver.probe);
    const model = new FakeBackend(tpConfig(), driver, 1000);

    lock.register(model.owner);

    await lock.acquire(model.owner);
    model.holds(AWAKE_MB);

    // ─── asleep ───────────────────────────────────────────
    // Sleep hands the weights back to host RAM; the process, and the CUDA
    // context it holds on each card, stay exactly where they were.
    model.holds(ASLEEP_MB);
    await lock.notifySleeping(model.owner);

    const asleep = await lock.status();
    for (const gpu of [0, 1]) {
      const card = asleep[gpu];
      expect(card.loaded).toHaveLength(1);

      const [resident] = card.loaded;
      expect(resident.name).toBe("tp-27b");
      expect(resident.sleeping).toBe(true);
      // This card's worker only. Summing both would read 2.8GB here.
      expect(resident.effectiveVRAM).toBeCloseTo(1.4, 2);
      // The dashboard reads these; they must survive a sleeping resident.
      expect(resident).toMatchObject({
        estimatedVRAM: expect.any(Number),
        priority: 0,
        lastActivity: expect.any(Number),
      });
      expect(resident).toHaveProperty("observedVRAM");

      // The driver sees 5.9GB used here; 1.4GB of it is ours, and the router
      // does not get to tax itself for memory its own workers are holding.
      expect(card.otherVRAM).toBeCloseTo(4.5, 2);
      expect(card.usedVRAM).toBeCloseTo(1.4, 2);
      // Note this is below the model's 17.9GB footprint: admitting a wake on
      // the full number — as a cold start — is exactly the 507 we're fixing.
      expect(card.availableVRAM).toBeCloseTo(17.1, 2);
    }

    // ─── waking ───────────────────────────────────────────
    await lock.acquire(model.owner);
    model.holds(AWAKE_MB);

    const awake = await lock.status();
    for (const gpu of [0, 1]) {
      const [resident] = awake[gpu].loaded;
      expect(resident.sleeping).toBe(false);
      expect(resident.effectiveVRAM).toBeCloseTo(17.9, 2);
      expect(awake[gpu].otherVRAM).toBeCloseTo(4.5, 2);
    }
  });

  test("a model still loading keeps its reservation", async () => {
    const driver = new FakeDriver([0]);
    const lock = new GpuLock(driver.probe);
    const model = new FakeBackend(
      {
        name: "loading-27b",
        modelPath: "/models/loading-27b-awq",
        backend: "vllm",
        gpu: 0,
        idleTimeout: 600_000,
        vramGB: 17.9,
        extraArgs: ["--gpu-memory-utilization", "0.70"],
      },
      driver,
      1000
    );

    lock.register(model.owner);
    await lock.acquire(model.owner);

    // Minutes into a cold start, the weights are still arriving: far less on
    // the card than was reserved for it. That is not a stale reservation.
    model.holds(3 * MB);
    setSystemTime(new Date(Date.now() + 6_000));

    const status = await lock.status();
    expect(status[0].loaded).toHaveLength(1);
    expect(status[0].loaded[0].effectiveVRAM).toBeCloseTo(17.9, 2);
    expect(status[0].availableVRAM).toBeCloseTo(0.6, 2);
  });

  test("a backend the router started is never foreign, whatever the resident list says", async () => {
    const driver = new FakeDriver([0]);
    const lock = new GpuLock(driver.probe);
    const model = new FakeBackend(
      {
        name: "lingering-27b",
        modelPath: "/models/lingering-27b-awq",
        backend: "vllm",
        gpu: 0,
        idleTimeout: 600_000,
        vramGB: 17.9,
        extraArgs: ["--gpu-memory-utilization", "0.70"],
      },
      driver,
      1000
    );

    lock.register(model.owner);
    await lock.acquire(model.owner);
    model.holds(AWAKE_MB);

    // Bookkeeping says the model is gone while its processes are still on the
    // card — which is how its own memory came to be billed to a stranger.
    lock.notifyStopped(model.owner);

    // Past the cached split, so the next status really re-reads the card.
    setSystemTime(new Date(Date.now() + 6_000));

    const status = await lock.status();
    expect(status[0].loaded).toHaveLength(0);
    expect(status[0].otherVRAM).toBeCloseTo(4.5, 2);
  });

  test("a model that genuinely does not fit is refused with the numbers", async () => {
    const driver = new FakeDriver([0]);
    const lock = new GpuLock(driver.probe);

    const resident = new FakeBackend(
      {
        name: "resident-27b",
        modelPath: "/models/resident-27b-awq",
        backend: "vllm",
        gpu: 0,
        idleTimeout: 600_000,
        vramGB: 17.9,
        extraArgs: ["--gpu-memory-utilization", "0.70"],
      },
      driver,
      1000
    );
    const incoming = new FakeBackend(
      {
        name: "incoming-20b",
        modelPath: "/models/incoming-20b-awq",
        backend: "vllm",
        gpu: 0,
        idleTimeout: 600_000,
        vramGB: 20,
        extraArgs: ["--gpu-memory-utilization", "0.80"],
      },
      driver,
      2000
    );

    lock.register(resident.owner);
    lock.register(incoming.owner);

    await lock.acquire(resident.owner);
    resident.holds(AWAKE_MB);

    // 24.0 total - 1.0 headroom - 4.5 desktop = 18.5GB, even after evicting
    // everything the router owns. 20GB was never going to fit.
    const refused = lock.acquire(incoming.owner);
    await expect(refused).rejects.toThrow(InsufficientVRAMError);
    await expect(refused).rejects.toThrow(
      "Cannot fit incoming-20b (20.0GB) on GPU 0: 24.0GB total, 1.0GB headroom, " +
        "4.5GB used outside the router, resident: resident-27b 17.9GB"
    );
  });

  test("a CPU-only sidecar is not evicted to make room — it holds nothing to give back", async () => {
    const driver = new FakeDriver([0]);
    const lock = new GpuLock(driver.probe);

    const sidecar = new FakeBackend(
      {
        name: "parakeet",
        modelPath: "/models/parakeet-tdt",
        backend: "parakeet-asr",
        gpu: 0,
        idleTimeout: 300_000,
        parameterCount: 0.6,
        vramGB: 0,
        extraArgs: ["--device", "cpu"],
      },
      driver,
      1000
    );
    const resident = new FakeBackend(
      {
        name: "resident-27b",
        modelPath: "/models/resident-27b-awq",
        backend: "vllm",
        gpu: 0,
        idleTimeout: 600_000,
        vramGB: 17.9,
        extraArgs: ["--gpu-memory-utilization", "0.70"],
      },
      driver,
      2000
    );
    const incoming = new FakeBackend(
      {
        name: "incoming-27b",
        modelPath: "/models/incoming-27b-awq",
        backend: "vllm",
        gpu: 0,
        idleTimeout: 600_000,
        vramGB: 17.9,
        extraArgs: ["--gpu-memory-utilization", "0.70"],
      },
      driver,
      3000
    );

    lock.register(sidecar.owner);
    lock.register(resident.owner);
    lock.register(incoming.owner);

    // The sidecar goes first, so LRU order offers it up before the 27B.
    await lock.acquire(sidecar.owner);
    sidecar.runsOffCard();
    await lock.acquire(resident.owner);
    resident.holds(AWAKE_MB);

    await lock.acquire(incoming.owner);

    expect(resident.owner.isAlive()).toBe(false);
    expect(sidecar.owner.isAlive()).toBe(true);

    const status = await lock.status();
    expect(status[0].loaded.map((m) => m.name)).toEqual(["parakeet", "incoming-27b"]);
  });
});
