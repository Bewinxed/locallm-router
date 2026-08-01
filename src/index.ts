#!/usr/bin/env bun
/**
 * locallm-router — Local LLM Router
 *
 * Single port serves all models. The `model` field in the request
 * payload determines which backend to route to. Models are lazily
 * downloaded and started on first request.
 *
 * Endpoints:
 *   /v1/models              — List all configured models
 *   /v1/chat/completions    — Chat (model field routes to backend)
 *   /v1/completions         — Completions
 *   /v1/embeddings          — Embeddings
 *   /manager/status         — Management API (all models + GPUs)
 *   /manager/start/<model>  — Force start a model
 *   /manager/stop/<model>   — Force stop a model
 *   /manager/stop-all       — Stop all running models
 *
 * Usage:
 *   bun run index.ts
 *
 * Environment:
 *   LLM_ROUTER_CONFIG          — Path to config.json
 *   HF_HOME                   — HuggingFace cache directory
 *   LLM_IDLE_TIMEOUT_MINUTES  — Override idle timeout for all models
 */

import { loadConfig } from "./config";
import { UnifiedProxy } from "./proxy";
import { GpuLock } from "./gpu-lock";
import { resolveEstimate } from "./model-size";

const config = loadConfig();
const gpuLock = new GpuLock();

console.log("╔════════════════════════════════════════════╗");
console.log("║        locallm-router                      ║");
console.log("╚════════════════════════════════════════════╝");
console.log();

// Initialize GPU lock
await gpuLock.initialize();
if (gpuLock.gpuCount === 0 && config.models.length > 0) {
  console.error("✗ No GPUs detected (nvidia-smi returned nothing) but models are configured.");
  console.error("  Install the NVIDIA driver/CUDA, or set LLM_ROUTER_ALLOW_NO_GPU=1 to run anyway");
  console.error("  (VRAM scheduling then works off estimates only).");
  if (!process.env.LLM_ROUTER_ALLOW_NO_GPU) process.exit(1);
}
console.log();

// Show configured models
console.log(`Configured models (${config.models.length}):`);
for (const m of config.models) {
  const estimate = await resolveEstimate(m);
  const gpuStr = m.gpus ? `GPUs ${m.gpus.join(",")}` : `GPU ${m.gpu}`;
  const numGpus = m.gpus ? m.gpus.length : 1;
  const vramStr = numGpus > 1
    ? `~${estimate.totalVRAM.toFixed(1)}GB total (~${(estimate.totalVRAM / numGpus).toFixed(1)}GB × ${numGpus})`
    : `~${estimate.totalVRAM.toFixed(1)}GB`;

  console.log(`  ${m.name}:`);
  console.log(`    Backend: ${m.backend || "vllm"}`);
  console.log(`    Model:   ${m.modelPath}`);
  console.log(`    GPU:     ${gpuStr}`);
  console.log(`    VRAM:    ${vramStr} (${estimate.confidence})`);
  console.log(`    Idle:    ${m.idleTimeout / 1000 / 60} min`);
  if (m.aliases?.length) console.log(`    Aliases: ${m.aliases.join(", ")}`);
  if (m.balanceGpus?.length) {
    console.log(`    Balance: GPUs [${m.balanceGpus.join(", ")}] — use :balance, :gpu0, :gpu1`);
  }
  console.log();
}

// Create the unified proxy
const proxy = new UnifiedProxy(config.models, gpuLock);
proxy.start(config.port);
proxy.startIdleSweep(60_000); // Safety net: check for idle models every 60s
proxy.startHealthChecks(config.healthCheckInterval);

// Management API on a separate port — the same handler the proxy serves under
// /manager/*, mounted at the root. Loopback-only unless told otherwise, since
// it can start and stop processes.
const managementPort = config.managementPort || 30099;
const managementHost = process.env.ROUTER_MGMT_HOST || "127.0.0.1";
const mgmtServer = Bun.serve({
  port: managementPort,
  hostname: managementHost,
  idleTimeout: 255,
  fetch: (req) => proxy.handleManagement(req),
  error: (err) => {
    console.error("[management] Unhandled request error:", err);
    return Response.json({ error: "Internal error", details: String(err) }, { status: 500 });
  },
});

// Dashboard — SvelteKit app (built with svelte-adapter-bun)
const dashboardPort = Number(process.env.DASHBOARD_PORT) || 30001;
let dashboardProc: import("bun").Subprocess | null = null;

// Resolve relative to this package, not the cwd — the CLI can run from anywhere.
const dashboardBuildPath = new URL("../dashboard/build/index.js", import.meta.url).pathname;
const dashboardExists = await Bun.file(dashboardBuildPath).exists();

if (dashboardExists) {
  dashboardProc = Bun.spawn(["bun", "run", dashboardBuildPath], {
    env: {
      ...process.env,
      PORT: String(dashboardPort),
      HOST: "0.0.0.0",
      // svelte-adapter-bun defaults idleTimeout to 10s, which kills chat
      // streams while a cold model is still loading. 255 is Bun's max.
      IDLE_TIMEOUT: "255",
    },
    stdout: "inherit",
    stderr: "inherit",
  });
  console.log(`Dashboard:      http://localhost:${dashboardPort} (pid ${dashboardProc.pid})`);
} else {
  console.log(`Dashboard:      SKIPPED (no build at ${dashboardBuildPath})`);
}

console.log(`Management API: http://${managementHost}:${managementPort}`);
console.log(`Unified proxy:  http://localhost:${config.port}`);
console.log();
console.log("Models will download + start on first request.");
console.log("Press Ctrl+C to exit.");

// Graceful shutdown
const shutdown = async () => {
  console.log("\nShutting down...");
  if (dashboardProc) {
    dashboardProc.kill();
    await dashboardProc.exited;
  }
  await proxy.stop();
  mgmtServer.stop();
  process.exit(0);
};

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

// A single bad response must not take the router — and every loaded model —
// down with it.
process.on("unhandledRejection", (reason) => {
  console.error("[router] Unhandled promise rejection:", reason);
});
process.on("uncaughtException", (err) => {
  console.error("[router] Uncaught exception:", err);
});

// Last-resort cleanup: on a hard exit the backends must not outlive us
// holding VRAM. Best effort only — no awaiting inside an exit handler.
process.on("exit", () => {
  dashboardProc?.kill();
  proxy.killAllNow();
});
