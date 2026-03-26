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
import { estimateVRAM, formatEstimate } from "./model-size";

const config = loadConfig();
const gpuLock = new GpuLock();

console.log("╔════════════════════════════════════════════╗");
console.log("║        locallm-router                      ║");
console.log("╚════════════════════════════════════════════╝");
console.log();

// Initialize GPU lock
await gpuLock.initialize();
console.log();

// Show configured models
console.log(`Configured models (${config.models.length}):`);
for (const m of config.models) {
  const estimate = estimateVRAM(m);
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

// Management API on separate port
const managementPort = config.managementPort || 30099;
const mgmtServer = Bun.serve({
  port: managementPort,
  fetch: async (req) => {
    const url = new URL(req.url);

    if (url.pathname === "/status") {
      return Response.json({
        models: proxy.getEntries().map((e) => ({
          name: e.config.name,
          aliases: e.config.aliases || [],
          backend: e.config.backend || "vllm",
          modelPath: e.config.modelPath,
          status: e.process.status,
          lastActivity: e.process.lastActivity,
          downloading: e.downloading,
          downloadError: e.downloadError,
          internalPort: e.internalPort,
        })),
        gpus: await gpuLock.status(),
      });
    }

    if (url.pathname === "/stop-all" && req.method === "POST") {
      await proxy.stop();
      return Response.json({ status: "all stopped" });
    }

    // ─── Reload config for a specific model ─────────────────
    const reloadMatch = url.pathname.match(/^\/reload-config\/(.+)$/);
    if (reloadMatch && req.method === "POST") {
      const modelName = decodeURIComponent(reloadMatch[1]);
      const newConfig = await proxy.reloadModelConfig(modelName);
      if (!newConfig) {
        return Response.json(
          { error: `Model ${modelName} not found in proxy or config file` },
          { status: 404 }
        );
      }
      return Response.json({ status: "reloaded", model: modelName, config: newConfig });
    }

    // ─── Restart a specific model (reload config + stop + start) ─
    // Supports runtime arg overrides:
    //   POST /restart/<model>?context=32768      → shorthand for -c override
    //   POST /restart/<model> { "overrides": { "-c": "32768" } }  → explicit overrides
    const restartMatch = url.pathname.match(/^\/restart\/(.+)$/);
    if (restartMatch && req.method === "POST") {
      const modelName = decodeURIComponent(restartMatch[1]);

      // Build arg overrides from query params and/or body
      let argOverrides: Record<string, string> = {};

      // Query param shorthand: ?context=N
      const contextParam = url.searchParams.get("context");
      if (contextParam) {
        argOverrides["-c"] = contextParam;
      }

      // Body overrides (merge, body takes precedence)
      try {
        const body = await req.json().catch(() => null);
        if (body?.overrides && typeof body.overrides === "object") {
          argOverrides = { ...argOverrides, ...body.overrides };
        }
      } catch {}

      const result = await proxy.restartModel(
        modelName,
        Object.keys(argOverrides).length > 0 ? argOverrides : undefined
      );
      if (result.error) {
        return Response.json(result, { status: result.status === "error" ? 500 : 200 });
      }
      return Response.json(result);
    }

    // ─── Stop a specific model ──────────────────────────────
    const stopMatch = url.pathname.match(/^\/stop\/(.+)$/);
    if (stopMatch && req.method === "POST") {
      const modelName = decodeURIComponent(stopMatch[1]);
      const entries = proxy.getEntries();
      const entry = entries.find((e) => e.config.name === modelName);
      if (!entry) {
        return Response.json({ error: `Unknown model: ${modelName}` }, { status: 404 });
      }
      // Use stopModel so GPU lock is released and balance shadows are cleaned up
      await proxy.stopModel(entry);
      return Response.json({ status: "stopped", model: modelName });
    }

    // ─── Start a specific model ─────────────────────────────
    const startMatch = url.pathname.match(/^\/start\/(.+)$/);
    if (startMatch && req.method === "POST") {
      const modelName = decodeURIComponent(startMatch[1]);
      const entries = proxy.getEntries();
      const entry = entries.find((e) => e.config.name === modelName);
      if (!entry) {
        return Response.json({ error: `Unknown model: ${modelName}` }, { status: 404 });
      }
      // Use ensureRunning so GPU lock + idle timer are properly set
      try {
        await proxy.startModel(entry);
        return Response.json({ status: "running", model: modelName });
      } catch (err) {
        return Response.json({ error: String(err) }, { status: 500 });
      }
    }

    return Response.json({
      port: config.port,
      managementPort,
      endpoints: {
        "/status": "GET — Status of all models + GPUs",
        "/stop-all": "POST — Stop all running models",
        "/reload-config/<model>": "POST — Reload config from disk for a model",
        "/restart/<model>": "POST — Restart a model with fresh config. Query: ?context=N for context size override. Body: { overrides: { '-c': 'N' } }",
        "/stop/<model>": "POST — Stop a specific model",
        "/start/<model>": "POST — Start a specific model",
      },
      models: proxy.getEntries().map((e) => ({
        name: e.config.name,
        status: e.process.status,
        backend: e.config.backend || "vllm",
      })),
    });
  },
});

// Dashboard — SvelteKit app (built with svelte-adapter-bun)
const dashboardPort = 30001;
let dashboardProc: import("bun").Subprocess | null = null;

const dashboardBuildPath = "./dashboard/build/index.js";
const dashboardExists = await Bun.file(dashboardBuildPath).exists();

if (dashboardExists) {
  dashboardProc = Bun.spawn(["bun", "run", dashboardBuildPath], {
    env: { ...process.env, PORT: String(dashboardPort), HOST: "0.0.0.0" },
    stdout: "inherit",
    stderr: "inherit",
  });
  console.log(`Dashboard:      http://localhost:${dashboardPort} (pid ${dashboardProc.pid})`);
} else {
  console.log(`Dashboard:      SKIPPED (no build at ${dashboardBuildPath})`);
}

console.log(`Management API: http://localhost:${managementPort}`);
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
