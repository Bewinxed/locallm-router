#!/usr/bin/env bun
/**
 * locallm-router CLI
 *
 *   bunx locallm-router up                 Run the router in the foreground
 *   bunx locallm-router init               Scaffold a config.json
 *   bunx locallm-router status             Show model + GPU status
 *   bunx locallm-router stop-all           Stop every running model
 *   bunx locallm-router doctor             Check backends, GPUs, and config
 *   bunx locallm-router service install    Run at login (systemd/launchd)
 *   bunx locallm-router service uninstall  Remove the login service
 *
 * Config resolution: --config flag → $LLM_ROUTER_CONFIG → ./config.json
 * → ~/.config/locallm-router/config.json
 */

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

const HOME = homedir();
const CONFIG_DIR = join(HOME, ".config", "locallm-router");
const DEFAULT_HF_HOME = join(HOME, "models");
const PKG_ROOT = new URL("..", import.meta.url).pathname;

const args = process.argv.slice(2);
const command = args[0];

function flagValue(name: string): string | undefined {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
}

function resolveConfigPath(): string | null {
  const explicit = flagValue("--config") ?? process.env.LLM_ROUTER_CONFIG;
  if (explicit) return resolve(explicit);
  const cwdConfig = resolve("config.json");
  if (existsSync(cwdConfig)) return cwdConfig;
  const userConfig = join(CONFIG_DIR, "config.json");
  if (existsSync(userConfig)) return userConfig;
  return null;
}

function applyEnvDefaults(configPath: string) {
  process.env.LLM_ROUTER_CONFIG = configPath;
  process.env.HF_HOME ??= DEFAULT_HF_HOME;
}

function readPort(configPath: string | null): number {
  try {
    if (configPath) return JSON.parse(readFileSync(configPath, "utf-8")).port || 30000;
  } catch {}
  return 30000;
}

function fail(message: string): never {
  console.error(`✗ ${message}`);
  process.exit(1);
}

// ─── Commands ─────────────────────────────────────────────────

async function up() {
  const configPath = resolveConfigPath();
  if (!configPath) {
    fail(
      `No config found. Run \`locallm-router init\` first, or pass --config <path>.\n` +
        `  Looked for: ./config.json, ${join(CONFIG_DIR, "config.json")}`
    );
  }
  applyEnvDefaults(configPath);
  console.log(`Config: ${configPath}`);
  console.log(`Models: ${process.env.HF_HOME}`);
  await import("./index");
}

function init() {
  const target = flagValue("--config") ?? join(CONFIG_DIR, "config.json");
  if (existsSync(target) && !args.includes("--force")) {
    fail(`${target} already exists (use --force to overwrite)`);
  }
  mkdirSync(dirname(target), { recursive: true });
  const example = join(PKG_ROOT, "config.example.json");
  writeFileSync(target, readFileSync(example, "utf-8"));
  console.log(`✓ Wrote ${target}`);
  console.log("  Edit it, then run: locallm-router up");
}

async function status() {
  const port = readPort(resolveConfigPath());
  try {
    const res = await fetch(`http://localhost:${port}/manager/status`, {
      signal: AbortSignal.timeout(5000),
    });
    const data = (await res.json()) as {
      models: { name: string; status: string; backend: string; gpu?: number }[];
      gpus: Record<string, { usedVRAM: number; totalVRAM: number }>;
    };
    console.log("Models:");
    for (const m of data.models) {
      const icon = m.status === "running" ? "●" : m.status === "sleeping" ? "◐" : "○";
      console.log(`  ${icon} ${m.name.padEnd(24)} ${m.status.padEnd(10)} ${m.backend} (gpu ${m.gpu ?? "-"})`);
    }
    console.log("GPUs:");
    for (const [gpu, lock] of Object.entries(data.gpus)) {
      console.log(`  GPU ${gpu}: ${(lock.usedVRAM / 1024).toFixed(1)} / ${(lock.totalVRAM / 1024).toFixed(1)} GB reserved`);
    }
  } catch {
    fail(`Router is not responding on :${port} — is it running? (locallm-router up)`);
  }
}

async function stopAll() {
  const port = readPort(resolveConfigPath());
  try {
    const res = await fetch(`http://localhost:${port}/manager/stop-all`, {
      method: "POST",
      signal: AbortSignal.timeout(60_000),
    });
    console.log(res.ok ? "✓ All models stopped" : `✗ Router returned ${res.status}`);
  } catch {
    fail(`Router is not responding on :${port}`);
  }
}

async function doctor() {
  const checks: [string, boolean, string][] = [];

  checks.push(["bun", true, Bun.version]);

  const llama = Bun.which("llama-server");
  checks.push(["llama-server", !!llama, llama ?? "not on PATH — GGUF models need it"]);

  const hf = Bun.which("hf") ?? Bun.which("huggingface-cli");
  checks.push(["hf cli", !!hf, hf ?? "not on PATH — HF downloads need it (uv tool install huggingface_hub)"]);

  const vllm = Bun.which("vllm");
  checks.push(["vllm", !!vllm, vllm ?? "not on PATH (only needed for vllm-backend models)"]);

  try {
    const proc = Bun.spawnSync(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"]);
    const out = proc.stdout.toString().trim();
    checks.push(["nvidia gpus", proc.exitCode === 0 && !!out, out.split("\n").join("; ") || "nvidia-smi failed"]);
  } catch {
    checks.push(["nvidia gpus", false, "nvidia-smi not found"]);
  }

  const configPath = resolveConfigPath();
  checks.push(["config", !!configPath, configPath ?? "none found — run locallm-router init"]);

  const dashboard = existsSync(join(PKG_ROOT, "dashboard", "build", "index.js"));
  checks.push(["dashboard", dashboard, dashboard ? "built" : "not built (API still works)"]);

  let ok = true;
  for (const [name, passed, detail] of checks) {
    console.log(`  ${passed ? "✓" : "✗"} ${name.padEnd(14)} ${detail}`);
    if (!passed && ["config"].includes(name)) ok = false;
  }
  process.exit(ok ? 0 : 1);
}

// ─── Login service (systemd user unit / launchd agent) ────────

const SERVICE_NAME = "locallm-router";

function servicePaths() {
  if (process.platform === "darwin") {
    return { kind: "launchd" as const, path: join(HOME, "Library", "LaunchAgents", `com.${SERVICE_NAME}.plist`) };
  }
  return { kind: "systemd" as const, path: join(HOME, ".config", "systemd", "user", `${SERVICE_NAME}.service`) };
}

function serviceInstall() {
  const configPath = resolveConfigPath();
  if (!configPath) fail("No config found — run `locallm-router init` before installing the service.");

  const bunBin = process.execPath;
  const cliPath = new URL(import.meta.url).pathname;
  const hfHome = process.env.HF_HOME ?? DEFAULT_HF_HOME;
  // User services get a minimal PATH; include bun's dir and ~/.local/bin
  // (llama-server, hf) explicitly.
  const path = `${dirname(bunBin)}:${join(HOME, ".local", "bin")}:/usr/local/bin:/usr/bin:/bin`;
  const { kind, path: unitPath } = servicePaths();

  mkdirSync(dirname(unitPath), { recursive: true });

  if (kind === "systemd") {
    writeFileSync(
      unitPath,
      `[Unit]
Description=locallm-router — local LLM router
After=network-online.target

[Service]
Type=simple
ExecStart=${bunBin} ${cliPath} up
Environment=LLM_ROUTER_CONFIG=${configPath}
Environment=HF_HOME=${hfHome}
Environment=PATH=${path}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
`
    );
    const run = (cmd: string[]) => Bun.spawnSync(cmd, { stdout: "inherit", stderr: "inherit" });
    run(["systemctl", "--user", "daemon-reload"]);
    run(["systemctl", "--user", "enable", "--now", SERVICE_NAME]);
    console.log(`✓ Installed ${unitPath}`);
    console.log("  Starts at login. Follow logs:  journalctl --user -u locallm-router -f");
    console.log("  Start at BOOT (no login needed): loginctl enable-linger $USER");
  } else {
    writeFileSync(
      unitPath,
      `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key><string>com.${SERVICE_NAME}</string>
	<key>ProgramArguments</key>
	<array>
		<string>${bunBin}</string>
		<string>${cliPath}</string>
		<string>up</string>
	</array>
	<key>EnvironmentVariables</key>
	<dict>
		<key>LLM_ROUTER_CONFIG</key><string>${configPath}</string>
		<key>HF_HOME</key><string>${hfHome}</string>
		<key>PATH</key><string>${path}</string>
	</dict>
	<key>RunAtLoad</key><true/>
	<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
	<key>StandardOutPath</key><string>/tmp/${SERVICE_NAME}.log</string>
	<key>StandardErrorPath</key><string>/tmp/${SERVICE_NAME}.err</string>
</dict>
</plist>
`
    );
    Bun.spawnSync(["launchctl", "unload", unitPath], {});
    Bun.spawnSync(["launchctl", "load", "-w", unitPath], { stdout: "inherit", stderr: "inherit" });
    console.log(`✓ Installed ${unitPath} (runs at login)`);
    console.log(`  Logs: /tmp/${SERVICE_NAME}.log`);
  }
}

function serviceUninstall() {
  const { kind, path: unitPath } = servicePaths();
  if (!existsSync(unitPath)) fail("Service is not installed.");
  if (kind === "systemd") {
    Bun.spawnSync(["systemctl", "--user", "disable", "--now", SERVICE_NAME], { stdout: "inherit", stderr: "inherit" });
  } else {
    Bun.spawnSync(["launchctl", "unload", "-w", unitPath], {});
  }
  rmSync(unitPath);
  console.log(`✓ Removed ${unitPath}`);
}

// ─── Dispatch ─────────────────────────────────────────────────

switch (command) {
  case "up":
    await up();
    break;
  case "init":
    init();
    break;
  case "status":
    await status();
    break;
  case "stop-all":
    await stopAll();
    break;
  case "doctor":
    await doctor();
    break;
  case "service":
    if (args[1] === "install") serviceInstall();
    else if (args[1] === "uninstall") serviceUninstall();
    else fail("Usage: locallm-router service <install|uninstall>");
    break;
  default:
    console.log(`locallm-router — one endpoint, any local model, on demand

Usage:
  locallm-router up                  Run the router (foreground)
  locallm-router init                Scaffold a config.json
  locallm-router status              Model + GPU status
  locallm-router stop-all            Stop every running model
  locallm-router doctor              Check backends, GPUs, and config
  locallm-router service install     Run automatically at login
  locallm-router service uninstall   Remove the login service

Options:
  --config <path>    Config file (default: ./config.json or ~/.config/locallm-router/config.json)
`);
    process.exit(command ? 1 : 0);
}
