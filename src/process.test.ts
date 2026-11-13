/**
 * The command lines the backends are spawned with.
 *
 * vLLM: an entry that opts out of sleep mode must not also be handed the flag
 * that opts in. argparse takes the last one either way, so the process behaves
 * — but the command line and the log it is printed to contradict each other,
 * and that is the line anyone debugging a sleep problem reads first.
 *
 * FreeToken: `ft serve` and the device it is pinned to, which is the whole of
 * what the router decides for it — every knob it has stays in extraArgs.
 */

import { describe, expect, spyOn, test } from "bun:test";
import type { ModelConfig } from "./config";
import { createProcess } from "./process";

/** A port no backend is configured for: the real ones start at 40001. */
const SCRATCH_PORT = 45999;

function vllmConfig(extraArgs: string[]): ModelConfig {
  return {
    name: "sleep-flags",
    modelPath: "vendor/Some-Model",
    backend: "vllm",
    gpu: 0,
    idleTimeout: 600_000,
    extraArgs,
  };
}

/**
 * Runs a launch as far as the spawn and no further. The backend never starts:
 * the stub records the argv and environment and throws, which launch() reports
 * as a failed spawn without leaving a process behind.
 */
async function spawnCall(
  config: ModelConfig
): Promise<{ cmd: string[]; env: Record<string, string> }> {
  const realSpawn = Bun.spawn;
  let cmd: string[] = [];
  let env: Record<string, string> = {};

  const spy = spyOn(Bun, "spawn").mockImplementation(((options: any) => {
    // claimPort probes the port with `ss` on the way in; that one is real.
    if (options.cmd[0] === "ss") return realSpawn(options);
    cmd = options.cmd;
    env = options.env;
    throw new Error("stubbed spawn");
  }) as typeof Bun.spawn);

  try {
    await createProcess(config, SCRATCH_PORT)
      .start()
      .catch(() => {});
  } finally {
    spy.mockRestore();
  }
  return { cmd, env };
}

describe("vLLM sleep flags", () => {
  test("an entry that disables sleep mode is not also handed --enable-sleep-mode", async () => {
    const { cmd } = await spawnCall(
      vllmConfig(["--no-enable-sleep-mode", "--tensor-parallel-size", "2"])
    );

    expect(cmd).toContain("--no-enable-sleep-mode");
    expect(cmd).not.toContain("--enable-sleep-mode");
  });

  test("every other entry still gets sleep mode", async () => {
    const { cmd } = await spawnCall(vllmConfig(["--max-model-len", "8192"]));

    expect(cmd).toContain("--enable-sleep-mode");
  });
});

describe("FreeToken command line", () => {
  const freetokenConfig: ModelConfig = {
    name: "edge-moe",
    modelPath: "vendor/Some-MoE-284B",
    backend: "freetoken",
    gpu: 1,
    idleTimeout: 600_000,
    extraArgs: ["--memory-ratio", "0.85", "--moe-backend", "offload"],
  };

  test("serves the configured model on the loopback port", async () => {
    const { cmd } = await spawnCall(freetokenConfig);

    expect(cmd.join(" ")).toContain(
      `serve --model vendor/Some-MoE-284B --host 127.0.0.1 --port ${SCRATCH_PORT}`
    );
    // The MoE knobs are the user's, and reach ft untouched.
    expect(cmd.join(" ")).toEndWith("--memory-ratio 0.85 --moe-backend offload");
  });

  test("is pinned to the GPU its entry names", async () => {
    const { env } = await spawnCall(freetokenConfig);

    expect(env.CUDA_VISIBLE_DEVICES).toBe("1");
  });
});
